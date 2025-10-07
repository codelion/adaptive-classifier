import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional, Tuple, Any, Set, Union
import logging
import copy
from pathlib import Path
from safetensors.torch import save_file, load_file
import json
from sklearn.cluster import KMeans
from huggingface_hub import ModelHubMixin, hf_hub_download
import os
import shutil

from .models import Example, AdaptiveHead, ModelConfig
from .memory import PrototypeMemory
from .ewc import EWC
from .strategic import (
    StrategicCostFunction, CostFunctionFactory, StrategicOptimizer, StrategicEvaluator
)


logger = logging.getLogger(__name__)

class AdaptiveClassifier(ModelHubMixin):
    """A flexible classifier that can adapt to new classes and examples."""
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42,  # Add seed parameter
        use_onnx: Optional[Union[bool, str]] = "auto"  # "auto", True, False
    ):
        """Initialize the adaptive classifier.

        Args:
            model_name: Name of the HuggingFace transformer model
            device: Device to run the model on (default: auto-detect)
            config: Optional configuration dictionary
            seed: Random seed for initialization
            use_onnx: Whether to use ONNX Runtime ("auto", True, False).
                     "auto" uses ONNX on CPU, PyTorch on GPU.
        """
        # Set seed for initialization
        torch.manual_seed(seed)
        self.config = ModelConfig(config)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Determine if we should use ONNX
        self.use_onnx = self._should_use_onnx(use_onnx)

        # Initialize transformer model and tokenizer
        if self.use_onnx:
            try:
                from optimum.onnxruntime import ORTModelForFeatureExtraction
                logger.info(f"Initializing ONNX model for {model_name}")
                self.model = ORTModelForFeatureExtraction.from_pretrained(
                    model_name,
                    export=True  # Auto-export to ONNX if not already in ONNX format
                )
                logger.info("Successfully loaded ONNX model")
            except ImportError:
                logger.warning(
                    "optimum[onnxruntime] not installed. Falling back to PyTorch. "
                    "Install with: pip install optimum[onnxruntime]"
                )
                self.use_onnx = False
                self.model = AutoModel.from_pretrained(model_name).to(self.device)
            except Exception as e:
                logger.warning(
                    f"Failed to load ONNX model: {e}. Falling back to PyTorch."
                )
                self.use_onnx = False
                self.model = AutoModel.from_pretrained(model_name).to(self.device)
        else:
            self.model = AutoModel.from_pretrained(model_name).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize memory system
        self.embedding_dim = self.model.config.hidden_size
        self.memory = PrototypeMemory(
            self.embedding_dim,
            config=self.config
        )
        
        # Initialize adaptive head
        self.adaptive_head = None
        
        # Label mappings
        self.label_to_id = {}
        self.id_to_label = {}
        
        # Statistics
        self.train_steps = 0
        self.training_history = {}  # Track cumulative training examples per class
        
        # Strategic classification components
        self.strategic_cost_function = None
        self.strategic_optimizer = None
        self.strategic_evaluator = None
        
        # Initialize strategic components if enabled
        if self.config.enable_strategic_mode:
            self._initialize_strategic_components()

    def _should_use_onnx(self, use_onnx: Union[bool, str]) -> bool:
        """Determine if ONNX should be used based on configuration and device.

        Args:
            use_onnx: User preference ("auto", True, False)

        Returns:
            True if ONNX should be used, False otherwise
        """
        if use_onnx == "auto":
            # Auto-detect: Use ONNX on CPU, PyTorch on GPU
            return self.device == "cpu"
        elif isinstance(use_onnx, bool):
            return use_onnx
        else:
            logger.warning(f"Invalid use_onnx value: {use_onnx}. Using auto-detection.")
            return self.device == "cpu"

    def add_examples(self, texts: List[str], labels: List[str]):
        """Add new examples with special handling for new classes."""
        if not texts or not labels:
            raise ValueError("Empty input lists")
        if len(texts) != len(labels):
            raise ValueError("Mismatched text and label lists")

        # Check if classifier has any existing classes (before updating mappings)
        has_existing_classes = len(self.label_to_id) > 0

        # Check for new classes
        new_classes = set(labels) - set(self.label_to_id.keys())
        is_adding_new_classes = len(new_classes) > 0

        # Update label mappings - sort new classes alphabetically for consistent IDs
        for label in sorted(new_classes):
            idx = len(self.label_to_id)
            self.label_to_id[label] = idx
            self.id_to_label[idx] = label

        # Get embeddings for all texts
        embeddings = self._get_embeddings(texts)

        # Add examples to memory and update training history
        for text, embedding, label in zip(texts, embeddings, labels):
            example = Example(text, label, embedding)
            self.memory.add_example(example, label)

            # Update training history
            if label not in self.training_history:
                self.training_history[label] = 0
            self.training_history[label] += 1

        # Determine training strategy: only use special new class handling for incremental learning
        is_incremental_learning = is_adding_new_classes and has_existing_classes

        if is_incremental_learning:
            # Adding new classes to existing classifier - use special handling
            # Store old head for EWC before modifying structure
            old_head = copy.deepcopy(self.adaptive_head) if self.adaptive_head is not None else None

            # Expand existing head to accommodate new classes (preserves weights)
            num_classes = len(self.label_to_id)
            self.adaptive_head.update_num_classes(num_classes)
            # Move to correct device after update
            self.adaptive_head = self.adaptive_head.to(self.device)

            # Train with focus on new classes
            self._train_new_classes(old_head, new_classes)
        else:
            # Initial training or regular updates - use normal training
            # Initialize head if needed
            if self.adaptive_head is None:
                self._initialize_adaptive_head()
            elif is_adding_new_classes:
                # Edge case: expanding head for new classes but treating as regular training
                num_classes = len(self.label_to_id)
                self.adaptive_head.update_num_classes(num_classes)
                self.adaptive_head = self.adaptive_head.to(self.device)

            # Regular training
            self._train_adaptive_head()

            # Strategic training step if enabled
            if self.strategic_mode and self.train_steps % self.config.strategic_training_frequency == 0:
                self._perform_strategic_training()

        # Ensure FAISS index is up to date after adding examples
        self.memory._rebuild_index()
    
    def _train_new_classes(self, old_head: Optional[nn.Module], new_classes: Set[str]):
        """Train the model with focus on new classes while preserving old class knowledge."""
        if not self.memory.examples:
            return
        
        # Prepare training data with balanced sampling
        all_embeddings = []
        all_labels = []
        examples_per_class = {}
        
        # Count examples per class
        for label in self.memory.examples:
            examples_per_class[label] = len(self.memory.examples[label])
        
        # Improved sampling strategy for many-class scenarios
        min_examples = min(examples_per_class.values())
        max_examples = max(examples_per_class.values())

        # For many-class scenarios, use a more balanced approach
        num_classes = len(examples_per_class)
        target_samples_per_class = max(5, min(10, min_examples * 2))  # Adaptive target

        if num_classes > 20:  # Many-class scenario
            # Use stratified sampling to ensure all classes get representation
            for label, examples in self.memory.examples.items():
                if label in new_classes:
                    # Give new classes more representation, but not excessive
                    num_samples = min(len(examples), target_samples_per_class * 2)
                else:
                    # Ensure old classes maintain representation
                    num_samples = min(len(examples), target_samples_per_class)

                # Sample without replacement first, then with if needed
                if num_samples <= len(examples):
                    indices = np.random.choice(len(examples), size=num_samples, replace=False)
                else:
                    indices = np.random.choice(len(examples), size=num_samples, replace=True)

                for idx in indices:
                    example = examples[idx]
                    all_embeddings.append(example.embedding)
                    all_labels.append(self.label_to_id[label])
        else:
            # Original strategy for fewer classes
            sampling_weights = {}

            for label, count in examples_per_class.items():
                if label in new_classes:
                    # Oversample new classes
                    sampling_weights[label] = 2.0
                else:
                    # Sample old classes proportionally
                    sampling_weights[label] = min_examples / count

            # Sample examples with weights
            for label, examples in self.memory.examples.items():
                weight = sampling_weights[label]
                num_samples = max(min_examples, int(len(examples) * weight))

                # Randomly sample with replacement if needed
                indices = np.random.choice(
                    len(examples),
                    size=num_samples,
                    replace=num_samples > len(examples)
                )

                for idx in indices:
                    example = examples[idx]
                    all_embeddings.append(example.embedding)
                    all_labels.append(self.label_to_id[label])
        
        all_embeddings = torch.stack(all_embeddings)
        all_labels = torch.tensor(all_labels)
        
        # Create dataset and initialize EWC with lower penalty for new classes
        dataset = torch.utils.data.TensorDataset(all_embeddings, all_labels)

        ewc = None
        if old_head is not None:
            # Create a dataset for EWC that only includes examples from old classes
            old_embeddings = []
            old_labels = []
            old_label_to_id = {label: idx for idx, label in enumerate(self.id_to_label.values())
                               if label not in new_classes}

            for label, examples in self.memory.examples.items():
                if label not in new_classes:  # Only old classes
                    for example in examples[:5]:  # Limit to representative examples
                        old_embeddings.append(example.embedding)
                        old_labels.append(old_label_to_id[label])

            if old_embeddings:  # Only create EWC if we have old examples
                old_embeddings = torch.stack(old_embeddings)
                old_labels = torch.tensor(old_labels, dtype=torch.long)
                old_dataset = torch.utils.data.TensorDataset(old_embeddings, old_labels)

                ewc = EWC(
                    old_head,
                    old_dataset,
                    device=self.device,
                    ewc_lambda=5.0  # Balanced EWC penalty
                )
        
        # Training setup
        self.adaptive_head.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.adaptive_head.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        
        # Create data loader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            generator=torch.Generator().manual_seed(42)
        )
        
        # Training loop
        best_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(15):  # More epochs for new classes
            total_loss = 0
            for batch_embeddings, batch_labels in loader:
                batch_embeddings = batch_embeddings.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.adaptive_head(batch_embeddings)
                
                # Compute task loss
                task_loss = criterion(outputs, batch_labels)
                
                # Add EWC loss if applicable
                if ewc is not None:
                    ewc_loss = ewc.ewc_loss(batch_size=len(batch_embeddings))
                    loss = task_loss + ewc_loss
                else:
                    loss = task_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.adaptive_head.parameters(),
                    max_norm=1.0
                )
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.debug(f"Early stopping at epoch {epoch + 1}")
                    break
        
        self.train_steps += 1
    
    def _perform_strategic_training(self):
        """Perform strategic training on current examples."""
        if not self.strategic_mode or not self.memory.examples:
            return
        
        # Prepare training data
        all_embeddings = []
        all_labels = []
        
        for label in self.memory.examples:
            for example in self.memory.examples[label]:
                all_embeddings.append(example.embedding)
                all_labels.append(self.label_to_id[label])
        
        if all_embeddings:
            all_embeddings = torch.stack(all_embeddings)
            all_labels = torch.tensor(all_labels, dtype=torch.long, device=self.device)
            
            # Perform strategic training step
            self._strategic_training_step(all_embeddings, all_labels)
            
            logger.debug("Performed strategic training step")
    
    def predict(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Predict with dual prediction system - blends strategic and regular predictions.
        
        If no cost function is provided, uses existing prediction logic (zero changes).
        If cost function is provided, blends strategic and regular predictions.
        
        Args:
            text: Input text to classify
            k: Number of top predictions to return
            
        Returns:
            List of (label, confidence) tuples
        """
        if not text:
            raise ValueError("Empty input text")
        
        # If strategic mode is not enabled, use regular prediction
        if not self.strategic_mode:
            return self._predict_regular(text, k)
        
        # Dual prediction system: blend strategic and regular predictions
        return self._predict_dual(text, k)
    
    def _predict_regular(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Regular prediction logic (original implementation)."""
        # Ensure deterministic behavior
        with torch.no_grad():
            # Get embedding
            embedding = self._get_embeddings([text])[0]
            
            # Get prototype predictions for ALL classes (not limited by k)
            # This ensures complete scoring information for proper combination
            max_classes = len(self.id_to_label) if self.id_to_label else k
            proto_preds = self.memory.get_nearest_prototypes(embedding, k=max_classes)

            # Get neural predictions if available for ALL classes (not limited by k)
            if self.adaptive_head is not None:
                self.adaptive_head.eval()  # Ensure eval mode
                # Add batch dimension and move to device
                input_embedding = embedding.unsqueeze(0).to(self.device)
                logits = self.adaptive_head(input_embedding)
                # Squeeze batch dimension
                logits = logits.squeeze(0)
                probs = F.softmax(logits, dim=0)

                # Get predictions for ALL classes for proper scoring combination
                values, indices = torch.topk(probs, len(self.id_to_label))
                head_preds = [
                    (self.id_to_label[idx.item()], val.item())
                    for val, idx in zip(values, indices)
                ]
            else:
                head_preds = []
        
        # Combine predictions with adjusted weights
        combined_scores = {}
        
        # Use training history to determine weights
        for label, score in proto_preds:
            # Check training history instead of current storage
            trained_examples = self.training_history.get(label, 0)
            if trained_examples < 10:
                # For newer classes (fewer training examples), trust neural predictions more
                weight = 0.3  # Lower prototype weight for new classes
            else:
                weight = 0.7  # Higher prototype weight for established classes
            combined_scores[label] = score * weight
        
        for label, score in head_preds:
            # Use training history for neural weights too
            trained_examples = self.training_history.get(label, 0)
            if trained_examples < 10:
                weight = 0.7  # Higher neural weight for new classes
            else:
                weight = 0.3  # Lower neural weight for established classes
            combined_scores[label] = combined_scores.get(label, 0) + score * weight
        
        # Normalize scores
        predictions = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        total = sum(score for _, score in predictions)
        if total > 0:
            predictions = [(label, score/total) for label, score in predictions]
        
        return predictions[:k]
    
    def _predict_dual(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Dual prediction system that blends strategic and regular predictions."""
        # Get regular predictions
        regular_preds = self._predict_regular(text, k)
        
        # Get strategic predictions
        strategic_preds = self.predict_strategic(text, k)
        
        # Blend predictions based on configuration
        blended_scores = {}
        
        # Weight for blending (configurable)
        regular_weight = self.config.strategic_blend_regular_weight
        strategic_weight = self.config.strategic_blend_strategic_weight
        
        # Combine regular predictions
        for label, score in regular_preds:
            blended_scores[label] = score * regular_weight
        
        # Combine strategic predictions
        for label, score in strategic_preds:
            blended_scores[label] = blended_scores.get(label, 0) + score * strategic_weight
        
        # Sort and normalize
        blended_predictions = sorted(
            blended_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Normalize scores
        total = sum(score for _, score in blended_predictions)
        if total > 0:
            blended_predictions = [
                (label, score / total) for label, score in blended_predictions
            ]
        
        # Log dual prediction for debugging
        logger.debug(f"Dual prediction - Regular: {regular_preds[:3]}, Strategic: {strategic_preds[:3]}, Blended: {blended_predictions[:3]}")
        
        return blended_predictions[:k]
    
    def _save_pretrained(
        self,
        save_directory: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
        include_onnx: bool = True,
        quantize_onnx: bool = True,
        **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Save the model to a directory.

        Args:
            save_directory: Directory to save the model to
            config: Optional additional configuration
            include_onnx: Whether to include ONNX export (default: True)
            quantize_onnx: Whether to quantize ONNX model (requires include_onnx=True)
            **kwargs: Additional arguments passed to save_pretrained

        Returns:
            Tuple of (dict of filenames, dict of objects to save)
        """
        save_directory = Path(save_directory)
        os.makedirs(save_directory, exist_ok=True)

        # Save configuration and metadata
        config_dict = {
            'model_name': self.model.config._name_or_path,
            'embedding_dim': self.embedding_dim,
            'label_to_id': self.label_to_id,
            'id_to_label': {str(k): v for k, v in self.id_to_label.items()},
            'train_steps': self.train_steps,
            'training_history': self.training_history,  # Save cumulative training counts
            'config': self.config.to_dict(),
            'library_name': 'adaptive-classifier'  # Tell HuggingFace Hub this requires the adaptive-classifier library
        }

        # Save examples in a separate file to keep config clean
        saved_examples = {}
        for label, examples in self.memory.examples.items():
            saved_examples[label] = [
                ex.to_dict() for ex in
                self.select_representative_examples(
                    examples, k=self.config.num_representative_examples)
            ]

        # Save model tensors
        tensor_dict = {}
        
        # Save prototypes
        for label, proto in self.memory.prototypes.items():
            tensor_dict[f'prototype_{label}'] = proto
        
        # Save adaptive head if it exists
        if self.adaptive_head is not None:
            for name, param in self.adaptive_head.state_dict().items():
                tensor_dict[f'adaptive_head_{name}'] = param

        # Save files
        config_file = save_directory / "config.json"
        examples_file = save_directory / "examples.json"
        tensors_file = save_directory / "model.safetensors"

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)
            
        with open(examples_file, "w", encoding="utf-8") as f:
            json.dump(saved_examples, f, indent=2, sort_keys=True)

        save_file(tensor_dict, tensors_file)

        # Generate model card if it doesn't exist
        model_card_path = save_directory / "README.md"
        if not model_card_path.exists():
            model_card_content = self._generate_model_card()
            with open(model_card_path, "w", encoding="utf-8") as f:
                f.write(model_card_content)

        # Export ONNX if requested
        if include_onnx:
            try:
                onnx_dir = save_directory / "onnx"
                self.export_onnx(
                    onnx_dir,
                    quantize=quantize_onnx
                )
                logger.info(f"ONNX model exported to {onnx_dir}")
            except ImportError:
                logger.warning(
                    "Skipping ONNX export: optimum[onnxruntime] not installed. "
                    "Install with: pip install optimum[onnxruntime]"
                )
            except Exception as e:
                logger.warning(f"Skipping ONNX export due to error: {e}")

        # Return files that were created
        saved_files = {
            "config": config_file.name,
            "examples": examples_file.name,
            "model": tensors_file.name,
            "model_card": model_card_path.name,
        }

        if include_onnx and (save_directory / "onnx").exists():
            saved_files["onnx"] = "onnx/"

        return saved_files, {}

    @classmethod
    def _from_pretrained(
        cls,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        use_onnx: Optional[Union[bool, str]] = "auto",
        prefer_quantized: bool = True,
        **kwargs
    ) -> "AdaptiveClassifier":
        """Load a model from the HuggingFace Hub or local directory.

        Args:
            model_id: HuggingFace Hub model ID or path to local directory
            revision: Revision of the model on the Hub
            cache_dir: Cache directory for downloaded models
            force_download: Force download of models
            proxies: Proxies to use for downloading
            resume_download: Resume downloading if interrupted
            local_files_only: Use local files only, don't download
            token: Authentication token for Hub
            use_onnx: Whether to use ONNX Runtime ("auto", True, False)
            prefer_quantized: Use quantized ONNX model if available (default: True)
                             Set to False to use unquantized model for maximum accuracy
            **kwargs: Additional arguments passed to from_pretrained

        Returns:
            Loaded AdaptiveClassifier instance

        Examples:
            >>> # Load with quantized ONNX (default - faster, smaller)
            >>> classifier = AdaptiveClassifier.load("adaptive-classifier/llm-router")
            >>>
            >>> # Load with unquantized ONNX (maximum accuracy)
            >>> classifier = AdaptiveClassifier.load("adaptive-classifier/llm-router", prefer_quantized=False)
            >>>
            >>> # Force PyTorch (no ONNX)
            >>> classifier = AdaptiveClassifier.load("adaptive-classifier/llm-router", use_onnx=False)
        """
       
        # Check if model_id is a local directory
        model_path = Path(model_id)
        try:
            if model_path.is_dir() and (model_path / "config.json").exists():
                # Local directory with required files
                pass
            else:
                # Download files from HuggingFace Hub
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename="config.json",
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                model_path = Path(os.path.dirname(config_file))
                
                # Download examples file
                hf_hub_download(
                    repo_id=model_id,
                    filename="examples.json",
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                
                # Download model file
                hf_hub_download(
                    repo_id=model_id,
                    filename="model.safetensors",
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )

                # Try to download ONNX files if they exist
                try:
                    # Download quantized ONNX model (primary)
                    hf_hub_download(
                        repo_id=model_id,
                        filename="onnx/model_quantized.onnx",
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        token=token,
                        local_files_only=local_files_only,
                    )
                    # Download ONNX config files
                    for onnx_file in ["config.json", "ort_config.json", "tokenizer.json",
                                      "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]:
                        try:
                            hf_hub_download(
                                repo_id=model_id,
                                filename=f"onnx/{onnx_file}",
                                revision=revision,
                                cache_dir=cache_dir,
                                force_download=force_download,
                                proxies=proxies,
                                resume_download=resume_download,
                                token=token,
                                local_files_only=local_files_only,
                            )
                        except:
                            pass  # Some files might not exist
                    logger.info("Downloaded ONNX model files from Hub")
                except Exception as e:
                    logger.debug(f"ONNX model not available on Hub: {e}")
        except Exception as e:
            raise ValueError(f"Error loading model from {model_id}: {e}")

        # Load configuration
        with open(model_path / "config.json", "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        # Load examples
        with open(model_path / "examples.json", "r", encoding="utf-8") as f:
            saved_examples = json.load(f)

        # Check if ONNX model exists (quantized or unquantized)
        onnx_path = model_path / "onnx"
        has_onnx = onnx_path.exists() and ((onnx_path / "model_quantized.onnx").exists() or (onnx_path / "model.onnx").exists())

        # Determine if we should use ONNX
        final_use_onnx = use_onnx
        if use_onnx == "auto":
            device = kwargs.get("device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
            # Use ONNX if available and on CPU
            final_use_onnx = has_onnx and device == "cpu"
        elif use_onnx is True and not has_onnx:
            logger.warning(
                "ONNX model requested but not found in save directory. "
                "Loading PyTorch model instead."
            )
            final_use_onnx = False

        # Initialize classifier
        device = kwargs.get("device", None)

        # If loading ONNX from save directory, use a special path
        if final_use_onnx and has_onnx:
            # Load ONNX model from saved onnx directory
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            logger.info(f"Loading ONNX model from {onnx_path}")

            # Create a temporary classifier with ONNX disabled first
            classifier = cls.__new__(cls)
            torch.manual_seed(42)
            classifier.config = ModelConfig(config_dict.get('config', None))
            classifier.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            classifier.use_onnx = True

            # Load ONNX model (prefer quantized by default)
            # Check which ONNX files exist
            has_quantized = (onnx_path / "model_quantized.onnx").exists()
            has_unquantized = (onnx_path / "model.onnx").exists()

            # Determine which file to load
            if prefer_quantized and has_quantized:
                onnx_file = "model_quantized.onnx"
                logger.info("Loading quantized ONNX model for optimal performance")
            elif has_unquantized:
                onnx_file = "model.onnx"
                logger.info("Loading unquantized ONNX model")
            elif has_quantized:
                onnx_file = "model_quantized.onnx"
                logger.info("Loading quantized ONNX model (only version available)")
            else:
                raise ValueError(f"No ONNX model files found in {onnx_path}")

            classifier.model = ORTModelForFeatureExtraction.from_pretrained(
                onnx_path,
                file_name=onnx_file
            )
            classifier.tokenizer = AutoTokenizer.from_pretrained(config_dict['model_name'])

            # Initialize memory and other components
            classifier.embedding_dim = classifier.model.config.hidden_size
            classifier.memory = PrototypeMemory(
                classifier.embedding_dim,
                config=classifier.config
            )
            classifier.adaptive_head = None
            classifier.label_to_id = {}
            classifier.id_to_label = {}
            classifier.train_steps = 0
            classifier.training_history = {}
            classifier.strategic_cost_function = None
            classifier.strategic_optimizer = None
            classifier.strategic_evaluator = None

            # Initialize subclass-specific attributes (e.g., for MultiLabelAdaptiveClassifier)
            # These will be overwritten if the subclass has its own initialization logic
            if not hasattr(classifier, 'default_threshold'):
                classifier.default_threshold = 0.5
            if not hasattr(classifier, 'min_predictions'):
                classifier.min_predictions = 1
            if not hasattr(classifier, 'max_predictions'):
                classifier.max_predictions = None
            if not hasattr(classifier, 'label_thresholds'):
                classifier.label_thresholds = {}

            if classifier.config.enable_strategic_mode:
                classifier._initialize_strategic_components()
        else:
            # Standard initialization
            classifier = cls(
                config_dict['model_name'],
                device=device,
                config=config_dict.get('config', None),
                use_onnx=final_use_onnx if isinstance(final_use_onnx, bool) else False
            )

        # Restore label mappings
        classifier.label_to_id = config_dict['label_to_id']
        classifier.id_to_label = {
            int(k): v for k, v in config_dict['id_to_label'].items()
        }
        classifier.train_steps = config_dict['train_steps']
        
        # Restore training history with backward compatibility
        classifier.training_history = config_dict.get('training_history', {})

        # Load tensors
        tensors = load_file(model_path / "model.safetensors")

        # Restore saved examples
        for label, examples_data in saved_examples.items():
            classifier.memory.examples[label] = [
                Example.from_dict(ex_data) for ex_data in examples_data
            ]

        # Restore prototypes
        for label in classifier.label_to_id.keys():
            prototype_key = f'prototype_{label}'
            if prototype_key in tensors:
                prototype = tensors[prototype_key]
                classifier.memory.prototypes[label] = prototype

        # Rebuild memory system
        classifier.memory._restore_from_save()

        # Restore adaptive head if it exists
        adaptive_head_params = {
            k.replace('adaptive_head_', ''): v 
            for k, v in tensors.items() 
            if k.startswith('adaptive_head_')
        }

        if adaptive_head_params:
            classifier._initialize_adaptive_head()
            classifier.adaptive_head.load_state_dict(adaptive_head_params)

        # Backward compatibility: estimate training history if not present
        if not classifier.training_history:
            for label, examples in saved_examples.items():
                # Estimate based on saved examples (default saves 5, typical training uses 100+)
                # Using 20x multiplier as reasonable estimate
                classifier.training_history[label] = len(examples) * 20

        return classifier

    def _generate_model_card(self) -> str:
        """Generate a model card for the classifier.
        
        Returns:
            Model card content as string
        """
        stats = self.get_memory_stats()
        
        model_card = f"""---
language: multilingual
tags:
- adaptive-classifier
- text-classification
- continuous-learning
license: apache-2.0
---

# Adaptive Classifier

This model is an instance of an [adaptive-classifier](https://github.com/codelion/adaptive-classifier) that allows for continuous learning and dynamic class addition.

## Installation

**IMPORTANT:** To use this model, you must first install the `adaptive-classifier` library. You do **NOT** need `trust_remote_code=True`.

```bash
pip install adaptive-classifier
```

## Model Details

- Base Model: {self.model.config._name_or_path}
- Number of Classes: {stats['num_classes']}
- Total Examples: {stats['total_examples']}
- Embedding Dimension: {self.embedding_dim}

## Class Distribution

```
{self._format_class_distribution(stats)}
```

## Usage

After installing the `adaptive-classifier` library, you can load and use this model:

```python
from adaptive_classifier import AdaptiveClassifier

# Load the model (no trust_remote_code needed!)
classifier = AdaptiveClassifier.from_pretrained("adaptive-classifier/model-name")

# Make predictions
text = "Your text here"
predictions = classifier.predict(text)
print(predictions)  # List of (label, confidence) tuples

# Add new examples for continuous learning
texts = ["Example 1", "Example 2"]
labels = ["class1", "class2"]
classifier.add_examples(texts, labels)
```

**Note:** This model uses the `adaptive-classifier` library distributed via PyPI. You do **NOT** need to set `trust_remote_code=True` - just install the library first.

## Training Details

- Training Steps: {self.train_steps}
- Examples per Class: See distribution above
- Prototype Memory: Active
- Neural Adaptation: {"Active" if self.adaptive_head is not None else "Inactive"}

## Limitations

This model:
- Requires at least {self.config.min_examples_per_class} examples per class
- Has a maximum of {self.config.max_examples_per_class} examples per class
- Updates prototypes every {self.config.prototype_update_frequency} examples

## Citation

```bibtex
@software{{adaptive_classifier,
  title = {{Adaptive Classifier: Dynamic Text Classification with Continuous Learning}},
  author = {{Sharma, Asankhaya}},
  year = {{2025}},
  publisher = {{GitHub}},
  url = {{https://github.com/codelion/adaptive-classifier}}
}}
```
"""
        return model_card

    def _format_class_distribution(self, stats: Dict[str, Any]) -> str:
        """Format class distribution for model card.
        
        Args:
            stats: Statistics from get_memory_stats()
            
        Returns:
            Formatted string of class distribution
        """
        if 'examples_per_class' not in stats:
            return "No examples stored"
            
        lines = []
        total = sum(stats['examples_per_class'].values())
        
        for label, count in sorted(stats['examples_per_class'].items()):
            percentage = (count / total) * 100 if total > 0 else 0
            lines.append(f"{label}: {count} examples ({percentage:.1f}%)")
            
        return "\n".join(lines)

    def export_onnx(
        self,
        save_directory: Union[str, Path],
        quantize: bool = False,
        quantization_config: Optional[str] = "arm64"
    ) -> Path:
        """Export the transformer model to ONNX format.

        Args:
            save_directory: Directory to save ONNX model
            quantize: Whether to apply INT8 quantization
            quantization_config: Quantization configuration ("arm64", "avx512", "avx2")

        Returns:
            Path to the saved ONNX model directory

        Raises:
            ImportError: If optimum[onnxruntime] is not installed
            ValueError: If model is already in ONNX format
        """
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig
        except ImportError:
            raise ImportError(
                "optimum[onnxruntime] is required for ONNX export. "
                "Install with: pip install optimum[onnxruntime]"
            )

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Check if already ONNX
        if self.use_onnx:
            logger.warning("Model is already in ONNX format. Saving current model.")
            self.model.save_pretrained(save_directory)
            return save_directory

        # Get the base model name
        model_name = self.model.config._name_or_path

        logger.info(f"Exporting {model_name} to ONNX format...")

        # Export PyTorch model to ONNX
        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            model_name,
            export=True
        )

        # Always save unquantized version first
        ort_model.save_pretrained(save_directory)
        logger.info(f"Saved unquantized ONNX model to {save_directory}")

        if quantize:
            logger.info(f"Applying {quantization_config} INT8 quantization...")

            # Select quantization config
            if quantization_config == "arm64":
                qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
            elif quantization_config == "avx512":
                qconfig = AutoQuantizationConfig.avx512(is_static=False, per_channel=False)
            elif quantization_config == "avx2":
                qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
            else:
                logger.warning(f"Unknown quantization config: {quantization_config}. Using arm64.")
                qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)

            # Apply quantization (saves quantized version alongside unquantized)
            quantizer = ORTQuantizer.from_pretrained(ort_model)
            quantizer.quantize(
                save_dir=save_directory,
                quantization_config=qconfig
            )
            logger.info(f"Saved quantized ONNX model to {save_directory}")

        logger.info(f"ONNX model exported to {save_directory}")
        return save_directory

    def push_to_hub(
        self,
        repo_id: str,
        include_onnx: bool = True,
        quantize_onnx: bool = True,
        token: Optional[str] = None,
        commit_message: Optional[str] = None,
        private: bool = False,
        **kwargs
    ):
        """Push model to HuggingFace Hub with ONNX export by default.

        Args:
            repo_id: Repository ID on HuggingFace Hub (e.g., "username/model-name")
            include_onnx: Whether to include ONNX version of the model (default: True)
            quantize_onnx: Whether to quantize the ONNX model (requires include_onnx=True)
            token: HuggingFace Hub authentication token (or set HF_TOKEN env var)
            commit_message: Commit message for the push
            private: Whether to create a private repository
            **kwargs: Additional arguments passed to HfApi.upload_folder

        Examples:
            >>> classifier.push_to_hub("my-org/my-classifier")  # ONNX included by default
            >>> classifier.push_to_hub("my-org/my-classifier", quantize_onnx=True)
            >>> classifier.push_to_hub("my-org/my-classifier", include_onnx=False)  # Opt-out
        """
        import tempfile
        import os
        from huggingface_hub import HfApi

        # Get token from parameter or environment
        token = token or os.environ.get("HF_TOKEN")
        if not token:
            logger.warning(
                "No HuggingFace token provided. Set HF_TOKEN environment variable or pass token parameter. "
                "You may need to login with `huggingface-cli login`"
            )

        # Create temporary directory for saving
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)

            # Save model with optional ONNX
            self._save_pretrained(
                save_path,
                include_onnx=include_onnx,
                quantize_onnx=quantize_onnx
            )

            # Use HfApi to upload the folder directly
            api = HfApi()

            # Create repo if it doesn't exist
            try:
                api.create_repo(
                    repo_id=repo_id,
                    token=token,
                    private=private,
                    exist_ok=True
                )
            except Exception as e:
                logger.warning(f"Could not create repo (may already exist): {e}")

            # Upload all files from the temp directory
            commit_info = api.upload_folder(
                folder_path=str(save_path),
                repo_id=repo_id,
                token=token,
                commit_message=commit_message or "Upload model with adaptive-classifier",
                **kwargs
            )

            logger.info(f"Successfully pushed model to https://huggingface.co/{repo_id}")
            return f"https://huggingface.co/{repo_id}"

    # Keep existing save/load methods for backwards compatibility
    def save(self, save_dir: str, include_onnx: bool = True, quantize_onnx: bool = True):
        """Legacy save method for backwards compatibility.

        Args:
            save_dir: Directory to save to
            include_onnx: Whether to include ONNX export (default: True)
            quantize_onnx: Whether to quantize ONNX model
        """
        return self._save_pretrained(
            save_dir,
            include_onnx=include_onnx,
            quantize_onnx=quantize_onnx
        )

    @classmethod
    def load(cls, save_dir: str, device: Optional[str] = None, use_onnx: Optional[Union[bool, str]] = "auto", prefer_quantized: bool = True) -> 'AdaptiveClassifier':
        """Legacy load method for backwards compatibility.

        Args:
            save_dir: Directory to load from
            device: Device to load model on
            use_onnx: Whether to use ONNX Runtime ("auto", True, False)
            prefer_quantized: Use quantized ONNX model if available (default: True)
        """
        kwargs = {}
        if device is not None:
            kwargs['device'] = device
        return cls._from_pretrained(save_dir, use_onnx=use_onnx, prefer_quantized=prefer_quantized, **kwargs)
    
    def to(self, device: str) -> 'AdaptiveClassifier':
        """Move the model to specified device.
        
        Args:
            device: Device to move to ("cuda" or "cpu")
            
        Returns:
            Self for chaining
        """
        self.device = device
        self.model = self.model.to(device)
        if self.adaptive_head is not None:
            self.adaptive_head = self.adaptive_head.to(device)
        return self
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Dictionary of memory statistics
        """
        return self.memory.get_stats()
    
    def _initialize_adaptive_head(self):
        """Initialize or reinitialize the adaptive head with improved configuration."""
        num_classes = len(self.label_to_id)
        hidden_dims = [self.embedding_dim, self.embedding_dim // 2]
        
        self.adaptive_head = AdaptiveHead(
            self.embedding_dim,
            num_classes,
            hidden_dims=hidden_dims
        ).to(self.device)

    def _get_embeddings(self, texts: List[str]) -> List[torch.Tensor]:
        """Get embeddings for input texts."""
        # Temporarily set model to eval mode (only for PyTorch models)
        was_training = False
        if not self.use_onnx and hasattr(self.model, 'training'):
            was_training = self.model.training
            self.model.eval()

        # Get embeddings
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                max_length=self.config.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            # For ONNX models, inputs don't need to be moved to device
            if not self.use_onnx:
                inputs = inputs.to(self.device)

            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]

            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Restore original training mode (only for PyTorch models)
        if was_training and hasattr(self.model, 'train'):
            self.model.train()

        # Return embeddings as list
        return [emb.cpu() for emb in embeddings]

    def get_example_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored examples and model state."""
        stats = {
            'total_examples': sum(len(exs) for exs in self.memory.examples.values()),
            'examples_per_class': {
                label: len(exs) for label, exs in self.memory.examples.items()
            },
            'num_classes': len(self.label_to_id),
            'train_steps': self.train_steps,
            'memory_usage': {
                'prototypes': sum(p.nelement() * p.element_size() 
                                for p in self.memory.prototypes.values()),
                'examples': sum(sum(ex.embedding.nelement() * ex.embedding.element_size() 
                                  for ex in exs) 
                              for exs in self.memory.examples.values())
            }
        }
        
        if self.adaptive_head is not None:
            stats['model_params'] = sum(p.nelement() for p in 
                                      self.adaptive_head.parameters())
        
        return stats

    def predict_batch(
        self,
        texts: List[str],
        k: int = 5,
        batch_size: int = 32
    ) -> List[List[Tuple[str, float]]]:
        """Predict labels for a batch of texts with improved batching."""
        if not texts:
            raise ValueError("Empty input batch")
        
        all_predictions = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Get embeddings for batch
            batch_embeddings = self._get_embeddings(batch_texts)
            
            # Get predictions for each embedding
            batch_predictions = []
            for embedding in batch_embeddings:
                # Get prototype predictions
                proto_preds = self.memory.get_nearest_prototypes(
                    embedding,
                    k=k
                )
                
                # Get neural predictions if available
                if self.adaptive_head is not None:
                    self.adaptive_head.eval()
                    with torch.no_grad():
                        # Add batch dimension and move to device
                        input_embedding = embedding.unsqueeze(0).to(self.device)
                        logits = self.adaptive_head(input_embedding)
                        # Squeeze batch dimension
                        logits = logits.squeeze(0)
                        probs = F.softmax(logits, dim=0)
                        
                        values, indices = torch.topk(
                            probs,
                            min(k, len(self.id_to_label))
                        )
                        head_preds = [
                            (self.id_to_label[idx.item()], val.item())
                            for val, idx in zip(values, indices)
                        ]
                else:
                    head_preds = []
                
                # Combine predictions
                combined_scores = {}
                proto_weight = 0.7  # More weight to prototypes
                head_weight = 0.3   # Less weight to neural network
                
                for label, score in proto_preds:
                    combined_scores[label] = score * proto_weight
                    
                for label, score in head_preds:
                    combined_scores[label] = (
                        combined_scores.get(label, 0) + score * head_weight
                    )
                
                # Sort and normalize predictions
                predictions = sorted(
                    combined_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Normalize scores
                total = sum(score for _, score in predictions)
                if total > 0:
                    predictions = [(label, score/total) 
                                for label, score in predictions]
                
                batch_predictions.append(predictions[:k])
            
            all_predictions.extend(batch_predictions)
        
        return all_predictions

    def clear_memory(self, labels: Optional[List[str]] = None):
        """Clear memory for specified labels or all if none specified."""
        if labels is None:
            self.memory.clear()
        else:
            for label in labels:
                if label in self.memory.examples:
                    del self.memory.examples[label]
                if label in self.memory.prototypes:
                    del self.memory.prototypes[label]
            self.memory._rebuild_index()

    def merge_classifiers(self, other: 'AdaptiveClassifier') -> 'AdaptiveClassifier':
        """Merge another classifier into this one."""
        # Verify compatibility
        if self.embedding_dim != other.embedding_dim:
            raise ValueError("Classifiers have different embedding dimensions")
            
        # Merge label mappings
        next_idx = max(self.id_to_label.keys()) + 1
        for label in other.label_to_id:
            if label not in self.label_to_id:
                self.label_to_id[label] = next_idx
                self.id_to_label[next_idx] = label
                next_idx += 1
        
        # Merge examples and update prototypes
        for label, examples in other.memory.examples.items():
            for example in examples:
                self.memory.add_example(example, label)
        
        # Retrain adaptive head
        if self.adaptive_head is not None:
            self._initialize_adaptive_head()
            self._train_adaptive_head()
        
        return self
    
    def _train_adaptive_head(self, epochs: int = 10):
        """Train the adaptive head with improved stability."""
        if not self.memory.examples:
            return
            
        # Prepare training data
        all_embeddings = []
        all_labels = []
        
        # Sort examples for deterministic order
        for label in sorted(self.memory.examples.keys()):
            examples = sorted(self.memory.examples[label], key=lambda x: x.text)
            for example in examples:
                all_embeddings.append(example.embedding)
                # Convert string labels to numeric indices
                all_labels.append(self.label_to_id[example.label])
        
        all_embeddings = torch.stack(all_embeddings)
        # Ensure labels are Long tensor
        all_labels = torch.tensor(all_labels, dtype=torch.long, device=self.device)
        
        # Normalize embeddings for stable training
        all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
        
        # Create deterministic data loader
        dataset = torch.utils.data.TensorDataset(all_embeddings, all_labels)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(32, len(all_embeddings)),
            shuffle=True,
            generator=torch.Generator().manual_seed(42)
        )
        
        # Training setup
        self.adaptive_head.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.adaptive_head.parameters(),
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler for stability
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
        )
        
        best_loss = float('inf')
        patience_counter = 0
        patience = 3
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_embeddings, batch_labels in loader:
                batch_embeddings = batch_embeddings.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.adaptive_head(batch_embeddings)
                
                # Add shape debugging
                if epoch == 0 and total_loss == 0:  # Only for first batch of first epoch
                    logger.debug(f"outputs shape: {outputs.shape}")
                    logger.debug(f"batch_labels shape: {batch_labels.shape}")
                    logger.debug(f"batch_labels content: {batch_labels}")
                
                loss = criterion(outputs, batch_labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.adaptive_head.parameters(),
                    max_norm=1.0
                )
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.debug(f"Early stopping at epoch {epoch + 1}")
                    break
        
        self.train_steps += 1
    
    def _update_adaptive_head(self):
        """Update adaptive head for new classes."""
        num_classes = len(self.label_to_id)
        
        if self.adaptive_head is None:
            self._initialize_adaptive_head()
        elif num_classes > self.adaptive_head.model[-1].out_features:
            self.adaptive_head.update_num_classes(num_classes)

    def select_representative_examples(self, examples: List[Example], k: int = 5) -> List[Example]:
        """Select k most representative examples using k-means clustering.
        
        Args:
            examples: List of examples to select from
            k: Number of examples to select
            
        Returns:
            List of selected examples
        """
        if len(examples) <= k:
            return examples
            
        # Stack embeddings
        embeddings = torch.stack([ex.embedding for ex in examples])
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Use k-means to find centroids
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        kmeans.fit(embeddings.numpy())
        
        # Find examples closest to centroids
        selected_indices = []
        centroids = torch.tensor(kmeans.cluster_centers_)
        
        for centroid in centroids:
            # Compute distances to centroid
            distances = torch.norm(embeddings - centroid, dim=1)
            # Get index of closest example
            closest_idx = torch.argmin(distances).item()
            selected_indices.append(closest_idx)
        
        return [examples[idx] for idx in selected_indices]
    
    def _initialize_strategic_components(self):
        """Initialize strategic classification components."""
        try:
            # Create cost function from config
            if self.config.cost_coefficients:
                self.strategic_cost_function = CostFunctionFactory.create_cost_function(
                    cost_type=self.config.cost_function_type,
                    cost_coefficients=self.config.cost_coefficients
                )
                
                # Initialize strategic optimizer and evaluator
                self.strategic_optimizer = StrategicOptimizer(self.strategic_cost_function)
                self.strategic_evaluator = StrategicEvaluator(self.strategic_cost_function)
                
                logger.info(f"Initialized strategic mode with {self.config.cost_function_type} cost function")
            else:
                logger.warning("Strategic mode enabled but no cost coefficients provided")
        except Exception as e:
            logger.error(f"Failed to initialize strategic components: {e}")
            self.config.enable_strategic_mode = False
    
    @property
    def strategic_mode(self) -> bool:
        """Check if strategic mode is enabled and properly initialized."""
        return (
            self.config.enable_strategic_mode and 
            self.strategic_cost_function is not None
        )
    
    def _strategic_training_step(self, all_embeddings: torch.Tensor, all_labels: torch.Tensor):
        """Perform strategic training step."""
        if not self.strategic_mode or self.adaptive_head is None:
            return
        
        # Use strategic optimizer for training
        self.adaptive_head.train()
        optimizer = torch.optim.AdamW(
            self.adaptive_head.parameters(),
            lr=self.config.learning_rate * 0.5,  # Lower learning rate for strategic training
            weight_decay=0.01
        )
        
        # Create data loader for strategic training
        dataset = torch.utils.data.TensorDataset(all_embeddings, all_labels)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(16, len(all_embeddings)),  # Smaller batch size for strategic training
            shuffle=True,
            generator=torch.Generator().manual_seed(42)
        )
        
        # Strategic training loop
        for epoch in range(5):  # Fewer epochs for strategic training
            for batch_embeddings, batch_labels in loader:
                batch_embeddings = batch_embeddings.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Compute strategic loss
                strategic_loss = self.strategic_optimizer.strategic_loss(
                    self.adaptive_head,
                    batch_embeddings,
                    batch_labels,
                    self.config.strategic_lambda
                )
                
                strategic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.adaptive_head.parameters(),
                    max_norm=1.0
                )
                optimizer.step()
        
        logger.debug("Completed strategic training step")
    
    def predict_strategic(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Predict assuming the input might be strategically modified.
        
        This method simulates how a strategic agent might modify the input
        to get a better classification outcome, then predicts on that modified input.
        
        Args:
            text: Input text to classify
            k: Number of top predictions to return
            
        Returns:
            List of (label, confidence) tuples for strategic predictions
        """
        if not self.strategic_mode:
            return self._predict_regular(text, k)
        
        try:
            # Get embedding
            embedding = self._get_embeddings([text])[0]
            
            # Create classifier function for strategic optimization
            def classifier_func(x):
                with torch.no_grad():
                    if self.adaptive_head is not None:
                        self.adaptive_head.eval()
                        # Ensure proper input shape
                        if x.dim() == 1:
                            x = x.unsqueeze(0)
                        logits = self.adaptive_head(x.to(self.device))
                        return F.softmax(logits, dim=-1)
                    else:
                        # Fallback to uniform distribution if no neural head
                        num_classes = len(self.label_to_id) if self.label_to_id else 1
                        return torch.ones(1, num_classes) / num_classes
            
            # Compute what the strategic response would be
            strategic_embedding = self.strategic_cost_function.compute_best_response(
                embedding, classifier_func
            )
            
            # Predict on the strategic embedding
            return self._predict_from_embedding(strategic_embedding, k, strategic=True)
            
        except Exception as e:
            logger.warning(f"Strategic prediction failed: {e}. Falling back to regular prediction.")
            return self._predict_regular(text, k)
    
    def predict_robust(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Predict assuming input has already been strategically modified.
        
        This method assumes the input text has already been strategically manipulated
        and applies robust prediction techniques that are less susceptible to such manipulation.
        
        Args:
            text: Input text (potentially strategically modified)
            k: Number of top predictions to return
            
        Returns:
            List of (label, confidence) tuples for robust predictions
        """
        if not self.strategic_mode:
            return self._predict_regular(text, k)
        
        try:
            # Get embedding
            embedding = self._get_embeddings([text])[0]
            
            # Use strategic-aware prediction that considers manipulation
            return self._predict_from_embedding(embedding, k, robust=True)
            
        except Exception as e:
            logger.warning(f"Robust prediction failed: {e}. Falling back to regular prediction.")
            return self._predict_regular(text, k)
    
    def _predict_from_embedding(
        self, 
        embedding: torch.Tensor, 
        k: int = 5, 
        robust: bool = False,
        strategic: bool = False
    ) -> List[Tuple[str, float]]:
        """Helper method to predict from embedding with strategic considerations.
        
        Args:
            embedding: Input embedding tensor
            k: Number of top predictions to return
            robust: If True, applies robust prediction weights
            strategic: If True, indicates this is for strategic prediction
            
        Returns:
            List of (label, confidence) tuples
        """
        with torch.no_grad():
            # Get prototype predictions
            proto_preds = self.memory.get_nearest_prototypes(embedding, k=k)
            
            # Get neural predictions if available
            if self.adaptive_head is not None:
                self.adaptive_head.eval()
                input_embedding = embedding.unsqueeze(0).to(self.device)
                logits = self.adaptive_head(input_embedding)
                logits = logits.squeeze(0)
                probs = F.softmax(logits, dim=0)
                
                values, indices = torch.topk(probs, min(k, len(self.id_to_label)))
                head_preds = [
                    (self.id_to_label[idx.item()], val.item())
                    for val, idx in zip(values, indices)
                ]
            else:
                head_preds = []
        
        # Combine predictions with strategic adjustments
        combined_scores = {}
        
        # Determine weights based on prediction mode
        if self.strategic_mode and robust:
            # In robust mode, weight prototypes more heavily
            # as they're less susceptible to strategic manipulation
            proto_weight = self.config.strategic_robust_proto_weight
            head_weight = self.config.strategic_robust_head_weight
        elif self.strategic_mode and strategic:
            # In strategic mode, balance weights differently
            # Neural networks may be more responsive to strategic changes
            proto_weight = self.config.strategic_prediction_proto_weight
            head_weight = self.config.strategic_prediction_head_weight
        else:
            # Regular weighting
            proto_weight = self.config.prototype_weight
            head_weight = self.config.neural_weight
        
        for label, score in proto_preds:
            combined_scores[label] = score * proto_weight
        
        for label, score in head_preds:
            combined_scores[label] = combined_scores.get(label, 0) + score * head_weight
        
        # Normalize scores
        predictions = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        total = sum(score for _, score in predictions)
        if total > 0:
            predictions = [(label, score/total) for label, score in predictions]
        
        return predictions[:k]
    
    def evaluate_strategic_robustness(
        self,
        test_texts: List[str],
        test_labels: List[str],
        gaming_levels: List[float] = [0.0, 0.5, 1.0]
    ) -> Dict[str, float]:
        """Evaluate strategic robustness of the classifier."""
        if not self.strategic_mode:
            raise ValueError("Strategic mode not enabled")
        
        # Get test embeddings
        test_embeddings = torch.stack(self._get_embeddings(test_texts))
        
        # Convert labels to indices
        test_label_indices = torch.tensor([
            self.label_to_id[label] for label in test_labels
        ])
        
        # Evaluate robustness
        return self.strategic_evaluator.evaluate_robustness(
            self.adaptive_head,
            test_embeddings,
            test_label_indices,
            gaming_levels
        )
