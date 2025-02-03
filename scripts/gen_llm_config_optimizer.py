import argparse
import json
import os
import logging
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from adaptive_classifier import AdaptiveClassifier
import textdistance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1", 
    api_key=os.environ.get("OPENAI_API_KEY")
)

@dataclass
class TemperatureConfig:
    """Configuration for temperature-based classification."""
    class_ranges = {
        "DETERMINISTIC": (0.0, 0.1),
        "FOCUSED": (0.2, 0.5),
        "BALANCED": (0.6, 1.0),
        "CREATIVE": (1.1, 1.5),
        "EXPERIMENTAL": (1.6, 2.0)
    }
    
    sample_temperatures = {
        "DETERMINISTIC": [0.0, 0.1],
        "FOCUSED": [0.3, 0.4],
        "BALANCED": [0.7, 0.8],
        "CREATIVE": [1.2, 1.3],
        "EXPERIMENTAL": [1.7, 1.8]
    }
    
    @classmethod
    def get_class_for_temperature(cls, temperature: float) -> str:
        """Get the class name for a given temperature."""
        for class_name, (min_temp, max_temp) in cls.class_ranges.items():
            if min_temp <= temperature <= max_temp:
                return class_name
        return "BALANCED"  # Default fallback

    @classmethod
    def get_temperatures_for_class(cls, class_name: str) -> List[float]:
        """Get sample temperatures for a class."""
        return cls.sample_temperatures.get(class_name, [0.7])

@dataclass
class TrainingConfig:
    """Training configuration."""
    model: str
    max_retries: int = 3
    retry_delay: int = 1
    similarity_threshold: float = 0.5
    max_examples: int = 512
    batch_size: int = 4

class ResponseEvaluator:
    """Evaluates response quality using multiple metrics."""
    
    def __init__(self):
        """Initialize evaluator with models and vectorizers."""
        self.tfidf = TfidfVectorizer(stop_words='english')
        try:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load semantic model: {e}")
            self.semantic_model = None
    
    def evaluate_responses(self, response_1: str, response_2: str) -> Tuple[float, Dict[str, float]]:
        """Evaluate responses using multiple metrics."""
        scores = {}
        
        # Length normalization factor
        len_ratio = min(len(response_1), len(response_2)) / max(len(response_1), len(response_2))
        scores['length_ratio'] = len_ratio
        
        # Lexical similarity (TF-IDF)
        try:
            tfidf_matrix = self.tfidf.fit_transform([response_1, response_2])
            scores['lexical_similarity'] = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except Exception:
            scores['lexical_similarity'] = 0.0
        
        # Semantic similarity
        if self.semantic_model:
            try:
                embeddings = self.semantic_model.encode([response_1, response_2])
                scores['semantic_similarity'] = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
            except Exception:
                scores['semantic_similarity'] = 0.0
        
        # Levenshtein ratio for character-level similarity
        scores['edit_similarity'] = textdistance.levenshtein.normalized_similarity(response_1, response_2)
        
        # Compute weighted average score
        weights = {
            'semantic_similarity': 0.4,
            'lexical_similarity': 0.3,
            'edit_similarity': 0.2,
            'length_ratio': 0.1
        }
        
        total_score = sum(scores.get(k, 0) * v for k, v in weights.items())
        return total_score, scores

class ConfigOptimizer:
    """Optimizer class to find best temperature configurations for queries."""
    
    def __init__(
        self,
        training_config: TrainingConfig
    ):
        """Initialize the optimizer."""
        self.training_config = training_config
        self.classifier = AdaptiveClassifier("distilbert-base-uncased")
        self.evaluator = ResponseEvaluator()
        self.stats = {
            "total_queries": 0,
            "successful_optimizations": 0,
            "failed_optimizations": 0,
            "avg_similarity_score": 0.0,
            "class_distribution": {
                class_name: 0 for class_name in TemperatureConfig.class_ranges.keys()
            },
            "detailed_scores": []
        }
    
    def find_best_temperature_class(self, query: str) -> Tuple[Optional[str], float, Dict[str, float]]:
        """Find best temperature class for a query."""
        best_score = -1
        best_class = None
        best_metrics = {}
        configs_tested = 0
        
        # Try each temperature class
        for class_name in TemperatureConfig.class_ranges.keys():
            # Test sample temperatures for this class
            class_temps = TemperatureConfig.get_temperatures_for_class(class_name)
            for temp in class_temps:
                configs_tested += 1
                score, metrics = self._evaluate_temperature(query, temp)
                logger.debug(f"Temperature {temp:.1f} achieved score {score:.3f}")
                if score > best_score:
                    best_score = score
                    best_class = class_name
                    best_metrics = metrics
                    
                if best_score >= 0.8:  # Early stopping if we find a very good match
                    break
            
            if best_score >= 0.8:
                break
        
        logger.info(f"Tested {configs_tested} configurations for query")
        return best_class, best_score, best_metrics
    
    def _evaluate_temperature(self, query: str, temperature: float) -> Tuple[float, Dict[str, float]]:
        """Evaluate a temperature setting using RTC."""
        config = {"temperature": temperature, "top_p": 1.0}
        
        # Get initial response
        response_1 = self._get_llm_response(query, config)
        if not response_1:
            return 0.0, {}
            
        # Generate alternate query
        inverse_prompt = f"""Given this query and response pair, generate a new query that would lead to a similar response:

Original Query: {query}
Response: {response_1}

Generate a new query that would elicit a similar response:"""

        alternate_query = self._get_llm_response(inverse_prompt, config)
        if not alternate_query:
            return 0.0, {}
            
        # Get response for alternate query
        response_2 = self._get_llm_response(alternate_query, config)
        if not response_2:
            return 0.0, {}
            
        # Evaluate similarity
        similarity_score, metrics = self.evaluator.evaluate_responses(response_1, response_2)
        metrics['temperature'] = temperature
        return similarity_score, metrics
    
    def _get_llm_response(self, prompt: str, config: Dict[str, float]) -> Optional[str]:
        """Get response from the LLM with improved error handling."""
        messages = [{"role": "user", "content": prompt}]
        
        for attempt in range(self.training_config.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.training_config.model,
                    messages=messages,
                    max_tokens=4096,
                    **config
                )
                
                # Validate response
                if not response or not hasattr(response, 'choices') or not response.choices:
                    logger.warning(f"Invalid response structure (attempt {attempt + 1})")
                    continue
                
                content = response.choices[0].message.content
                if not content or not isinstance(content, str):
                    logger.warning(f"Invalid content (attempt {attempt + 1})")
                    continue
                
                return content.strip()
                
            except Exception as e:
                logger.error(f"Error getting LLM response (attempt {attempt + 1}): {e}")
            
            # Wait before retry with exponential backoff
            if attempt < self.training_config.max_retries - 1:
                sleep_time = self.training_config.retry_delay * (2 ** attempt)
                time.sleep(sleep_time)
        
        return None
    
    def optimize_and_train(self, save_path: str, push_to_hub: str):
        """Run optimization and training process."""
        try:
            dataset = load_dataset("lmarena-ai/arena-hard-auto-v0.1")
            logger.info("Successfully loaded dataset")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return

        logger.info(f"Starting optimization for model: {self.training_config.model}")
        
        successful_examples = []
        
        # Process examples in batches
        for i in tqdm(range(0, min(len(dataset["train"]), self.training_config.max_examples), 
                           self.training_config.batch_size)):
            batch = dataset["train"][i:i + self.training_config.batch_size]
            
            for item in batch:
                query = item['text'] if isinstance(item, dict) else str(item)
                
                self.stats["total_queries"] += 1
                
                # Find best temperature class
                best_class, score, metrics = self.find_best_temperature_class(query)
                
                if best_class and score >= self.training_config.similarity_threshold:
                    successful_examples.append((query, best_class))
                    
                    self.stats["successful_optimizations"] += 1
                    self.stats["avg_similarity_score"] = (
                        (self.stats["avg_similarity_score"] * (len(successful_examples) - 1) + score) /
                        len(successful_examples)
                    )
                    
                    self.stats["class_distribution"][best_class] += 1
                    self.stats["detailed_scores"].append({
                        "query": query,
                        "class": best_class,
                        "score": score,
                        "metrics": metrics
                    })
                else:
                    self.stats["failed_optimizations"] += 1
                
                # Print intermediate stats
                if self.stats["total_queries"] % 50 == 0:
                    self._print_intermediate_stats()
            
            # Train classifier on accumulated examples
            if successful_examples:
                queries, labels = zip(*successful_examples)
                self.classifier.add_examples(list(queries), list(labels))
                successful_examples = []
        
        # Save results
        self._save_results(save_path)
        
        # Push to HuggingFace if requested
        if push_to_hub:
            repo_id = f"adaptive-classifier/{push_to_hub}"
            logger.info(f"\nPushing to HuggingFace Hub: {repo_id}")
            try:
                self.classifier.push_to_hub(repo_id)
                logger.info("Successfully pushed to HuggingFace Hub")
            except Exception as e:
                logger.error(f"Error pushing to HuggingFace Hub: {e}")
        
        # Print final stats
        self._print_final_stats()
    
    def _print_intermediate_stats(self):
        """Print intermediate statistics."""
        logger.info("\nIntermediate Statistics:")
        logger.info(f"Processed queries: {self.stats['total_queries']}")
        logger.info(f"Successful optimizations: {self.stats['successful_optimizations']}")
        success_rate = (self.stats['successful_optimizations'] / self.stats['total_queries']) * 100
        logger.info(f"Current success rate: {success_rate:.2f}%")
        logger.info(f"Average similarity score: {self.stats['avg_similarity_score']:.3f}")
        logger.info("\nClass distribution:")
        for class_name, count in self.stats['class_distribution'].items():
            if count > 0:
                percentage = (count / self.stats['successful_optimizations']) * 100
                logger.info(f"{class_name}: {count} ({percentage:.1f}%)")
    
    def _print_final_stats(self):
        """Print final detailed statistics."""
        logger.info("\nFinal Statistics:")
        logger.info(f"Total queries processed: {self.stats['total_queries']}")
        logger.info(f"Successful optimizations: {self.stats['successful_optimizations']}")
        logger.info(f"Failed optimizations: {self.stats['failed_optimizations']}")
        
        if self.stats['successful_optimizations'] > 0:
            success_rate = (self.stats['successful_optimizations'] / self.stats['total_queries']) * 100
            logger.info(f"Success rate: {success_rate:.2f}%")
            logger.info(f"Average similarity score: {self.stats['avg_similarity_score']:.3f}")
            
            logger.info("\nTemperature Class Distribution:")
            for class_name, count in sorted(
                self.stats['class_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                if count > 0:
                    percentage = (count / self.stats['successful_optimizations']) * 100
                    logger.info(f"{class_name}: {count} ({percentage:.1f}%)")
            
            # Print average scores by class
            logger.info("\nAverage Scores by Class:")
            class_scores = {}
            for result in self.stats['detailed_scores']:
                class_name = result['class']
                if class_name not in class_scores:
                    class_scores[class_name] = []
                class_scores[class_name].append(result['score'])
            
            for class_name, scores in class_scores.items():
                avg_score = sum(scores) / len(scores)
                logger.info(f"{class_name}: {avg_score:.3f}")
    
    def _save_results(self, save_path: str):
        """Save classifier and statistics."""
        # Save classifier
        self.classifier.save(save_path)
        
        # Save stats
        stats_path = Path(save_path) / "optimization_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"\nResults saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Train an adaptive classifier for LLM temperature optimization"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model to optimize"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=500,
        help="Maximum number of examples to process"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./config_optimizer",
        help="Path to save the classifier"
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        help="Name to use when pushing to HuggingFace"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for processing examples"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Minimum similarity score threshold"
    )
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Print configuration summary
    logger.info("Starting optimization with configuration:")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max examples: {args.max_examples}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Similarity threshold: {args.similarity_threshold}")
    logger.info(f"Save path: {args.save_path}")
    logger.info(f"Push to hub: {args.push_to_hub}")
    
    # Initialize configs
    training_config = TrainingConfig(
        model=args.model,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        similarity_threshold=args.similarity_threshold
    )
    
    # Initialize optimizer
    optimizer = ConfigOptimizer(training_config)
    
    # Run optimization and training
    optimizer.optimize_and_train(args.save_path, args.push_to_hub)

if __name__ == "__main__":
    main()