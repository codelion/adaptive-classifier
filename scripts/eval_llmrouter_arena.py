import argparse
import json
import os
import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
from adaptive_classifier import AdaptiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(base_url="http://localhost:8000/v1", api_key=os.environ.get("OPENAI_API_KEY"))
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@dataclass
class RouterConfig:
    """Configuration for the LLM Router evaluation."""
    high_model: str = "gpt-4o"
    low_model: str = "gpt-4o-mini"
    similarity_threshold: float = 0.6
    max_retries: int = 3
    retry_delay: int = 1
    adaptive_router_path: str = "./adaptive_router"

class LLMRouter:
    """Router class to direct queries to appropriate models."""
    
    def __init__(self, config: RouterConfig, enable_adaptation: bool = True):
        """Initialize the router with classifier and configuration."""
        self.config = config
        self.enable_adaptation = enable_adaptation
        self.classifier = AdaptiveClassifier.load(config.adaptive_router_path)
        self.stats = {
            "total_queries": 0,
            "high_routes": 0,
            "low_routes": 0,
            "high_success": 0,
            "low_success": 0,
            "adapted_examples": 0
        }

    def route_and_evaluate(self, query: str) -> Tuple[bool, Dict]:
        """Route query to appropriate model and evaluate results."""
        # Get routing decision
        predictions = self.classifier.predict(query)
        route = predictions[0][0]  # Get top prediction
        
        # Select model based on route
        model = self.config.high_model if route == "HIGH" else self.config.low_model
        
        # Update stats
        self.stats["total_queries"] += 1
        if route == "HIGH":
            self.stats["high_routes"] += 1
        else:
            self.stats["low_routes"] += 1
            
        # Perform RTC evaluation
        passed_rtc, similarity_score, details = perform_rtc_evaluation(
            query, model, self.config
        )
        
        # Update success stats
        if passed_rtc:
            if route == "HIGH":
                self.stats["high_success"] += 1
            else:
                self.stats["low_success"] += 1
                
        # Adapt if enabled and RTC passed
        if self.enable_adaptation and passed_rtc:
            self.adapt_to_example(query, route)
            self.stats["adapted_examples"] += 1
            
        evaluation_result = {
            "query": query,
            "route": route,
            "model": model,
            "passed_rtc": passed_rtc,
            "similarity_score": similarity_score,
            "evaluation_details": details
        }
        
        return passed_rtc, evaluation_result

    def adapt_to_example(self, query: str, label: str):
        """Add successful example to classifier."""
        if self.enable_adaptation:
            self.classifier.add_examples([query], [label])
            
    def save_classifier(self):
        """Save the adapted classifier."""
        if self.enable_adaptation:
            self.classifier.save(self.config.adaptive_router_path)

    def get_stats(self) -> Dict:
        """Get routing statistics."""
        stats = self.stats.copy()
        stats["high_success_rate"] = (
            stats["high_success"] / stats["high_routes"] 
            if stats["high_routes"] > 0 else 0
        )
        stats["low_success_rate"] = (
            stats["low_success"] / stats["low_routes"]
            if stats["low_routes"] > 0 else 0
        )
        stats["overall_success_rate"] = (
            (stats["high_success"] + stats["low_success"]) / stats["total_queries"]
            if stats["total_queries"] > 0 else 0
        )
        stats["cost_saving_ratio"] = (
            stats["low_success"] / stats["total_queries"]
            if stats["total_queries"] > 0 else 0
        )
        return stats

def perform_rtc_evaluation(
    query: str,
    model: str,
    config: RouterConfig
) -> Tuple[bool, float, Dict]:
    """Perform Round-Trip Correctness evaluation."""
    # Get initial response
    response_1 = get_llm_response([
        {"role": "user", "content": query}
    ], model, config)
    
    if not response_1:
        return False, 0.0, {"error": "Failed to get initial response"}
        
    # Generate alternate query
    inverse_prompt = f"""Given this query and response pair, generate a new query that would lead to a similar response. Focus on the key aspects that would generate equivalent content:

Original Query: {query}
Response: {response_1}

Generate a new query that would elicit a similar response:"""

    alternate_query = get_llm_response([
        {"role": "user", "content": inverse_prompt}
    ], model, config)
    
    if not alternate_query:
        return False, 0.0, {"error": "Failed to generate alternate query"}
        
    # Get response for alternate query
    response_2 = get_llm_response([
        {"role": "user", "content": alternate_query}
    ], model, config)
    
    if not response_2:
        return False, 0.0, {"error": "Failed to get second response"}
        
    # Compute similarity
    similarity_score = compute_similarity(response_1, response_2)
    
    evaluation_details = {
        "original_query": query,
        "response_1": response_1,
        "alternate_query": alternate_query,
        "response_2": response_2,
        "similarity_score": similarity_score
    }
    
    return similarity_score >= config.similarity_threshold, similarity_score, evaluation_details

def get_llm_response(
    messages: List[Dict],
    model: str,
    config: RouterConfig
) -> Optional[str]:
    """Get response from the LLM with retry logic."""
    for attempt in range(config.max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=4096
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error getting LLM response (attempt {attempt + 1}): {e}")
            if attempt < config.max_retries - 1:
                time.sleep(config.retry_delay)
            continue
    return None

def compute_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts using TF-IDF."""
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        return 0.0

def extract_first_turn_content(turns: List[Dict]) -> str:
    """Extract content from first turn in conversation."""
    if not turns or not isinstance(turns, list):
        return ""
    return turns[0].get("content", "")

def evaluate_dataset(config: RouterConfig, enable_adaptation: bool, output_file: str):
    """Evaluate the dataset using the LLM router."""
    # Initialize router
    router = LLMRouter(config, enable_adaptation=enable_adaptation)
    
    # Load dataset
    dataset = load_dataset("lmarena-ai/arena-hard-auto-v0.1")
    
    results = []
    
    # Process each example
    for item in tqdm(dataset["train"], desc="Evaluating examples"):
        query = extract_first_turn_content(item["turns"])
        if not query:
            continue
            
        passed_rtc, evaluation_result = router.route_and_evaluate(query)
        results.append(evaluation_result)
        
        # Save intermediate results
        save_results(output_file, router, results)
    
    # Save final state if adaptation was enabled
    if enable_adaptation:
        router.save_classifier()
        
    # Print final summary
    print_summary(router)

def save_results(output_file: str, router: LLMRouter, results: List[Dict]):
    """Save evaluation results to file."""
    with open(output_file, 'w') as f:
        json.dump({
            "stats": router.get_stats(),
            "results": results
        }, f, indent=2)

def print_summary(router: LLMRouter):
    """Print evaluation summary."""
    stats = router.get_stats()
    
    logger.info("\nEvaluation Summary:")
    logger.info(f"Total queries processed: {stats['total_queries']}")
    logger.info(f"High-model routes: {stats['high_routes']}")
    logger.info(f"Low-model routes: {stats['low_routes']}")
    logger.info(f"High-model successes: {stats['high_success']}")
    logger.info(f"Low-model successes: {stats['low_success']}")
    logger.info(f"Adapted examples: {stats['adapted_examples']}")
    logger.info(f"High-model success rate: {stats['high_success_rate']*100:.2f}%")
    logger.info(f"Low-model success rate: {stats['low_success_rate']*100:.2f}%")
    logger.info(f"Overall success rate: {stats['overall_success_rate']*100:.2f}%")
    logger.info(f"Potential cost savings: {stats['cost_saving_ratio']*100:.2f}%")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM router on arena-hard-auto dataset"
    )
    parser.add_argument(
        "--high-model",
        type=str,
        default="gpt-4o",
        help="Model to use for high-complexity queries"
    )
    parser.add_argument(
        "--low-model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for low-complexity queries"
    )
    parser.add_argument(
        "--without-adaptation",
        action="store_true",
        help="Disable adaptive learning during evaluation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="router_eval_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--router-path",
        type=str,
        default="./adaptive_router",
        help="Path to load/save the adaptive router"
    )
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs("benchmark_results", exist_ok=True)
    output_file = os.path.join("benchmark_results", args.output)
    
    # Create configuration
    config = RouterConfig(
        high_model=args.high_model,
        low_model=args.low_model,
        adaptive_router_path=args.router_path
    )
    
    # Run evaluation
    evaluate_dataset(
        config,
        enable_adaptation=not args.without_adaptation,
        output_file=output_file
    )

if __name__ == "__main__":
    main()