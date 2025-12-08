import json
import time
import requests
from typing import Dict, List, Any, Literal

# ----------------------------
# 1. Result Merger
# ----------------------------
class ResultMerger:
    """Combines baseline and embedding retrieval results"""
    
    def merge_results(self, baseline_results: Dict, embedding_results: Dict) -> Dict:
        """
        Merge results from both retrieval methods
        
        Args:
            baseline_results: Output from FPLBaselineRetriever
            embedding_results: Output from FPLEmbeddingRetriever
            
        Returns:
            Unified context dictionary
        """
        merged = {
            "structured_data": [],  # From Cypher queries
            "semantic_matches": [], # From embeddings
            "entities": baseline_results.get("entities", {}),
            "intent": baseline_results.get("intent", "unknown")
        }
        
        # Add baseline results
        for result in baseline_results.get("results", []):
            merged["structured_data"].extend(result.get("data", []))
        
        # Add embedding results (both models)
        for model_name in ["model1", "model2"]:
            if model_name in embedding_results:
                merged["semantic_matches"].extend(embedding_results[model_name])
        
        # Remove duplicates based on player name
        merged["semantic_matches"] = self._deduplicate(
            merged["semantic_matches"], 
            key="name"
        )
        
        # Rank by similarity score
        merged["semantic_matches"].sort(
            key=lambda x: x.get("similarity_score", 0), 
            reverse=True
        )
        
        # Keep top 10 semantic matches
        merged["semantic_matches"] = merged["semantic_matches"][:10]
        
        return merged
    
    def _deduplicate(self, items: List[Dict], key: str) -> List[Dict]:
        """Remove duplicates based on a key"""
        seen = set()
        unique = []
        for item in items:
            if item.get(key) not in seen:
                seen.add(item.get(key))
                unique.append(item)
        return unique


# ----------------------------
# 2. Prompt Builder
# ----------------------------
class PromptBuilder:
    """Builds structured prompts with Context, Persona, Task"""
    
    def build_prompt(self, user_query: str, merged_context: Dict) -> str:
        """
        Create structured prompt following Context-Persona-Task format
        
        Args:
            user_query: Original user question
            merged_context: Combined KG results
            
        Returns:
            Formatted prompt string
        """
        # CONTEXT: KG information
        context = self._format_context(merged_context)
        
        # PERSONA: Define assistant role
        persona = """You are an expert Fantasy Premier League (FPL) assistant with deep knowledge of player statistics, 
team performance, and strategic recommendations. You provide accurate, data-driven advice based on the provided information."""
        
        # TASK: Clear instructions
        task = f"""Using ONLY the information provided in the context above, answer the following question:

Question: {user_query}

Requirements:
1. Answer based solely on the provided data
2. If information is insufficient, clearly state what's missing
3. Cite specific statistics when relevant (e.g., "Salah scored 20 goals with 250 total points")
4. Be concise and actionable
5. Do not make up or assume information not in the context
6. Format numbers clearly (e.g., "£8.5m" for value)

Answer:"""
        
        # Combine all parts
        full_prompt = f"""### CONTEXT ###
{context}

### PERSONA ###
{persona}

### TASK ###
{task}
"""
        return full_prompt
    
    def _format_context(self, merged_context: Dict) -> str:
        """Format merged context into readable text"""
        context_parts = []
        
        # Add intent and entities first
        context_parts.append(f"User Intent: {merged_context.get('intent', 'unknown')}")
        if merged_context.get("entities"):
            context_parts.append(f"Extracted Entities: {json.dumps(merged_context['entities'])}")
        context_parts.append("")
        
        # Format structured data (Cypher results)
        if merged_context.get("structured_data"):
            context_parts.append("=== Structured Query Results (from Knowledge Graph) ===")
            for i, item in enumerate(merged_context["structured_data"][:15], 1):
                # Format each item nicely
                formatted_item = ", ".join([f"{k}: {v}" for k, v in item.items()])
                context_parts.append(f"{i}. {formatted_item}")
            context_parts.append("")
        
        # Format semantic matches (embedding results)
        if merged_context.get("semantic_matches"):
            context_parts.append("=== Semantic Search Results (similar players) ===")
            for i, item in enumerate(merged_context["semantic_matches"][:8], 1):
                context_parts.append(
                    f"{i}. {item['name']} ({item['position']}) - "
                    f"Total Points: {item['total_points']}, "
                    f"Goals: {item.get('goals', 0)}, "
                    f"Assists: {item.get('assists', 0)}, "
                    f"Value: £{item['avg_value']}m, "
                    f"Form: {item.get('avg_form', 0)}"
                )
            context_parts.append("")
        
        return "\n".join(context_parts)


# ----------------------------
# 3. LLM Client (Ollama with Llama, Mistral, Phi-3)
# ----------------------------
class LLMClient:
    """Client for Ollama LLM models"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.merger = ResultMerger()
        self.prompt_builder = PromptBuilder()
        
        # Define the three models
        self.models = {
            "llama": "llama3.2",     # Llama 3.2
            "mistral": "mistral",     # Mistral 7B
            "phi3": "phi3"            # Phi-3
        }
    
    def generate_response(
        self, 
        user_query: str, 
        baseline_results: Dict, 
        embedding_results: Dict,
        model_name: Literal["llama", "mistral", "phi3"] = "llama"
    ) -> Dict:
        """
        Generate LLM response using merged KG results
        
        Args:
            user_query: Original user question
            baseline_results: Cypher query results from retriever.py
            embedding_results: Semantic search results from retriever.py
            model_name: Which model to use ("llama", "mistral", or "phi3")
            
        Returns:
            Dict with response, metadata, and metrics
        """
        print(f"\n{'='*60}")
        print(f"Generating response with {self.models[model_name]}...")
        print(f"{'='*60}")
        
        # Step 1: Merge results
        merged_context = self.merger.merge_results(baseline_results, embedding_results)
        print(f"✓ Merged {len(merged_context['structured_data'])} structured results")
        print(f"✓ Merged {len(merged_context['semantic_matches'])} semantic matches")
        
        # Step 2: Build structured prompt
        prompt = self.prompt_builder.build_prompt(user_query, merged_context)
        print(f"✓ Built prompt ({len(prompt)} characters)")
        
        # Step 3: Call Ollama API
        start_time = time.time()
        
        try:
            response = self._call_ollama(prompt, model_name)
            end_time = time.time()
            
            return {
                "answer": response["text"],
                "model": self.models[model_name],
                "model_name": model_name,
                "response_time": end_time - start_time,
                "token_count": response.get("eval_count", 0),
                "prompt_token_count": response.get("prompt_eval_count", 0),
                "merged_context": merged_context,
                "prompt": prompt,
                "success": True
            }
        
        except Exception as e:
            end_time = time.time()
            return {
                "error": str(e),
                "model": self.models[model_name],
                "model_name": model_name,
                "response_time": end_time - start_time,
                "success": False
            }
    
    def _call_ollama(self, prompt: str, model_name: str) -> Dict:
        """Call local Ollama API"""
        url = f"{self.ollama_url}/api/generate"
        
        payload = {
            "model": self.models[model_name],
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        return {
            "text": result.get("response", ""),
            "eval_count": result.get("eval_count", 0),  # Output tokens
            "prompt_eval_count": result.get("prompt_eval_count", 0)  # Input tokens
        }
    
    def compare_models(
        self, 
        user_query: str, 
        baseline_results: Dict, 
        embedding_results: Dict
    ) -> Dict:
        """
        Compare all three models (Llama, Mistral, Phi-3) on the same query
        
        Returns:
            Dict with results from all models for comparison
        """
        print(f"\n{'='*60}")
        print(f"COMPARING 3 MODELS ON QUERY: {user_query}")
        print(f"{'='*60}")
        
        results = {}
        
        for model_name in ["llama", "mistral", "phi3"]:
            try:
                results[model_name] = self.generate_response(
                    user_query,
                    baseline_results,
                    embedding_results,
                    model_name
                )
                
                if results[model_name]["success"]:
                    print(f"\n✓ {model_name.upper()}: Generated response in {results[model_name]['response_time']:.2f}s")
                else:
                    print(f"\n✗ {model_name.upper()}: Failed - {results[model_name].get('error')}")
            
            except Exception as e:
                print(f"\n✗ {model_name.upper()}: Exception - {str(e)}")
                results[model_name] = {
                    "error": str(e),
                    "model": self.models[model_name],
                    "success": False
                }
        
        return results


# ----------------------------
# 4. Model Evaluator
# ----------------------------
class ModelEvaluator:
    """Evaluate and compare LLM performance"""
    
    def evaluate_response(
        self, 
        response: Dict, 
        ground_truth: str = None,
        user_rating: int = None
    ) -> Dict:
        """
        Evaluate a single model response
        
        Args:
            response: Output from LLMClient.generate_response()
            ground_truth: Expected answer (optional)
            user_rating: Human rating 1-5 (optional)
            
        Returns:
            Evaluation metrics
        """
        if not response.get("success"):
            return {
                "error": response.get("error"),
                "success": False
            }
        
        answer = response.get("answer", "")
        
        metrics = {
            # Quantitative metrics
            "response_time_seconds": response["response_time"],
            "output_tokens": response["token_count"],
            "input_tokens": response.get("prompt_token_count", 0),
            "total_tokens": response["token_count"] + response.get("prompt_token_count", 0),
            "answer_length_chars": len(answer),
            "answer_length_words": len(answer.split()),
            
            # Qualitative metrics
            "user_rating": user_rating,  # 1-5 scale (manual)
            "contains_numbers": self._contains_statistics(answer),
            "cites_context": self._cites_context(answer, response["merged_context"]),
            "answer_completeness": self._check_completeness(answer),
            "success": True
        }
        
        if ground_truth:
            metrics["keyword_overlap"] = self._calculate_keyword_overlap(answer, ground_truth)
        
        return metrics
    
    def _contains_statistics(self, answer: str) -> bool:
        """Check if answer contains numerical data"""
        import re
        return bool(re.search(r'\d+', answer))
    
    def _cites_context(self, answer: str, context: Dict) -> bool:
        """Check if answer references provided data"""
        # Check if answer mentions any player names from context
        all_names = []
        
        # From structured data
        for item in context.get("structured_data", []):
            if "p.player_name" in item:
                all_names.append(item["p.player_name"])
            elif "name" in item:
                all_names.append(item["name"])
        
        # From semantic matches
        for item in context.get("semantic_matches", []):
            all_names.append(item["name"])
        
        # Check if any name appears in answer
        return any(name.lower() in answer.lower() for name in all_names if name)
    
    def _check_completeness(self, answer: str) -> str:
        """Check if answer seems complete"""
        if len(answer.strip()) < 20:
            return "too_short"
        elif "I don't" in answer or "no information" in answer.lower():
            return "insufficient_data"
        elif answer.strip().endswith(('.', '!', '?')):
            return "complete"
        else:
            return "incomplete"
    
    def _calculate_keyword_overlap(self, answer: str, ground_truth: str) -> float:
        """Simple keyword overlap score"""
        answer_words = set(answer.lower().split())
        truth_words = set(ground_truth.lower().split())
        
        if not truth_words:
            return 0.0
        
        overlap = len(answer_words.intersection(truth_words))
        return overlap / len(truth_words)
    
    def compare_models_report(self, comparison_results: Dict, evaluations: Dict = None) -> str:
        """Generate detailed comparison report"""
        report = [
            "\n" + "="*80,
            "MODEL COMPARISON REPORT",
            "="*80
        ]
        
        for model_name in ["llama", "mistral", "phi3"]:
            result = comparison_results.get(model_name, {})
            
            report.append(f"\n{'─'*80}")
            report.append(f"MODEL: {model_name.upper()} ({result.get('model', 'N/A')})")
            report.append(f"{'─'*80}")
            
            if not result.get("success"):
                report.append(f"❌ ERROR: {result.get('error', 'Unknown error')}")
                continue
            
            # Basic info
            report.append(f"\n⏱️  Response Time: {result['response_time']:.2f} seconds")
            report.append(f"🔢 Tokens: {result.get('prompt_token_count', 0)} input + {result['token_count']} output = {result.get('prompt_token_count', 0) + result['token_count']} total")
            
            # Show answer
            report.append(f"\n📝 ANSWER:")
            report.append("-" * 80)
            report.append(result['answer'][:500] + ("..." if len(result['answer']) > 500 else ""))
            report.append("-" * 80)
            
            # Show evaluation metrics if available
            if evaluations and model_name in evaluations:
                eval_metrics = evaluations[model_name]
                if eval_metrics.get("success"):
                    report.append(f"\n📊 EVALUATION METRICS:")
                    report.append(f"  • Contains Statistics: {'✓' if eval_metrics['contains_numbers'] else '✗'}")
                    report.append(f"  • Cites Context: {'✓' if eval_metrics['cites_context'] else '✗'}")
                    report.append(f"  • Completeness: {eval_metrics['answer_completeness']}")
                    report.append(f"  • Answer Length: {eval_metrics['answer_length_words']} words")
                    if eval_metrics.get('user_rating'):
                        report.append(f"  • User Rating: {eval_metrics['user_rating']}/5 ⭐")
        
        report.append("\n" + "="*80)
        return "\n".join(report)
    
    def generate_summary_table(self, comparison_results: Dict, evaluations: Dict = None) -> str:
        """Generate a summary comparison table"""
        from io import StringIO
        
        output = StringIO()
        output.write("\n" + "="*100 + "\n")
        output.write("SUMMARY COMPARISON TABLE\n")
        output.write("="*100 + "\n")
        output.write(f"{'Model':<15} {'Time (s)':<12} {'Tokens':<12} {'Has Stats':<12} {'Cites Data':<12} {'Complete':<12}\n")
        output.write("-"*100 + "\n")
        
        for model_name in ["llama", "mistral", "phi3"]:
            result = comparison_results.get(model_name, {})
            
            if not result.get("success"):
                output.write(f"{model_name.upper():<15} {'ERROR':<12} {'-':<12} {'-':<12} {'-':<12} {'-':<12}\n")
                continue
            
            eval_data = evaluations.get(model_name, {}) if evaluations else {}
            
            time_str = f"{result['response_time']:.2f}"
            tokens_str = f"{result['token_count']}"
            has_stats = "✓" if eval_data.get("contains_numbers") else "✗"
            cites = "✓" if eval_data.get("cites_context") else "✗"
            complete = eval_data.get("answer_completeness", "N/A")
            
            output.write(f"{model_name.upper():<15} {time_str:<12} {tokens_str:<12} {has_stats:<12} {cites:<12} {complete:<12}\n")
        
        output.write("="*100 + "\n")
        return output.getvalue()


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    print("LLM Client initialized with Llama, Mistral, and Phi-3")
    print("Use this module with retriever.py to generate responses")
    print("\nExample:")
    print("  from retriever import FPLHybridRetriever")
    print("  from llm_client import LLMClient, ModelEvaluator")
    print("  ")
    print("  retriever = FPLHybridRetriever()")
    print("  client = LLMClient()")
    print("  ")
    print("  results = retriever.retrieve('Who scored the most goals?', use_embeddings=True)")
    print("  response = client.generate_response('Who scored the most goals?', ")
    print("                                       results['baseline'], ")
    print("                                       results['semantic_search'],")
    print("                                       model_name='llama')")
    print("  print(response['answer'])")