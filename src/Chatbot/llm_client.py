import time
import torch
import json
import gc
from typing import Dict, List, Any, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# 1. Result Merger (Unchanged)
# ----------------------------
class ResultMerger:
    def merge_results(self, baseline_results: Dict, embedding_results: Dict) -> Dict:
        merged = {
            "structured_data": [], 
            "semantic_matches": [], 
            "entities": baseline_results.get("entities", {}),
            "intent": baseline_results.get("intent", "unknown")
        }
        
        for result in baseline_results.get("results", []):
            merged["structured_data"].extend(result.get("data", []))
        
        for model_name in ["model1", "model2"]:
            if model_name in embedding_results:
                merged["semantic_matches"].extend(embedding_results[model_name])
        
        merged["semantic_matches"] = self._deduplicate(merged["semantic_matches"], key="name")
        merged["semantic_matches"].sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        merged["semantic_matches"] = merged["semantic_matches"][:10]
        
        return merged
    
    def _deduplicate(self, items: List[Dict], key: str) -> List[Dict]:
        seen = set()
        unique = []
        for item in items:
            if item.get(key) not in seen:
                seen.add(item.get(key))
                unique.append(item)
        return unique

# ----------------------------
# 2. Prompt Builder (Unchanged)
# ----------------------------
class PromptBuilder:
    def build_context_string(self, merged_context: Dict) -> str:
        context_parts = []
        context_parts.append(f"User Intent: {merged_context.get('intent', 'unknown')}")
        if merged_context.get("entities"):
            context_parts.append(f"Extracted Entities: {json.dumps(merged_context['entities'])}")
        context_parts.append("")
        
        if merged_context.get("structured_data"):
            context_parts.append("=== Database Query Results ===")
            for i, item in enumerate(merged_context["structured_data"][:10], 1):
                formatted_item = ", ".join([f"{k}: {v}" for k, v in item.items()])
                context_parts.append(f"{i}. {formatted_item}")
            context_parts.append("")
        
        if merged_context.get("semantic_matches"):
            context_parts.append("=== Similar Players Found ===")
            for i, item in enumerate(merged_context["semantic_matches"][:8], 1):
                player_info = f"{item['name']} ({item['position']})"
                stats = f"Points: {item['total_points']}, Goals: {item.get('goals', 0)}"
                context_parts.append(f"{i}. {player_info} - {stats}")
            context_parts.append("")
        return "\n".join(context_parts)

# ----------------------------
# 3. LLM Client (Polished for GPU)
# ----------------------------
class LLMClient:
    def __init__(self, hf_api_token: str = None):
        self.hf_token = hf_api_token
        self.merger = ResultMerger()
        self.prompt_builder = PromptBuilder()
        
        # Robust Device Detection
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        print(f"⚙️  Hardware detected: {self.device.upper()}")
        
        self.current_model_name = None
        self.model = None
        self.tokenizer = None
        
        # --- UPDATED MODEL LIST ---
        self.models = {
            "llama": "meta-llama/Llama-3.2-1B-Instruct",
            "qwen": "Qwen/Qwen2.5-1.5B-Instruct", 
            "smollm": "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        }

    def _load_model(self, model_key: str):
        if self.current_model_name == model_key:
            return 

        print(f"🔄 Switching model to {model_key.upper()}...")
        
        # Clear VRAM
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        model_id = self.models[model_key]
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                token=self.hf_token,
                trust_remote_code=True
            )
            
            # --- FIX: Ensure pad_token is set (Fixes attention mask warning) ---
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=self.hf_token,
                torch_dtype="auto",  # Changed from torch_dtype to avoid warning
                device_map=self.device,
                trust_remote_code=True
            )
            self.current_model_name = model_key
            print(f"✅ Loaded {model_id}")
            
        except Exception as e:
            raise Exception(f"Failed to load {model_id}: {str(e)}")

    def generate_response(
        self, 
        user_query: str, 
        baseline_results: Dict, 
        embedding_results: Dict,
        model_name: Literal["llama", "qwen", "smollm"] = "llama"
    ) -> Dict:
        # Step 1: Merge Context
        merged_context = self.merger.merge_results(baseline_results, embedding_results)
        context_str = self.prompt_builder.build_context_string(merged_context)
        
        # Step 2: Build Messages
        system_prompt = """You are an expert FPL assistant. 
Using ONLY the provided context, answer the user's question.
If the context doesn't have the answer, say "I don't have that information".
Be concise and data-driven."""

        messages = [
            {"role": "system", "content": system_prompt + "\n\nCONTEXT:\n" + context_str},
            {"role": "user", "content": user_query}
        ]

        start_time = time.time()
        
        try:
            self._load_model(model_name)
            
            input_ids = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(self.device)
            
            prompt_length = input_ids.shape[1]
            
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response_text = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
            end_time = time.time()
            
            return {
                "answer": response_text.strip(),
                "model": self.models[model_name],
                "model_name": model_name,
                "response_time": end_time - start_time,
                "token_count": len(outputs[0]) - prompt_length,
                "prompt_token_count": prompt_length,
                "merged_context": merged_context,
                "success": True
            }
        
        except Exception as e:
            return {
                "error": str(e),
                "model": self.models[model_name],
                "model_name": model_name,
                "response_time": time.time() - start_time,
                "success": False
            }

    def compare_models(self, user_query, baseline_results, embedding_results):
        results = {}
        for model_name in ["llama", "qwen", "smollm"]:
            print(f"\n--- Testing {model_name.upper()} ---")
            results[model_name] = self.generate_response(
                user_query, baseline_results, embedding_results, model_name
            )
        return results

# ----------------------------
# 4. Model Evaluator
# ----------------------------
class ModelEvaluator:
    def evaluate_response(self, response: Dict, ground_truth: str = None, user_rating: int = None) -> Dict:
        if not response.get("success"):
            return {"error": response.get("error"), "success": False}
        
        answer = response.get("answer", "")
        metrics = {
            "response_time_seconds": response["response_time"],
            "output_tokens": response.get("token_count", 0),
            "input_tokens": response.get("prompt_token_count", 0),
            "answer_length_chars": len(answer),
            "answer_length_words": len(answer.split()),
            "user_rating": user_rating,
            "contains_numbers": bool(any(c.isdigit() for c in answer)),
            "success": True
        }
        return metrics

    def compare_models_report(self, comparison_results: Dict, evaluations: Dict = None) -> str:
        report = ["\n" + "="*80, "MODEL COMPARISON REPORT", "="*80]
        for model_name in ["llama", "qwen", "smollm"]:
            result = comparison_results.get(model_name, {})
            report.append(f"\nMODEL: {model_name.upper()}")
            if not result.get("success"):
                report.append(f"❌ ERROR: {result.get('error')}")
                continue
            report.append(f"⏱️  Time: {result['response_time']:.2f}s")
            report.append(f"📝 Answer: {result['answer'][:200]}...")
        return "\n".join(report)

    def generate_summary_table(self, comparison_results: Dict, evaluations: Dict = None) -> str:
        from io import StringIO
        output = StringIO()
        output.write(f"\n{'Model':<15} {'Time (s)':<12} {'Tokens':<12}\n")
        output.write("-"*40 + "\n")
        for model_name in ["llama", "qwen", "smollm"]:
            result = comparison_results.get(model_name, {})
            if not result.get("success"):
                output.write(f"{model_name.upper():<15} ERROR\n")
                continue
            output.write(f"{model_name.upper():<15} {result['response_time']:.2f}{'':<8} {result['token_count']}\n")
        return output.getvalue()