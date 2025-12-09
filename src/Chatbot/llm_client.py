import time
import torch
import json
import gc
import re
from typing import Dict, List, Any, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# 0. Comparison Post-Processor
# ----------------------------
class ComparisonPostProcessor:
    """Post-process LLM responses to fix numerical comparison errors"""
    
    def process_response(self, response_text: str, intent: str, merged_context: Dict) -> str:
        """Main post-processing to fix comparison hallucinations"""
        try:
            if intent != "compare_players":
                return response_text
            
            # Extract ground truth from structured data
            if not merged_context.get("structured_data"):
                return response_text
            
            player_stats = {}
            for item in merged_context["structured_data"]:
                name_full = item.get('name', '')
                if not name_full:
                    continue
                    
                # Get last name for matching
                name = name_full.lower().split()[-1] if ' ' in name_full else name_full.lower()
                if name and name not in player_stats:  # Avoid duplicates
                    player_stats[name] = {
                        'goals': item.get('goals_scored', 0),
                        'assists': item.get('assists', 0),
                        'points': item.get('total_points', 0)
                    }
            
            # Need exactly 2 players for comparison
            if len(player_stats) != 2:
                return response_text
            
            # Get the two players
            players = list(player_stats.keys())
            p1, p2 = players[0], players[1]
            
            # Determine correct winners for each metric
            if player_stats[p1]['goals'] > player_stats[p2]['goals']:
                goals_winner = p1
            else:
                goals_winner = p2
            
            if player_stats[p1]['assists'] > player_stats[p2]['assists']:
                assists_winner = p1
            else:
                assists_winner = p2
            
            if player_stats[p1]['points'] > player_stats[p2]['points']:
                points_winner = p1
            else:
                points_winner = p2
            
            # Fix each line
            fixed_lines = []
            for line in response_text.split('\n'):
                line_lower = line.lower()
                
                # Fix goals statements
                if 'goal' in line_lower and ('more' in line_lower or 'fewer' in line_lower):
                    # Check if line incorrectly says p1 has more when p2 actually does
                    if p1 in line_lower and 'more' in line_lower and goals_winner == p2:
                        line = line.replace(p1.title(), p2.title())
                        line = line.replace(p1, p2)
                    elif p2 in line_lower and 'more' in line_lower and goals_winner == p1:
                        line = line.replace(p2.title(), p1.title())
                        line = line.replace(p2, p1)
                    # Fix "fewer" statements
                    elif p1 in line_lower and 'fewer' in line_lower and goals_winner == p1:
                        line = line.replace('fewer', 'more')
                    elif p2 in line_lower and 'fewer' in line_lower and goals_winner == p2:
                        line = line.replace('fewer', 'more')
                
                # Fix assists statements
                if 'assist' in line_lower and ('more' in line_lower or 'fewer' in line_lower):
                    if p1 in line_lower and 'more' in line_lower and assists_winner == p2:
                        line = line.replace(p1.title(), p2.title())
                        line = line.replace(p1, p2)
                    elif p2 in line_lower and 'more' in line_lower and assists_winner == p1:
                        line = line.replace(p2.title(), p1.title())
                        line = line.replace(p2, p1)
                    elif p1 in line_lower and 'fewer' in line_lower and assists_winner == p1:
                        line = line.replace('fewer', 'more')
                    elif p2 in line_lower and 'fewer' in line_lower and assists_winner == p2:
                        line = line.replace('fewer', 'more')
                
                # Fix points statements
                if 'point' in line_lower and ('more' in line_lower or 'higher' in line_lower or 'fewer' in line_lower):
                    if p1 in line_lower and ('more' in line_lower or 'higher' in line_lower) and points_winner == p2:
                        line = line.replace(p1.title(), p2.title())
                        line = line.replace(p1, p2)
                    elif p2 in line_lower and ('more' in line_lower or 'higher' in line_lower) and points_winner == p1:
                        line = line.replace(p2.title(), p1.title())
                        line = line.replace(p2, p1)
                
                fixed_lines.append(line)
            
            return '\n'.join(fixed_lines)
        
        except Exception as e:
            # If post-processing fails, return original response
            print(f"⚠️  Post-processing error: {e}")
            return response_text

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
# 2. Prompt Builder (3-Part Structure: PERSONA + CONTEXT + TASK)
# ----------------------------
class PromptBuilder:
    """Build structured prompts that work universally for any FPL query"""
    
    @staticmethod
    def build_persona() -> str:
        """
        PART 1: PERSONA - Define the assistant's role and expertise
        This stays the same for ALL queries
        """
        return """You are an FPL (Fantasy Premier League) expert assistant with comprehensive knowledge of:

• Player Performance Analysis: Goals, assists, points, form, value
• Team Dynamics: Clean sheets, fixtures, team statistics
• Strategic Recommendations: Budget optimization, captain picks, differential players
• Historical Data: Season trends, player comparisons, statistical patterns

Your approach:
- Provide accurate, data-driven insights
- Base ALL answers strictly on the provided knowledge base data
- Include relevant statistics to support your responses
- Acknowledge when information is insufficient"""

    @staticmethod
    def build_context(merged_context: Dict) -> str:
        """
        PART 2: CONTEXT - Format the retrieved KG data
        This changes based on what was retrieved from Neo4j
        """
        context_parts = ["\n\n### KNOWLEDGE BASE DATA ###\n"]
        
        # Add intent and entities for context
        intent = merged_context.get('intent', 'unknown')
        context_parts.append(f"Query Type: {intent.replace('_', ' ').title()}")
        
        if merged_context.get("entities"):
            entities_str = ", ".join([f"{k}: {v}" for k, v in merged_context['entities'].items() if v])
            context_parts.append(f"Detected Parameters: {entities_str}")
        
        context_parts.append("")
        
        # Format structured database results
        if merged_context.get("structured_data"):
            context_parts.append("**Database Query Results:**")
            for i, item in enumerate(merged_context["structured_data"][:15], 1):
                # Clean up field names (remove prefixes, format nicely)
                formatted_fields = []
                for key, value in item.items():
                    clean_key = key.replace("p.", "").replace("r.", "").replace("_", " ").title()
                    # Format numbers nicely
                    if isinstance(value, float):
                        value = round(value, 2)
                    formatted_fields.append(f"{clean_key}: {value}")
                
                context_parts.append(f"  {i}. {', '.join(formatted_fields)}")
            context_parts.append("")
        
        # Format semantic search matches (if any)
        if merged_context.get("semantic_matches"):
            context_parts.append("**Additional Relevant Players:**")
            for i, item in enumerate(merged_context["semantic_matches"][:5], 1):
                player_info = f"{item['name']} ({item['position']})"
                stats = f"Points: {item['total_points']}, Price: £{item.get('price', 'N/A')}m"
                context_parts.append(f"  {i}. {player_info} - {stats}")
            context_parts.append("")
        
        # If no data at all
        if not merged_context.get("structured_data") and not merged_context.get("semantic_matches"):
            context_parts.append("**No specific data available for this query.**")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    @staticmethod
    def build_task(user_query: str) -> str:
        """
        PART 3: TASK - Clear instructions on what to do with the context
        This is generic and works for ANY question
        """
        return f"""

### YOUR TASK ###

User Question: "{user_query}"

**CRITICAL RULES:**
1. If the Knowledge Base Data section above shows "No specific data available" or contains no relevant player information, you MUST respond with: "I don't have data available to answer this question."
2. NEVER make up player names, statistics, or comparisons that aren't explicitly shown in the Knowledge Base Data above
3. NEVER use information from your training data - ONLY use the data provided above

Instructions:
1. Answer using ONLY the data in the Knowledge Base Data section above
2. When comparing players, state the numbers clearly for each metric, then make your comparison
   - Example: "Salah scored 23 goals, Kane scored 30 goals. Kane has more goals."
3. Make sure your conclusion matches the numbers: if 14 > 9, then the player with 14 has MORE
4. If data is incomplete or missing, say: "I don't have enough data to answer this question."
5. Be concise and direct

Provide your answer below:"""

    def build_full_prompt(self, user_query: str, merged_context: Dict) -> str:
        """
        Combine all three parts into a complete structured prompt
        Works universally for ANY FPL query
        """
        persona = self.build_persona()
        context = self.build_context(merged_context)
        task = self.build_task(user_query)
        
        return f"{persona}{context}{task}"

# ----------------------------
# 3. LLM Client (Polished for GPU)
# ----------------------------
class LLMClient:
    def __init__(self, hf_api_token: str = None):
        self.hf_token = hf_api_token
        self.merger = ResultMerger()
        self.prompt_builder = PromptBuilder()
        self.post_processor = ComparisonPostProcessor()  # Add post-processor
        
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
        
        # Step 2: Build Universal 3-Part Structured Prompt
        full_prompt = self.prompt_builder.build_full_prompt(user_query, merged_context)

        messages = [
            {"role": "system", "content": full_prompt}
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
                attention_mask=torch.ones_like(input_ids),  # Fix attention mask warning
                max_new_tokens=200,  # Increased from 150 to prevent cutoff
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response_text = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
            
            # **POST-PROCESS: Fix comparison hallucinations**
            intent = baseline_results.get('intent', 'unknown')
            response_text = self.post_processor.process_response(
                response_text.strip(), 
                intent, 
                merged_context
            )
            
            end_time = time.time()
            
            return {
                "answer": response_text,
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
            report.append(f"📝 Answer: {result['answer']}")
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