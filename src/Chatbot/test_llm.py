import sys
import os
import argparse
import dotenv
import gc
import torch
from transformers import PreTrainedModel
# 1. FIX: Handle fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from retriever import FPLHybridRetriever
from llm_client import LLMClient, ModelEvaluator

def load_config():
    """Load env variables dynamically"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(current_dir, '.env')
    loaded = dotenv.load_dotenv(dotenv_path)
    if not loaded: loaded = dotenv.load_dotenv(dotenv.find_dotenv())
    if not loaded:
        print("\n❌ Error: Could not find .env file.")
        sys.exit(1)
    token = os.getenv("HF_API_TOKEN")
    if not token:
        print("\n❌ Error: HF_API_TOKEN variable is missing from your .env file.")
        sys.exit(1)
    return token

def nuclear_cleanup():
    """Aggressively hunt down and destroy models in VRAM"""
    print("🧹 Starting Nuclear VRAM Cleanup...")
    
    # 1. Force Python Garbage Collection
    gc.collect()
    
    # 2. Hunt for any HuggingFace models still in memory
    for obj in gc.get_objects():
        try:
            if isinstance(obj, PreTrainedModel):
                print(f"   💀 Killing lingering model: {type(obj).__name__}")
                obj.cpu() # Move to CPU first
                del obj
        except Exception:
            pass # Ignore errors accessing objects
            
    # 3. Clear CUDA Cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 4. Report Free Memory
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"   ✨ GPU Free Memory: {free_mem:.2f} GB")

def test_single_model(model_name="llama"):
    hf_token = load_config()
    print(f"\n🚀 Initializing Single Model Test: {model_name.upper()}...")
    
    # --- PHASE 1: RETRIEVAL ---
    print("📊 Retrieving data (Loading Qwen for Intent)...")
    retriever = FPLHybridRetriever()
    query = "Who are the top 3 goal scorers?"
    kg_results = retriever.retrieve(query, use_embeddings=False)  # Disable embeddings for leaderboard
    
    # DEBUG: Print what we got
    print(f"\n🔍 DEBUG: Got {len(kg_results['baseline'].get('results', []))} result sets")
    if kg_results['baseline'].get('results'):
        print(f"🔍 DEBUG: First result has {len(kg_results['baseline']['results'][0].get('data', []))} records")
        print(f"🔍 DEBUG: Sample data: {kg_results['baseline']['results'][0]['data'][0]}")
    
    # --- PHASE 2: NUCLEAR CLEANUP ---
    print("\n🛑 Closing Retriever...")
    retriever.close()
    del retriever # Delete local reference
    nuclear_cleanup() # Kill the actual Qwen model objects
    
    # --- PHASE 3: GENERATION ---
    print(f"\n🤖 Generating response with {model_name}...")
    client = LLMClient(hf_api_token=hf_token)
    
    try:
        result = client.generate_response(
            query, 
            kg_results['baseline'], 
            kg_results['semantic_search'], 
            model_name=model_name
        )
        
        if result['success']:
            print(f"\n{'='*60}")
            print(f"RESPONSE ({result['response_time']:.2f}s)")
            print(f"{'='*60}")
            print(result['answer'])
            print(f"{'='*60}")
        else:
            print(f"\n❌ Error: {result.get('error')}")
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n❌ OOM Error! The model is still too big for 6GB VRAM.")
            print("   Try using 'smollm' which is smaller.")

def test_comparison():
    hf_token = load_config()
    print("\n🚀 Initializing Full Comparison...")
    
    query = "give me a good goalkeeper with at least 5 cleansheets"
    ground_truth = "good goalkeeper with at least 5 cleansheets"
    
    print(f"\n📊 Retrieving data...")
    retriever = FPLHybridRetriever()
    kg_results = retriever.retrieve(query, use_embeddings=False)  # Disable embeddings for leaderboard
    
    print("\n🛑 Cleaning up Retriever...")
    retriever.close()
    del retriever
    nuclear_cleanup()
    
    print("\n🤖 Looping through models...")
    client = LLMClient(hf_api_token=hf_token)
    evaluator = ModelEvaluator()
    
    # Run compare but force cleanup BETWEEN models
    comparison = {}
    for name in ["llama", "qwen", "smollm"]:
        print(f"\n--- Testing {name.upper()} ---")
        nuclear_cleanup() # Clean before each load
        result = client.generate_response(query, kg_results['baseline'], kg_results['semantic_search'], name)
        comparison[name] = result
        if result['success']:
             print(f"✓ Success ({result['response_time']:.2f}s)")
        else:
             print(f"✗ Failed: {result.get('error')}")

    evaluations = {}
    for name, result in comparison.items():
        if result.get("success"):
            evaluations[name] = evaluator.evaluate_response(result, ground_truth=ground_truth)
            
    print(evaluator.compare_models_report(comparison, evaluations))
    print(evaluator.generate_summary_table(comparison, evaluations))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--model", type=str, default="llama", choices=["llama", "qwen", "smollm"])
    args = parser.parse_args()
    
    if args.limit:
        # Simple retrieval test
        retriever = FPLHybridRetriever()
        retriever.retrieve("Test", use_embeddings=False)
        retriever.close()
    elif args.compare:
        test_comparison()
    else:
        test_single_model(args.model)