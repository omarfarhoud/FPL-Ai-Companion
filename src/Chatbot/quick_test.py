import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever import FPLHybridRetriever
from llm_client import LLMClient, ModelEvaluator

def quick_test():
    """Quick test with single query and single model"""
    
    print("\n🚀 Quick Test - Single Query, Single Model")
    print("="*60)
    
    # Initialize
    retriever = FPLHybridRetriever()
    client = LLMClient()
    evaluator = ModelEvaluator()
    
    # Single test query
    query = "Who are the top 3 goal scorers?"
    
    print(f"\nQuery: {query}")
    print("\n📊 Retrieving from Knowledge Graph...")
    
    # Get results
    kg_results = retriever.retrieve(query, use_embeddings=True)
    
    print(f"✓ Found {len(kg_results['baseline'].get('results', []))} structured results")
    print(f"✓ Found {len(kg_results['semantic_search'].get('model1', []))} semantic matches")
    
    # Test with Llama only (fastest)
    print("\n🤖 Generating response with Llama 3.2...")
    
    response = client.generate_response(
        query,
        kg_results['baseline'],
        kg_results['semantic_search'],
        model_name='llama'
    )
    
    if response.get("success"):
        print(f"\n✅ Response generated in {response['response_time']:.2f}s")
        print(f"📊 Tokens: {response['token_count']}")
        print(f"\n{'='*60}")
        print("ANSWER:")
        print('='*60)
        print(response['answer'])
        print('='*60)
        
        # Evaluate
        metrics = evaluator.evaluate_response(response)
        print(f"\n📈 METRICS:")
        print(f"  • Contains Statistics: {'✓' if metrics['contains_numbers'] else '✗'}")
        print(f"  • Cites Context: {'✓' if metrics['cites_context'] else '✗'}")
        print(f"  • Completeness: {metrics['answer_completeness']}")
        print(f"  • Response Time: {metrics['response_time_seconds']:.2f}s")
        print(f"  • Answer Length: {metrics['answer_length_words']} words")
    else:
        print(f"\n❌ Error: {response.get('error')}")
    
    retriever.close()
    print("\n✅ Test complete!")

if __name__ == "__main__":
    quick_test()
