from retriever import FPLHybridRetriever
from llm_client import LLMClient, ModelEvaluator

def test_dynamic_limit():
    """Test dynamic LIMIT extraction without LLM calls"""
    
    # Initialize
    print("\n🚀 Initializing FPL Retriever...")
    retriever = FPLHybridRetriever()
    
    # Test cases with different limits
    test_queries = [
        ("Who are the top 3 goal scorers?", 3),
        ("Show me the best 5 assist providers", 5),
        ("Top 10 point scorers this season", 10),
        ("Who are the top goal scorers?", 10),  # No number - should default to 10
    ]
    
    print("\n" + "="*80)
    print("TESTING DYNAMIC LIMIT EXTRACTION")
    print("="*80)
    
    for query, expected_limit in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Expected Limit: {expected_limit}")
        print(f"{'='*60}")
        
        # Get KG results (without semantic embeddings for speed)
        kg_results = retriever.retrieve(query, use_embeddings=False)
        
        # Check if limit was applied correctly
        baseline = kg_results.get('baseline', {})
        if baseline.get('results'):
            actual_records = len(baseline['results'][0]['data'])
            print(f"\n✓ Results: {actual_records} records returned")
            if actual_records == expected_limit:
                print(f"✅ PASS: Got expected {expected_limit} records")
            else:
                print(f"❌ FAIL: Expected {expected_limit} but got {actual_records}")
        else:
            print("\n⚠ No results returned")
    
    # Cleanup
    retriever.close()
    print("\n✅ Testing complete!")

def test_llm_system():
    """Test complete LLM integration with all three models"""
    
    # Initialize
    print("\n🚀 Initializing FPL AI System...")
    retriever = FPLHybridRetriever()
    client = LLMClient()
    evaluator = ModelEvaluator()
    
    # Test queries
    test_cases = [
            {
                "query": "Who are the top 5 goal scorers?",
                "ground_truth": "top 5 goal scoring players with statistics"
            }
        ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}/{len(test_cases)}")
        print(f"{'='*80}")
        print(f"Query: {case['query']}")
        
        # Step 1: Get KG results
        print("\n📊 Retrieving from Knowledge Graph...")
        kg_results = retriever.retrieve(case['query'], use_embeddings=True)
        
        # Step 2: Compare all three models
        print("\n🤖 Generating responses with 3 models...")
        comparison = client.compare_models(
            case['query'],
            kg_results['baseline'],
            kg_results['semantic_search']
        )
        
        # Step 3: Evaluate each model
        evaluations = {}
        for model_name in ["llama", "qwen", "phi3"]:
            if comparison[model_name].get("success"):
                evaluations[model_name] = evaluator.evaluate_response(
                    comparison[model_name],
                    ground_truth=case['ground_truth']
                )
        
        # Step 4: Print results
        print(evaluator.compare_models_report(comparison, evaluations))
        print(evaluator.generate_summary_table(comparison, evaluations))
    
    # Cleanup
    retriever.close()
    print("\nTesting complete!")

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--test-limit":
        test_dynamic_limit()
    else:
        test_llm_system()
