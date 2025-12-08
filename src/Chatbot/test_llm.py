from retriever import FPLHybridRetriever
from llm_client import LLMClient, ModelEvaluator

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
            "query": "Who are the best forwards under 8 million pounds?",
            "ground_truth": "top scoring forwards with value under 8m"
        },
        {
            "query": "How many goals did Salah score?",
            "ground_truth": "Salah's goal statistics"
        },
        {
            "query": "Compare Haaland and Kane's performance",
            "ground_truth": "comparison of Haaland and Kane stats"
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
        for model_name in ["llama", "mistral", "phi3"]:
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
    test_llm_system()