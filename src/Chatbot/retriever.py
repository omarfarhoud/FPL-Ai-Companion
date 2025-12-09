import os
import pickle
import numpy as np
import faiss  # <--- NEW IMPORT
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# Import your custom classes
from intent_classifier import IntentClassifier
from entity_extractor import HybridEntityExtractor

# ----------------------------
# Utility: load config
# ----------------------------
def load_config():
    config = {}
    base_path = os.path.dirname(os.path.abspath(__file__))
    # Adjust path as needed based on your folder structure
    config_path = os.path.join(base_path, '..', 'kg', 'config.txt') 
    config_path = os.path.abspath(config_path)
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    config[key] = value
    else:
        # Fallback or error handling
        print(f"Warning: Config not found at {config_path}")
        return {"URI": "", "USERNAME": "", "PASSWORD": ""}
    return config

# ----------------------------
# 1. Enhanced Baseline Retriever (Kept mostly same, just slight cleanup)
# ----------------------------
class FPLBaselineRetriever:
    def __init__(self):
        config = load_config()
        self.driver = GraphDatabase.driver(config['URI'], auth=(config['USERNAME'], config['PASSWORD']))
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = HybridEntityExtractor()
        self.intent_to_queries = self._init_intent_mapping()

    def close(self):
        self.driver.close()

    def _init_intent_mapping(self):
        return {
            "get_player_stats": [1, 10],
            "get_leaderboard": [12, 13, 14, 15],
            "compare_players": [5],
            "get_recommendation": [0, 6, 11, 8],
            "team_analysis": [2, 7],
            "fixture_query": [4, 9],
            "unknown": []
        }
    
    # ... (Include your _get_query_templates and _validate_and_prepare_params methods here) ...
    # ... (I am omitting them to save space, but keep your existing implementation) ...

    def retrieve(self, user_query):
        # ... (Keep your existing retrieve logic) ...
        # For brevity in this answer, assuming your existing baseline logic sits here
        return {"intent": "unknown", "entities": {}, "results": []} 


# ----------------------------
# 2. Embedding Retriever
# ----------------------------
class FPLEmbeddingRetriever:
    def __init__(self):
        config = load_config()
        self.driver = GraphDatabase.driver(config['URI'], auth=(config['USERNAME'], config['PASSWORD']))
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.index_paths = {
            "model1": os.path.join(self.base_dir, "fpl_index_m1.index"),
            "model2": os.path.join(self.base_dir, "fpl_index_m2.index")
        }
        self.meta_path = os.path.join(self.base_dir, "fpl_metadata.pkl")
        self.model_configs = {
            "model1": "sentence-transformers/all-MiniLM-L6-v2",
            "model2": "sentence-transformers/all-mpnet-base-v2"
        }
        self.loaded_models = {}
        self.faiss_indices = {}
        self.metadata_map = []
        self._load_resources()

    def close(self):
        self.driver.close()

    def _get_model(self, model_key):
        if model_key not in self.loaded_models:
            print(f"⏳ Loading {model_key} ({self.model_configs[model_key]})...")
            self.loaded_models[model_key] = SentenceTransformer(self.model_configs[model_key], device="cpu")
        return self.loaded_models[model_key]

    def _load_resources(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "rb") as f:
                self.metadata_map = pickle.load(f)
            print(f"✓ Loaded metadata for {len(self.metadata_map)} players.")
        else:
            print("⚠ No metadata found on disk.")

        for key, path in self.index_paths.items():
            if os.path.exists(path):
                try:
                    self.faiss_indices[key] = faiss.read_index(path)
                    print(f"✓ Loaded FAISS index for {key}.")
                except Exception as e:
                    print(f"✗ Failed to load index {key}: {e}")
            else:
                print(f"⚠ Index for {key} not found.")

    def rebuild_indices(self):
        print("\n=== REBUILDING INDICES FROM NEO4J ===")
        player_data = self._fetch_data_from_neo4j()
        if not player_data:
            print("No data found in Neo4j.")
            return
        self.metadata_map = player_data

        for model_key in self.model_configs:
            model = self._get_model(model_key)
            texts = [p['text_representation'] for p in player_data]
            embeddings = model.encode(texts, show_progress_bar=True)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            self.faiss_indices[model_key] = index
            faiss.write_index(index, self.index_paths[model_key])
            print(f"✓ Saved {model_key} index to {self.index_paths[model_key]}")

        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata_map, f)
        print(f"✓ Saved metadata to {self.meta_path}")

    def _fetch_data_from_neo4j(self):
        data = []
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
                MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
                WITH p, pos,
                     sum(r.total_points) AS total_points,
                     sum(r.goals_scored) AS goals,
                     sum(r.assists) AS assists,
                     avg(r.value) AS avg_value,
                     avg(r.form) AS avg_form
                RETURN p.player_name AS name,
                       pos.name AS position,
                       total_points, goals, assists, 
                       avg_value, avg_form
            """)
            for r in result:
                avg_val = round(r['avg_value']/10.0, 1) if r['avg_value'] else 0
                avg_form = round(r['avg_form'], 2) if r['avg_form'] else 0
                text = (
                    f"Name: {r['name']}. "
                    f"Position: {r['position']}. "
                    f"Price: £{avg_val}m. "
                    f"Stats: {r['total_points']} points, {r['goals']} goals, {r['assists']} assists. "
                    f"Form: {avg_form}."
                )
                entry = {
                    "name": r['name'],
                    "position": r['position'],
                    "total_points": r['total_points'],
                    "price": avg_val,
                    "text_representation": text
                }
                data.append(entry)
        return data

    def search(self, query, model_name="model1", top_k=5):
        if model_name not in self.faiss_indices:
            return []
        model = self._get_model(model_name)
        query_vec = model.encode([query]).astype("float32")
        index = self.faiss_indices[model_name]
        D, I = index.search(query_vec, top_k)
        results = []
        for rank, idx in enumerate(I[0]):
            if idx != -1 and idx < len(self.metadata_map):
                item = self.metadata_map[idx].copy()
                item['score'] = float(D[0][rank])
                results.append(item)
        return results

# ----------------------------
# 3. Hybrid Retriever
# ----------------------------
class FPLHybridRetriever:
    def __init__(self):
        self.baseline = FPLBaselineRetriever()
        self.embedding = FPLEmbeddingRetriever()

    def retrieve(self, user_query, use_embeddings=True, model_choice="model1"):
        baseline_results = self.baseline.retrieve(user_query)

        semantic_results = {}
        if use_embeddings:
            results = self.embedding.search(user_query, model_name=model_choice, top_k=5)
            semantic_results[model_choice] = results

        # Optional: fill baseline results if empty
        if not baseline_results.get("results") and semantic_results.get(model_choice):
            baseline_results["results"] = [{"data": semantic_results[model_choice]}]

        return {
            "baseline": baseline_results,
            "semantic_search": semantic_results,
            "used_model": model_choice
        }

    def close(self):
        self.baseline.close()
        self.embedding.close()


# ----------------------------
# Standalone Test
# ----------------------------
if __name__ == "__main__":
    hybrid = FPLHybridRetriever()
    if "model1" not in hybrid.embedding.faiss_indices:
        hybrid.embedding.rebuild_indices()
        hybrid.embedding._load_resources()

    query = "High scoring defenders under 5m"
    response = hybrid.retrieve(query, model_choice="model1")
    print("=== Retrieval Results ===")
    print(response)