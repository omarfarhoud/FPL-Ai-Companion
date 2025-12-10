import os
import pickle
import numpy as np
import faiss  # <--- NEW IMPORT
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import torch

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

# 1. Enhanced Baseline Retriever (FIXED FOR YOUR SCHEMA)

# ----------------------------

class FPLBaselineRetriever:

    def __init__(self):

        config = load_config()

        self.driver = GraphDatabase.driver(config['URI'], auth=(config['USERNAME'], config['PASSWORD']))

       

        # Initialize NLU components

        self.intent_classifier = IntentClassifier()

        self.entity_extractor = HybridEntityExtractor()

       

        # Map intents to query templates

        self.intent_to_queries = self._init_intent_mapping()



    def close(self):

        self.driver.close()



    def _init_intent_mapping(self):

        """Map each intent to relevant query template indices"""

        return {

            "get_player_stats": [1, 10],          # Player stats, recent form

            "compare_players": [5],                # Compare players

            "get_recommendation": [0, 6, 11, 8],   # Top by position, budget, value picks, captain

            "team_analysis": [2, 7],               # Team analysis, clean sheets

            "fixture_query": [4, 9],               # Fixtures, upcoming

            "get_leaderboard": [16, 12, 13, 14, 15],   # Leaderboard queries (cleansheets first, then goals, assists, points, bonus)

            "unknown": []

        }



    def _get_query_templates(self):

        """

        Define all 12 query templates MATCHING YOUR SCHEMA

        Key: Stats are on PLAYED_IN relationship, not Player node!

        """

        return {

            0: {  # Top players by position (aggregated across all fixtures)

                "cypher": """

                    MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)

                    MATCH (p)-[r:PLAYED_IN]->(f:Fixture)

                    WHERE pos.name = $position

                    WITH p, pos,

                         sum(r.total_points) AS total_points,

                         avg(r.value) AS avg_value,

                         sum(r.clean_sheets) AS clean_sheets

                    WHERE total_points > 20

                      AND ($min_cleansheets IS NULL OR clean_sheets >= $min_cleansheets)

                    RETURN p.player_name, total_points, avg_value, clean_sheets

                    ORDER BY total_points DESC

                    LIMIT 10

                """,

                "required": ["position"]

            },

            1: {  # Player total stats (aggregated across all fixtures)

                "cypher": """

                    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)

                    WHERE toLower(p.player_name) CONTAINS toLower($player)

                    WITH p,

                         sum(r.total_points) AS total_points,

                         sum(r.goals_scored) AS goals_scored,

                         sum(r.assists) AS assists,

                         sum(r.minutes) AS minutes_played,

                         avg(r.value) AS avg_value,

                         CASE WHEN toLower(p.player_name) = toLower($player) THEN 0

                              WHEN toLower(p.player_name) ENDS WITH ' ' + toLower($player) THEN 1

                              ELSE 2

                         END AS match_priority

                    ORDER BY match_priority, total_points DESC

                    RETURN p.player_name,

                           total_points,

                           goals_scored,

                           assists,

                           minutes_played,

                           avg_value

                    LIMIT 1

                """,

                "required": ["player"]

            },

            2: {  # Team analysis (players by team from fixtures)

                "cypher": """

                    MATCH (f:Fixture)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team {name: $team})

                    MATCH (p:Player)-[r:PLAYED_IN]->(f)

                    MATCH (p)-[:PLAYS_AS]->(pos:Position)

                    RETURN pos.name AS position,

                           count(DISTINCT p) AS player_count,

                           avg(r.total_points) AS avg_points,

                           sum(r.total_points) AS total_points

                    ORDER BY total_points DESC

                """,

                "required": ["team"]

            },

            3: {  # Player performance in specific gameweek

                "cypher": """

                    MATCH (gw:Gameweek {GW_number: $gameweek})-[:HAS_FIXTURE]->(f:Fixture)

                    MATCH (p:Player)-[r:PLAYED_IN]->(f)

                    WHERE toLower(p.player_name) CONTAINS toLower($player)

                    RETURN p.player_name,

                           r.total_points,

                           r.goals_scored,

                           r.assists,

                           r.minutes,

                           r.bonus,

                           gw.GW_number

                    LIMIT 1

                """,

                "required": ["player", "gameweek"]

            },

            4: {  # Fixtures for a team

                "cypher": """

                    MATCH (f:Fixture)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team {name: $team})

                    RETURN f.fixture_number,

                           f.kickoff_time,

                           f.season

                    ORDER BY f.fixture_number

                    LIMIT 10

                """,

                "required": ["team"]

            },

            5: {  # Compare players (aggregated stats) - FUZZY MATCH

                "cypher": """

                    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)

                    WHERE ANY(name IN $players WHERE toLower(p.player_name) CONTAINS toLower(name))

                    WITH p,

                         sum(r.total_points) AS total_points,

                         sum(r.goals_scored) AS goals_scored,

                         sum(r.assists) AS assists,

                         avg(r.value) AS avg_value

                    RETURN p.player_name,

                           total_points,

                           goals_scored,

                           assists,

                           avg_value

                    ORDER BY total_points DESC

                """,

                "required": ["players"]

            },

            6: {  # Players under budget (by average value) - NO REQUIRED PARAMS

                "cypher": """

                    MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)

                    MATCH (p)-[r:PLAYED_IN]->(f:Fixture)

                    WITH p, pos,

                         avg(r.value) AS avg_value,

                         sum(r.total_points) AS total_points

                    WHERE avg_value <= $max_value AND total_points > 0

                    RETURN p.player_name,

                           total_points,

                           avg_value,

                           pos.name AS position

                    ORDER BY total_points DESC

                    LIMIT 10

                """,

                "required": []  # Changed from ["max_value"] since we provide it as default

            },

            7: {  # Team clean sheet ranking (count fixtures with 0 goals conceded)

                "cypher": """

                    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)

                    MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team)

                    WITH t, f, sum(r.goals_conceded) AS team_goals_conceded

                    WHERE team_goals_conceded = 0

                    WITH t.name AS team, count(DISTINCT f) AS clean_sheets

                    RETURN team, clean_sheets

                    ORDER BY clean_sheets DESC

                    LIMIT 10

                """,

                "required": []

            },

            8: {  # Captain recommendation (top total points)

                "cypher": """

                    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)

                    WITH p, sum(r.total_points) AS total_points, avg(r.form) AS avg_form

                    RETURN p.player_name,

                           total_points,

                           avg_form

                    ORDER BY total_points DESC

                    LIMIT 5

                """,

                "required": []

            },

            9: {  # Gameweek top scorers

                "cypher": """

                    MATCH (gw:Gameweek {GW_number: $gameweek})-[:HAS_FIXTURE]->(f:Fixture)

                    MATCH (p:Player)-[r:PLAYED_IN]->(f)

                    RETURN p.player_name,

                           sum(r.total_points) AS total_points

                    ORDER BY total_points DESC

                    LIMIT 5

                """,

                "required": ["gameweek"]

            },

            10: {  # Player recent form (last 5 fixtures) - get best match first

                "cypher": """

                    // First, find the best matching player

                    MATCH (p:Player)

                    WHERE toLower(p.player_name) CONTAINS toLower($player)

                    WITH p,

                         CASE

                             WHEN toLower(p.player_name) ENDS WITH ' ' + toLower($player) THEN 0

                             WHEN toLower(p.player_name) = toLower($player) THEN 0

                             ELSE size(p.player_name)

                         END AS match_score

                    ORDER BY match_score

                    LIMIT 1

                    // Then get their recent fixtures

                    MATCH (p)-[r:PLAYED_IN]->(f:Fixture)

                    RETURN p.player_name,

                           f.fixture_number,

                           r.total_points,

                           r.goals_scored,

                           r.assists

                    ORDER BY f.fixture_number DESC

                    LIMIT 5

                """,

                "required": ["player"]

            },

            11: {  # Position-based value picks (points per value)

                "cypher": """

                    MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {name: $position})

                    MATCH (p)-[r:PLAYED_IN]->(f:Fixture)

                    WITH p,

                         sum(r.total_points) AS total_points,

                         avg(r.value) AS avg_value,

                         sum(r.clean_sheets) AS clean_sheets

                    WHERE avg_value > 0

                      AND ($min_cleansheets IS NULL OR clean_sheets >= $min_cleansheets)

                    WITH p, total_points, avg_value, clean_sheets,

                         (total_points * 1.0 / avg_value) AS points_per_value

                    RETURN p.player_name,

                           total_points,

                           avg_value,

                           points_per_value,

                           clean_sheets

                    ORDER BY points_per_value DESC

                    LIMIT 10

                """,

                "required": ["position"]

            },

            12: {  # Top goal scorers (leaderboard)
                "cypher": """
                    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                    MATCH (p)-[:PLAYS_AS]->(pos:Position)
                    WITH p, pos,
                         sum(r.goals_scored) AS total_goals,
                         sum(r.total_points) AS total_points,
                         avg(r.value) AS avg_value
                    WHERE total_goals > 0
                    RETURN p.player_name, 
                           pos.name AS position,
                           total_goals,
                           total_points,
                           avg_value
                    ORDER BY total_goals DESC
                    LIMIT $limit
                """,
                "required": []
            },
            13: {  # Top assist providers (leaderboard)
                "cypher": """
                    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                    MATCH (p)-[:PLAYS_AS]->(pos:Position)
                    WITH p, pos,
                         sum(r.assists) AS total_assists,
                         sum(r.total_points) AS total_points,
                         avg(r.value) AS avg_value
                    WHERE total_assists > 0
                    RETURN p.player_name, 
                           pos.name AS position,
                           total_assists,
                           total_points,
                           avg_value
                    ORDER BY total_assists DESC
                    LIMIT $limit
                """,
                "required": []
            },
            14: {  # Top point scorers (leaderboard)
                "cypher": """
                    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                    MATCH (p)-[:PLAYS_AS]->(pos:Position)
                    WITH p, pos,
                         sum(r.total_points) AS total_points,
                         avg(r.value) AS avg_value
                    WHERE total_points > 0
                    RETURN p.player_name, 
                           pos.name AS position,
                           total_points,
                           avg_value
                    ORDER BY total_points DESC
                    LIMIT $limit
                """,
                "required": []
            },
            15: {  # Top bonus point getters (leaderboard)
                "cypher": """
                    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                    MATCH (p)-[:PLAYS_AS]->(pos:Position)
                    WITH p, pos,
                         sum(r.bonus) AS total_bonus,
                         sum(r.total_points) AS total_points,
                         avg(r.value) AS avg_value
                    WHERE total_bonus > 0
                    RETURN p.player_name, 
                           pos.name AS position,
                           total_bonus,
                           total_points,
                           avg_value
                    ORDER BY total_bonus DESC
                    LIMIT $limit
                """,
                "required": []
            },
            16: {  # Top clean sheet keepers (leaderboard)
                "cypher": """
                    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                    MATCH (p)-[:PLAYS_AS]->(pos:Position)
                    WHERE ($position IS NULL OR pos.name = $position)
                    WITH p, pos,
                         sum(r.clean_sheets) AS total_clean_sheets,
                         sum(r.total_points) AS total_points,
                         avg(r.value) AS avg_value
                    WHERE total_clean_sheets > 0
                      AND ($min_cleansheets IS NULL OR total_clean_sheets >= $min_cleansheets)
                    RETURN p.player_name, 
                           pos.name AS position,
                           total_clean_sheets,
                           total_points,
                           avg_value
                    ORDER BY total_clean_sheets DESC
                    LIMIT $limit
                """,
                "required": []
            }

        }



    def _validate_and_prepare_params(self, entities, required_keys):

        """Validate entities and prepare query parameters"""

        params = {}

       

        # Handle list entities (take first element if list)

        for key in required_keys:

            # Special handling for 'players' key in compare query

            if key == "players":

                # Check both 'players' and 'player' keys

                if "players" in entities and entities["players"]:

                    params[key] = entities["players"] if isinstance(entities["players"], list) else [entities["players"]]

                elif "player" in entities and entities["player"]:

                    params[key] = entities["player"] if isinstance(entities["player"], list) else [entities["player"]]

                else:

                    return None

                continue

           

            if key not in entities or not entities[key]:

                return None  # Missing required entity

           

            value = entities[key]

            extracted_value = value[0] if isinstance(value, list) else value

           

            # Fix invalid season format

            if key == "season" and extracted_value == "YYYY-YY":

                extracted_value = "2023-24"

           

            # Convert gameweek to int if string

            if key == "gameweek":

                try:

                    extracted_value = int(str(extracted_value).replace("GW", "").strip())

                except:

                    extracted_value = 1

           

            params[key] = extracted_value

       

        # Add optional default parameters

        if "season" not in params:

            params["season"] = "2023-24"

        if "gameweek" not in params:

            params["gameweek"] = 1

        if "max_value" not in params:

            # Check for budget in entities and convert

            if "budget" in entities and entities["budget"]:

                budget_str = entities["budget"][0] if isinstance(entities["budget"], list) else entities["budget"]

                # Extract number from strings like "<5m", "5.5", "under 5"

                import re

                numbers = re.findall(r'\d+\.?\d*', str(budget_str))

                if numbers:

                    params["max_value"] = int(float(numbers[0]) * 10)  # Convert to tenths

                else:

                    params["max_value"] = 100

            else:

                params["max_value"] = 100  # Value is in 10ths (e.g., 100 = £10.0m)
        
        # Handle cleansheet constraint
        if "min_cleansheets" not in params:
            if "min_cleansheets" in entities and entities["min_cleansheets"]:
                params["min_cleansheets"] = entities["min_cleansheets"][0] if isinstance(entities["min_cleansheets"], list) else entities["min_cleansheets"]
            else:
                params["min_cleansheets"] = None
        
        # Handle limit parameter for leaderboard queries
        if "limit" not in params:
            # Check if limit was extracted from query and stored in entities
            if "limit" in entities and entities["limit"]:
                params["limit"] = int(entities["limit"])
            else:
                # Default to 10 if not already extracted from query
                params["limit"] = 10
        
        # Handle position parameter (optional for leaderboard queries)
        if "position" not in params:
            if "position" in entities and entities["position"]:
                params["position"] = entities["position"][0] if isinstance(entities["position"], list) else entities["position"]
            else:
                params["position"] = None

       

        return params



    def retrieve(self, user_query):

        """Main retrieval method"""

        print(f"\n{'='*60}")

        print(f"Processing query: {user_query}")

        print(f"{'='*60}")

       

        # Step 1: Classify intent

        intent = self.intent_classifier.predict(user_query)

        print(f"Classified Intent: {intent}")

       

        # Step 2: Extract entities

        entities = self.entity_extractor.extract(user_query)
        
        # Step 2.5: Extract number from query for "top N" patterns
        import re
        numbers = re.findall(r'(?:top|best|first)\s+(\d+)', user_query.lower())
        if numbers:
            entities["limit"] = int(numbers[0])

        print(f"Extracted Entities: {entities}")

       

        # Step 3: Get relevant query indices for this intent

        query_indices = self.intent_to_queries.get(intent, [])

        if not query_indices:

            return {"intent": intent, "message": "No relevant queries for this intent", "results": []}

       

        # Step 4: Execute relevant queries

        all_results = []

        query_templates = self._get_query_templates()

       

        with self.driver.session() as session:

            for idx in query_indices:

                template = query_templates[idx]

               

                # Prepare parameters

                params = self._validate_and_prepare_params(entities, template["required"])

               

                if params is None:

                    print(f"⚠ Query {idx} skipped: missing required entities {template['required']}")

                    continue

               

                try:

                    print(f"\n→ Executing Query {idx}")

                    print(f"  Parameters: {params}")

                   

                    result = session.run(template["cypher"], params)

                    records = [dict(record) for record in result]

                   

                    if records:

                        all_results.append({

                            "query_id": idx,

                            "data": records

                        })

                        print(f"  ✓ Retrieved {len(records)} records")

                    else:

                        print(f"  ⚠ No results found")

                       

                except Exception as e:

                    print(f"  ✗ Query {idx} failed: {e}")

       

        return {

            "intent": intent,

            "entities": entities,

            "results": all_results

        }



import os
import pickle
import faiss
import pandas as pd
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

class FPLEmbeddingRetriever:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.meta_path = os.path.join(self.base_dir, "fpl_metadata.pkl")
        self.index_paths = {
            "model1": os.path.join(self.base_dir, "fpl_index_m1.index"),
            "model2": os.path.join(self.base_dir, "fpl_index_m2.index")
        }
        self.model_configs = {
            "model1": "sentence-transformers/all-MiniLM-L6-v2",
            "model2": "sentence-transformers/all-mpnet-base-v2"
        }
        self.loaded_models = {}
        self.faiss_indices = {}
        self.metadata_map = []

        # Optional: Neo4j connection
        # config = self._load_config()
        # self.driver = GraphDatabase.driver(config['URI'], auth=(config['USERNAME'], config['PASSWORD']))

        self._load_resources()

    def close(self):
        if hasattr(self, "driver"):
            self.driver.close()

    # ----------------------------
    # Load FAISS + metadata
    # ----------------------------
    def _load_resources(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "rb") as f:
                self.metadata_map = pickle.load(f)
            print(f"✓ Loaded metadata for {len(self.metadata_map)} rows.")
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

    # ----------------------------
    # Internal: Load model
    # ----------------------------
    def _get_model(self, model_key, device="cpu"):
        if model_key not in self.loaded_models:
            print(f"⏳ Loading {model_key} ({self.model_configs[model_key]}) on {device}...")
            self.loaded_models[model_key] = SentenceTransformer(self.model_configs[model_key], device=device)
        return self.loaded_models[model_key]

    # ----------------------------
    # Load CSV and create text for embeddings
    # ----------------------------
    def _fetch_data_from_csv(self):
        csv_path = os.path.join(self.base_dir, "..", "kg", "fpl_two_seasons.csv")
        csv_path = os.path.abspath(csv_path)
        print(f"📄 Loading CSV from: {csv_path}")

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        player_data = []
        for _, row in df.iterrows():
            # Make a descriptive text of **all relevant info**
            text = " | ".join(
                [f"{col}: {row[col]}" for col in df.columns if pd.notnull(row[col])]
            )
            entry = row.to_dict()
            entry["text_representation"] = text
            player_data.append(entry)

        print(f"✓ Loaded {len(player_data)} rows from CSV.")
        return player_data

    # ----------------------------
    # Rebuild indices
    # ----------------------------
    def rebuild_indices(self, use_gpu=True):
        print("\n=== REBUILDING INDICES FROM CSV ===")
        player_data = self._fetch_data_from_csv()
        if not player_data:
            print("No data found in CSV.")
            return

        self.metadata_map = player_data

        for model_key in self.model_configs:
            device = "cuda" if use_gpu else "cpu"
            model = self._get_model(model_key, device=device)
            texts = [p['text_representation'] for p in player_data]

            print(f"⏳ Embedding {len(texts)} rows with {model_key} on {device}...")
            embeddings = model.encode(texts, show_progress_bar=True, batch_size=128, convert_to_numpy=True).astype("float32")
            
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)
            self.faiss_indices[model_key] = index
            faiss.write_index(index, self.index_paths[model_key])
            print(f"✓ Saved {model_key} index to {self.index_paths[model_key]}")

            # Clear GPU memory after each model
            if use_gpu:
                del model
                torch.cuda.empty_cache()
                print(f"♻ Cleared GPU memory after {model_key}")

        # Save metadata
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata_map, f)
        print(f"✓ Saved metadata to {self.meta_path}")

    # ----------------------------
    # Query FAISS
    # ----------------------------
    def search(self, query, model_name, top_k):
        if model_name not in self.faiss_indices:
            return []

        model = self._get_model(model_name, device="cpu")
        query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self.faiss_indices[model_name].search(query_vec, top_k)

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

    def retrieve(self, user_query, use_embeddings=True, model_choice="model2"):
        baseline_results = self.baseline.retrieve(user_query)

        semantic_results = {}
        if use_embeddings:
            results = self.embedding.search(user_query, model_name=model_choice, top_k=50)
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

    query = "Show me the fixtures for liverpool"
    response = hybrid.retrieve(query, model_choice="model1")
    print("=== Retrieval Results ===")
    print(response)