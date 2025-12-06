import os
import pickle
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import your custom classes
from intent_classifier import IntentClassifier
from entity_extractor import HybridEntityExtractor


# ----------------------------
# Utility: load config
# ----------------------------
def load_config():
    config = {}
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, '..', 'kg', 'config.txt')
    config_path = os.path.abspath(config_path)
    print(f"Looking for config at: {config_path}")
    
    with open(config_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                config[key] = value
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
                         avg(r.value) AS avg_value
                    WHERE total_points > 20
                    RETURN p.player_name, total_points, avg_value
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
            7: {  # Team clean sheet ranking
                "cypher": """
                    MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {name: 'Defender'})
                    MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
                    MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team)
                    WITH t.name AS team, sum(r.clean_sheets) AS total_clean_sheets
                    RETURN team, total_clean_sheets
                    ORDER BY total_clean_sheets DESC
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
                         avg(r.value) AS avg_value
                    WHERE avg_value > 0
                    WITH p, total_points, avg_value, 
                         (total_points * 1.0 / avg_value) AS points_per_value
                    RETURN p.player_name, 
                           total_points, 
                           avg_value, 
                           points_per_value
                    ORDER BY points_per_value DESC
                    LIMIT 10
                """,
                "required": ["position"]
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


# ----------------------------
# 2. Enhanced Embedding Retriever (TWO models)
# ----------------------------
class FPLEmbeddingRetriever:
    def __init__(self, 
                 model1="sentence-transformers/all-MiniLM-L6-v2",
                 model2="sentence-transformers/all-mpnet-base-v2",
                 cache_dir="embeddings_cache"):
        config = load_config()
        self.driver = GraphDatabase.driver(config['URI'], auth=(config['USERNAME'], config['PASSWORD']))
        
        # Cache directory
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache file paths
        self.cache_file_m1 = os.path.join(cache_dir, f"embeddings_model1.pkl")
        self.cache_file_m2 = os.path.join(cache_dir, f"embeddings_model2.pkl")
        self.cache_file_metadata = os.path.join(cache_dir, f"embeddings_metadata.pkl")
        
        # Load TWO embedding models for comparison
        print(f"Loading Model 1: {model1}")
        self.model1 = SentenceTransformer(model1)
        print(f"Loading Model 2: {model2}")
        self.model2 = SentenceTransformer(model2)
        
        self.node_embeddings_m1 = {}  # node_id -> embedding (model 1)
        self.node_embeddings_m2 = {}  # node_id -> embedding (model 2)
        self.node_metadata = {}       # node_id -> metadata

    def close(self):
        self.driver.close()
    
    def _load_from_cache(self):
        """Load embeddings from cache if available"""
        if (os.path.exists(self.cache_file_m1) and 
            os.path.exists(self.cache_file_m2) and 
            os.path.exists(self.cache_file_metadata)):
            
            print("Loading embeddings from cache...")
            with open(self.cache_file_m1, 'rb') as f:
                self.node_embeddings_m1 = pickle.load(f)
            with open(self.cache_file_m2, 'rb') as f:
                self.node_embeddings_m2 = pickle.load(f)
            with open(self.cache_file_metadata, 'rb') as f:
                self.node_metadata = pickle.load(f)
            print(f"✓ Loaded {len(self.node_embeddings_m1)} embeddings from cache")
            return True
        return False
    
    def _save_to_cache(self):
        """Save embeddings to cache"""
        print("Saving embeddings to cache...")
        with open(self.cache_file_m1, 'wb') as f:
            pickle.dump(self.node_embeddings_m1, f)
        with open(self.cache_file_m2, 'wb') as f:
            pickle.dump(self.node_embeddings_m2, f)
        with open(self.cache_file_metadata, 'wb') as f:
            pickle.dump(self.node_metadata, f)
        print(f"✓ Saved embeddings to {self.cache_dir}/")

    def embed_nodes(self, force_regenerate=False):
        """Create embeddings for all player nodes (aggregated stats)"""
        # Try to load from cache first
        if not force_regenerate and self._load_from_cache():
            return
        
        print("\nGenerating embeddings for all players...")
        
        with self.driver.session() as session:
            # Fetch aggregated player data
            result = session.run("""
                MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
                MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
                WITH p, pos,
                     sum(r.total_points) AS total_points,
                     sum(r.goals_scored) AS goals,
                     sum(r.assists) AS assists,
                     avg(r.value) AS avg_value,
                     avg(r.form) AS avg_form,
                     sum(r.minutes) AS total_minutes
                RETURN p.player_name AS name,
                       pos.name AS position,
                       total_points,
                       goals,
                       assists,
                       avg_value,
                       avg_form,
                       total_minutes
                LIMIT 1000
            """)
            
            for record in result:
                # Create rich text representation
                text_repr = self._create_text_representation(record)
                node_id = record['name']
                
                # Generate embeddings with both models
                self.node_embeddings_m1[node_id] = self.model1.encode(text_repr)
                self.node_embeddings_m2[node_id] = self.model2.encode(text_repr)
                
                # Store metadata
                self.node_metadata[node_id] = {
                    "name": record['name'],
                    "position": record['position'],
                    "total_points": record['total_points'],
                    "goals": record['goals'],
                    "assists": record['assists'],
                    "avg_value": round(record['avg_value'] / 10.0, 1) if record['avg_value'] else 0,  # Convert to millions
                    "avg_form": round(record['avg_form'], 2) if record['avg_form'] else 0
                }
        
        print(f"✓ Generated embeddings for {len(self.node_embeddings_m1)} players")
        
        # Save to cache
        self._save_to_cache()

    def _create_text_representation(self, record):
        """Create rich text representation for embedding"""
        avg_value_m = record['avg_value'] / 10.0 if record['avg_value'] else 0
        
        return (f"Player: {record['name']}, "
                f"Position: {record['position']}, "
                f"Total Points: {record['total_points']}, "
                f"Goals: {record['goals']}, "
                f"Assists: {record['assists']}, "
                f"Average Value: £{avg_value_m:.1f}m, "
                f"Average Form: {record['avg_form']:.2f}, "
                f"Total Minutes: {record['total_minutes']}")

    def embedding_search(self, query_text, top_k=5, model="both"):
        """
        Semantic search using embeddings
        
        Args:
            query_text: User query
            top_k: Number of results to return
            model: "model1", "model2", or "both" (for comparison)
        """
        results = {}
        
        if model in ["model1", "both"]:
            results["model1"] = self._search_with_model(
                query_text, self.model1, self.node_embeddings_m1, top_k
            )
        
        if model in ["model2", "both"]:
            results["model2"] = self._search_with_model(
                query_text, self.model2, self.node_embeddings_m2, top_k
            )
        
        return results

    def _search_with_model(self, query_text, model, embeddings_dict, top_k):
        """Perform search with a specific model"""
        query_vec = model.encode(query_text)
        similarities = []
        
        for node_id, node_vec in embeddings_dict.items():
            sim = cosine_similarity([query_vec], [node_vec])[0][0]
            similarities.append((node_id, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # Add metadata
        enriched_results = []
        for node_id, score in top_results:
            result = self.node_metadata[node_id].copy()
            result['similarity_score'] = float(score)
            enriched_results.append(result)
        
        return enriched_results


# ----------------------------
# 3. Hybrid Retriever (Combines both approaches)
# ----------------------------
class FPLHybridRetriever:
    def __init__(self):
        self.baseline = FPLBaselineRetriever()
        self.embedding = FPLEmbeddingRetriever()
        self.embedding.embed_nodes()
    
    def retrieve(self, user_query, use_embeddings=True):
        """
        Hybrid retrieval combining structured queries and semantic search
        """
        # Get baseline results
        baseline_results = self.baseline.retrieve(user_query)
        
        if not use_embeddings:
            return baseline_results
        
        # Enhance with embedding search
        embedding_results = self.embedding.embedding_search(user_query, top_k=5, model="both")
        
        return {
            "baseline": baseline_results,
            "semantic_search": embedding_results
        }
    
    def close(self):
        self.baseline.close()
        self.embedding.close()


# ----------------------------
# Example Usage & Testing
# ----------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING FPL RETRIEVAL SYSTEM")
    print("="*60)
    
    # Test queries
    test_queries = [
        "How many goals did Salah score this season?",
        "Compare Haaland and Kane",
        "Best defenders under 5 million",
        "Who should I captain this week?",
        "How is Liverpool performing?",
        "Who does Chelsea play next?"
    ]
    
    # Test Baseline
    print("\n" + "="*60)
    print("EXPERIMENT 1: BASELINE RETRIEVER")
    print("="*60)
    baseline = FPLBaselineRetriever()
    
    for query in test_queries[:3]:  # Test first 3
        result = baseline.retrieve(query)
        print(f"\n{'='*60}")
        print(f"RESULTS FOR: {query}")
        print(f"{'='*60}")
        if result['results']:
            for res in result['results']:
                print(f"\nQuery {res['query_id']} Results:")
                for row in res['data'][:3]:  # Show top 3
                    print(f"  {row}")
        else:
            print("No results found")
        print("-"*60)
    
    baseline.close()
    
    # Test Embedding
    print("\n" + "="*60)
    print("EXPERIMENT 2: EMBEDDING RETRIEVER (2 Models)")
    print("="*60)
    embedding = FPLEmbeddingRetriever()
    embedding.embed_nodes()  # Will load from cache on subsequent runs
    
    for query in test_queries[:3]:
        result = embedding.embedding_search(query, top_k=3, model="both")
        print(f"\n{'='*60}")
        print(f"RESULTS FOR: {query}")
        print(f"{'='*60}")
        print(f"\nModel 1 (all-MiniLM-L6-v2):")
        for row in result.get('model1', []):
            print(f"  {row}")
        print(f"\nModel 2 (all-mpnet-base-v2):")
        for row in result.get('model2', []):
            print(f"  {row}")
        print("-"*60)
    
    embedding.close()
    
    print("\n✓ Testing complete!")
    print("\nNote: Embeddings are cached in 'embeddings_cache/' directory.")
    print("To regenerate embeddings, delete this directory or use force_regenerate=True")