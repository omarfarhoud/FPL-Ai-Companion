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
# 2. Enhanced Embedding Retriever (Neo4j vector index)
# ----------------------------
class FPLEmbeddingRetriever:
    def __init__(self, 
                 model1="sentence-transformers/all-MiniLM-L6-v2",
                 model2="sentence-transformers/all-mpnet-base-v2"):
        config = load_config()
        self.driver = GraphDatabase.driver(config['URI'], auth=(config['USERNAME'], config['PASSWORD']))
        
        print(f"Loading Model 1: {model1}")
        self.model1 = SentenceTransformer(model1)
        print(f"Loading Model 2: {model2}")
        self.model2 = SentenceTransformer(model2)
        
        self.node_metadata = {}
        self.node_embeddings_m1 = {}
        self.node_embeddings_m2 = {}
        self.cache_loaded = self._load_from_neo4j()

    def close(self):
        self.driver.close()

    def _load_from_neo4j(self):
        """Load embeddings from Neo4j without limits"""
        with self.driver.session() as session:
            try:
                # REMOVED LIMIT 1000 to ensure full DB load
                result = session.run("""
                    MATCH (p:Player)
                    WHERE p.embedding_m1 IS NOT NULL AND p.embedding_m2 IS NOT NULL
                    RETURN p.player_name AS name,
                           p.embedding_m1 AS emb1,
                           p.embedding_m2 AS emb2
                """)
                
                loaded = 0
                for r in result:
                    if r["emb1"] and r["emb2"]:
                        self.node_embeddings_m1[r["name"]] = np.array(r["emb1"])
                        self.node_embeddings_m2[r["name"]] = np.array(r["emb2"])
                        loaded += 1
                
                # Load metadata separately (efficient batch fetch)
                if loaded > 0:
                    self._load_metadata_from_graph()
                
                print(f"✓ Loaded {loaded} embeddings from Neo4j")
                return loaded > 0
            
            except Exception as e:
                print(f"⚠ Could not load embeddings from Neo4j: {e}")
                return False

    def _load_metadata_from_graph(self):
        """Load player metadata from graph structure"""
        if not self.node_embeddings_m1:
            return

        with self.driver.session() as session:
            # We fetch metadata only for loaded players
            result = session.run("""
                MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
                MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
                WHERE p.player_name IN $names
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
            """, names=list(self.node_embeddings_m1.keys()))
            
            for r in result:
                self.node_metadata[r['name']] = {
                    "name": r['name'],
                    "position": r['position'],
                    "total_points": r['total_points'],
                    "goals": r['goals'],
                    "assists": r['assists'],
                    "avg_value": round(r['avg_value']/10.0, 1) if r['avg_value'] else 0,
                    "avg_form": round(r['avg_form'], 2) if r['avg_form'] else 0,
                    "total_minutes": r['total_minutes']
                }

    def _save_to_neo4j(self):
        """Persist generated embeddings to the database"""
        print("Saving embeddings to Neo4j (this may take a moment)...")
        with self.driver.session() as session:
            for node_id in self.node_embeddings_m1:
                session.run("""
                    MATCH (p:Player {player_name: $name})
                    SET p.embedding_m1 = $emb1,
                        p.embedding_m2 = $emb2
                """, name=node_id,
                     emb1=self.node_embeddings_m1[node_id].tolist(),
                     emb2=self.node_embeddings_m2[node_id].tolist())
        print("✓ Saved embeddings to Neo4j")

    def embed_nodes(self, force_regenerate=False):
        """Generate embeddings for all players in the DB"""
        if not force_regenerate and self.cache_loaded:
            return
        
        print("Generating new embeddings...")
        with self.driver.session() as session:
            # REMOVED LIMIT 1000
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
            """)
            self.node_embeddings_m1 = {}
            self.node_embeddings_m2 = {}
            
            count = 0
            for record in result:
                node_id = record['name']
                text_repr = self._create_text_representation(record)
                
                self.node_embeddings_m1[node_id] = self.model1.encode(text_repr)
                self.node_embeddings_m2[node_id] = self.model2.encode(text_repr)
                
                self.node_metadata[node_id] = {
                    "name": record['name'],
                    "position": record['position'],
                    "total_points": record['total_points'],
                    "goals": record['goals'],
                    "assists": record['assists'],
                    "avg_value": round(record['avg_value']/10.0,1) if record['avg_value'] else 0,
                    "avg_form": round(record['avg_form'],2) if record['avg_form'] else 0,
                    "total_minutes": record['total_minutes']
                }
                count += 1
                
        print(f"✓ Generated embeddings for {len(self.node_embeddings_m1)} players")
        self._save_to_neo4j()

    def _create_text_representation(self, record):
        """
        Creates a 'stuffed' text representation to improve vector specificity.
        Repeats key identifiers (Name, Position) to increase attention weight.
        """
        avg_value_m = record['avg_value']/10.0 if record['avg_value'] else 0
        
        text = (
            f"Name: {record['name']}. "
            f"Player: {record['name']}. "  # Repetition helps the vector focus on the name
            f"Position: {record['position']}. "
            f"Premier League Footballer playing as {record['position']}. "
            f"FPL Stats: {record['total_points']} total points, "
            f"{record['goals']} goals, {record['assists']} assists. "
            f"Current Price: £{avg_value_m:.1f} million. "
            f"Current Form: {record['avg_form']:.2f}. "
            f"Minutes played: {record['total_minutes']}."
        )
        return text

    def embedding_search(self, query_text, top_k=5, model="both", use_neo4j_index=False):
        """
        Search with option to use Neo4j vector index or manual similarity.
        Set use_neo4j_index=False to use Python-based cosine similarity (often faster for small datasets).
        """
        results = {}
        
        if use_neo4j_index:
            try:
                if model in ["model1", "both"]:
                    results["model1"] = self._search_with_neo4j(
                        query_text, top_k, index="playerEmbeddingIndex_m1"
                    )
                if model in ["model2", "both"]:
                    results["model2"] = self._search_with_neo4j(
                        query_text, top_k, index="playerEmbeddingIndex_m2"
                    )
            except Exception as e:
                print(f"⚠ Neo4j vector search failed: {e}")
                print("Falling back to manual similarity computation...")
                use_neo4j_index = False
        
        if not use_neo4j_index:
            # Use manual cosine similarity
            if model in ["model1", "both"]:
                results["model1"] = self._search_with_manual_similarity(
                    query_text, top_k, model_key="m1"
                )
            if model in ["model2", "both"]:
                results["model2"] = self._search_with_manual_similarity(
                    query_text, top_k, model_key="m2"
                )
        
        return results

    def _search_with_neo4j(self, query_text, top_k, index):
        """Execute vector search using Neo4j's native index"""
        model = self.model1 if "m1" in index else self.model2
        query_vec = model.encode(query_text).tolist()
        enriched_results = []
        
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes($index, $k, $vec)
                YIELD node, score
                RETURN node.player_name AS name, score
            """, index=index, vec=query_vec, k=top_k)
            
            for r in result:
                node_id = r["name"]
                score = r["score"]
                if node_id in self.node_metadata:
                    meta = self.node_metadata[node_id].copy()
                    meta["similarity_score"] = float(score)
                    enriched_results.append(meta)
        
        return enriched_results

    def _search_with_manual_similarity(self, query_text, top_k, model_key="m1"):
        """Fallback: compute similarities in Python"""
        model = self.model1 if model_key == "m1" else self.model2
        embeddings_dict = self.node_embeddings_m1 if model_key == "m1" else self.node_embeddings_m2
        
        if not embeddings_dict:
            return []

        # Encode query
        query_vec = model.encode(query_text).reshape(1, -1)
        
        # Compute similarities
        results = []
        for node_id, node_embedding in embeddings_dict.items():
            similarity = cosine_similarity(query_vec, node_embedding.reshape(1, -1))[0][0]
            if node_id in self.node_metadata:
                meta = self.node_metadata[node_id].copy()
                meta["similarity_score"] = float(similarity)
                results.append(meta)
        
        # Sort and return top_k
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

# ----------------------------
# 3. Hybrid Retriever
# ----------------------------
class FPLHybridRetriever:
    def __init__(self):
        self.baseline = FPLBaselineRetriever()
        self.embedding = FPLEmbeddingRetriever()
        self.embedding.embed_nodes()
    
    def retrieve(self, user_query, use_embeddings=True):
        baseline_results = self.baseline.retrieve(user_query)
        if not use_embeddings:
            return baseline_results
        embedding_results = self.embedding.embedding_search(user_query, top_k=5, model="both")
        return {"baseline": baseline_results, "semantic_search": embedding_results}
    
    def close(self):
        self.baseline.close()
        self.embedding.close()


if __name__ == "__main__":
    print("=" * 60)
    print("DEBUGGING EMBEDDINGS (BOTH MODELS)")
    print("=" * 60)
    
    # 1. Initialize
    embedding = FPLEmbeddingRetriever()
    
    # 2. Check/Generate Embeddings
    # Keep this False since you successfully generated them in the last run
    print("Checking for existing embeddings...")
    embedding.embed_nodes(force_regenerate=False)
    
    # 3. Test Queries
    test_queries = [
        "Mohamed Salah",
        "Haaland",
        "High scoring defenders under 5 million",
        "Premium Midfielders" 
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        
        # Request BOTH models
        # Note: If you haven't run the CREATE INDEX Cypher commands yet, 
        # this will print a warning and automatically fall back to manual calculation for both.
        results = embedding.embedding_search(query, top_k=3, model="both", use_neo4j_index=True)
        
        # --- Print Model 1 Results ---
        print(f"\n--- Model 1 (MiniLM - Faster) ---")
        m1_results = results.get('model1', [])
        if not m1_results:
            print("No results.")
        else:
            for i, p in enumerate(m1_results, 1):
                print(f" {i}. {p['name']} ({p['position']}) | Score: {p['similarity_score']:.4f}")
                print(f"    Stats: {p['total_points']} pts | £{p['avg_value']}m")

        # --- Print Model 2 Results ---
        print(f"\n--- Model 2 (MPNet - More Accurate) ---")
        m2_results = results.get('model2', [])
        if not m2_results:
            print("No results.")
        else:
            for i, p in enumerate(m2_results, 1):
                print(f" {i}. {p['name']} ({p['position']}) | Score: {p['similarity_score']:.4f}")
                print(f"    Stats: {p['total_points']} pts | £{p['avg_value']}m")

    embedding.close()