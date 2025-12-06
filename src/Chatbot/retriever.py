# retriever.py
import os
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Utility: load config
# ----------------------------
def load_config():
    config = {}
    # Absolute path to the kg folder where config.txt is
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, '..', 'kg', 'config.txt')  # move up one folder, then into kg
    config_path = os.path.abspath(config_path)
    print(f"Looking for config at: {config_path}")
    
    with open(config_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                config[key] = value
    return config


# ----------------------------
# 1. Baseline Retriever (Cypher queries)
# ----------------------------
class FPLBaselineRetriever:
    def __init__(self):
        config = load_config()
        self.driver = GraphDatabase.driver(config['URI'], auth=(config['USERNAME'], config['PASSWORD']))
        self.query_templates = self._init_queries()

    def close(self):
        self.driver.close()

    def _init_queries(self):
        return [
            # 1. Top players by position in a season
            lambda entities: f"""
                MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
                WHERE pos.name = '{entities.get("position","Midfielder")}' 
                AND p.season = '{entities.get("season","2023")}'
                RETURN p.player_name, p.total_points ORDER BY p.total_points DESC LIMIT 5
            """,
            # 2. Player stats in a season
            lambda entities: f"""
                MATCH (p:Player {{player_name:'{entities.get("player","Salah")}'}})
                RETURN p.player_name, p.total_points, p.goals_scored, p.assists
            """,
            # 3. Team analysis
            lambda entities: f"""
                MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
                WHERE p.team = '{entities.get("team","Liverpool")}'
                RETURN pos.name, avg(p.total_points) AS avg_points
            """,
            # 4. Player performance in a gameweek
            lambda entities: f"""
                MATCH (p:Player)-[r:PLAYED_IN]->(gw:Gameweek)
                WHERE p.player_name = '{entities.get("player","Salah")}' 
                AND gw.GW_number = {entities.get("gameweek",1)}
                RETURN p.player_name, r.total_points, r.goals_scored, r.assists
            """,
            # 5. Fixtures for a team
            lambda entities: f"""
                MATCH (f:Fixture)-[:HAS_HOME_TEAM|:HAS_AWAY_TEAM]->(t:Team)
                WHERE t.name = '{entities.get("team","Liverpool")}'
                RETURN f.fixture_number, f.kickoff_time
            """,
            # 6. Compare two players
            lambda entities: f"""
                MATCH (p:Player)
                WHERE p.player_name IN {entities.get("players", ["Salah","Haaland"])}
                RETURN p.player_name, p.total_points, p.goals_scored, p.assists
            """,
            # 7. Players under budget
            lambda entities: f"""
                MATCH (p:Player)
                WHERE p.value <= {entities.get("max_value", 10)}
                RETURN p.player_name, p.total_points ORDER BY p.total_points DESC
            """,
            # 8. Team clean sheet ranking
            lambda entities: f"""
                MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
                WHERE pos.name = 'Defender'
                RETURN p.team, sum(p.clean_sheets) AS total_clean_sheets
                ORDER BY total_clean_sheets DESC
            """,
            # 9. Captain recommendation
            lambda entities: f"""
                MATCH (p:Player)
                RETURN p.player_name, sum(p.total_points) AS total_points
                ORDER BY total_points DESC LIMIT 1
            """,
            # 10. Gameweek top scorers
            lambda entities: f"""
                MATCH (p:Player)-[r:PLAYED_IN]->(gw:Gameweek)
                WHERE gw.GW_number = {entities.get("gameweek",1)}
                RETURN p.player_name, r.total_points
                ORDER BY r.total_points DESC LIMIT 5
            """
        ]

    def retrieve(self, entities):
        results = []
        with self.driver.session() as session:
            for template in self.query_templates:
                query = template(entities)
                res = session.run(query)
                results.append([dict(r) for r in res])
        return results

# ----------------------------
# 2. Embedding-based Retriever
# ----------------------------
class FPLEmbeddingRetriever:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        config = load_config()
        self.driver = GraphDatabase.driver(config['URI'], auth=(config['USERNAME'], config['PASSWORD']))
        self.model = SentenceTransformer(embedding_model)
        self.node_embeddings = {}  # node_name -> vector

    def close(self):
        self.driver.close()

    def embed_nodes(self):
        with self.driver.session() as session:
            players = session.run("""
                MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                RETURN p.player_name AS name, collect(r.total_points) AS points,
                       collect(r.goals_scored) AS goals, collect(r.assists) AS assists
            """)
            for record in players:
                text_repr = f"Player: {record['name']}, Points: {record['points']}, Goals: {record['goals']}, Assists: {record['assists']}"
                self.node_embeddings[record['name']] = self.model.encode(text_repr)

    def embedding_search(self, query_text, top_k=5):
        query_vec = self.model.encode(query_text)
        results = []
        for node_name, node_vec in self.node_embeddings.items():
            sim = cosine_similarity([query_vec], [node_vec])[0][0]
            results.append((node_name, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Baseline
    baseline = FPLBaselineRetriever()
    entities = {"player": "Salah", "season": "2023", "gameweek": 5, "position": "Midfielder"}
    print("Baseline results:")
    print(baseline.retrieve(entities))
    baseline.close()

    # Embedding
    emb = FPLEmbeddingRetriever()
    emb.embed_nodes()
    print("Embedding search results:")
    print(emb.embedding_search("Who scored most points in GW5?"))
    emb.close()
