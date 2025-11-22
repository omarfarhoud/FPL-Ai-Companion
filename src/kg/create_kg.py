from neo4j import GraphDatabase
import pandas as pd
import os

# 1. Load Configuration
def load_config():
        config = {}
        # Get the absolute path to the folder containing this script
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Join it with the filename
        config_path = os.path.join(base_path, 'config.txt')
        
        print(f"Looking for config at: {config_path}") # Debug print
        
        with open(config_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    config[key] = value
        return config

class FPLGraphBuilder:
    def __init__(self):
        config = load_config()
        self.driver = GraphDatabase.driver(
            config['URI'], 
            auth=(config['USERNAME'], config['PASSWORD'])
        )

    def close(self):
        self.driver.close()

    def setup_constraints(self):
        """
        Sets up unique constraints strictly based on the Schema.
        """
        queries = [
            # Season: [season_name]
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Season) REQUIRE s.season_name IS UNIQUE",
            # Team: [name]
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Team) REQUIRE t.name IS UNIQUE",
            # Position: [name]
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Position) REQUIRE p.name IS UNIQUE",
            # Gameweek: [season, GW_number]
            "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gameweek) REQUIRE (g.season, g.GW_number) IS UNIQUE",
            # Fixture: [season, fixture_number]
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Fixture) REQUIRE (f.season, f.fixture_number) IS UNIQUE",
            # Player: [player_name, player_element]
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Player) REQUIRE (p.player_name, p.player_element) IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for q in queries:
                session.run(q)
            print("Constraints and Indexes created.")

    def load_data(self, csv_path):
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Convert DataFrame to list of dicts
        data = df.to_dict('records')
        
        # Batch size for efficient loading
        batch_size = 1000
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

        with self.driver.session() as session:
            total_batches = len(batches)
            for i, batch in enumerate(batches):
                print(f"Processing batch {i+1}/{total_batches}...")
                session.run(self.get_import_query(), rows=batch)

    def get_import_query(self):
        """
        Cypher query mapping CSV columns to the Schema.
        """
        return """
        UNWIND $rows AS row
        
        // 1. Merge Basic Nodes
        MERGE (season:Season {season_name: row.season})
        
        // Note: Team nodes are created via Home/Away relationships below 
        // to ensure we catch all teams appearing in fixtures.
        
        MERGE (pos:Position {name: row.position})
        
        MERGE (player:Player {
            player_name: row.name, 
            player_element: toInteger(row.element)
        })
        
        // 2. Merge Gameweek [season, GW_number]
        MERGE (gw:Gameweek {
            season: row.season, 
            GW_number: toInteger(row.GW)
        })
        
        // 3. Merge Fixture [season, fixture_number]
        MERGE (fix:Fixture {
            season: row.season, 
            fixture_number: toInteger(row.fixture)
        })
        ON CREATE SET fix.kickoff_time = row.kickoff_time
        
        // 4. Relationships
        
        // (Season) -[:HAS_GW]-> (Gameweek)
        MERGE (season)-[:HAS_GW]->(gw)
        
        // (Gameweek) -[:HAS_FIXTURE]-> (Fixture)
        MERGE (gw)-[:HAS_FIXTURE]->(fix)
        
        // (Player) -[:PLAYS_AS]-> (Position)
        MERGE (player)-[:PLAYS_AS]->(pos)
        
        // (Player) -[:PLAYED_IN]-> (Fixture) with properties
        MERGE (player)-[r:PLAYED_IN]->(fix)
        SET r.minutes = toInteger(row.minutes),
            r.goals_scored = toInteger(row.goals_scored),
            r.assists = toInteger(row.assists),
            r.total_points = toInteger(row.total_points),
            r.bonus = toInteger(row.bonus),
            r.clean_sheets = toInteger(row.clean_sheets),
            r.goals_conceded = toInteger(row.goals_conceded),
            r.own_goals = toInteger(row.own_goals),
            r.penalties_saved = toInteger(row.penalties_saved),
            r.penalties_missed = toInteger(row.penalties_missed),
            r.yellow_cards = toInteger(row.yellow_cards),
            r.red_cards = toInteger(row.red_cards),
            r.saves = toInteger(row.saves),
            r.bps = toInteger(row.bps),
            r.influence = toFloat(row.influence),
            r.creativity = toFloat(row.creativity),
            r.threat = toFloat(row.threat),
            r.ict_index = toFloat(row.ict_index),
            r.form = toFloat(row.form),
            
            // Extra properties from CSV often useful
            r.selected = toInteger(row.selected),
            r.transfers_in = toInteger(row.transfers_in),
            r.transfers_out = toInteger(row.transfers_out),
            r.value = toInteger(row.value)
        
        // 5. Fixture Teams (Home and Away)
        WITH row, fix
        MERGE (homeT:Team {name: row.home_team})
        MERGE (awayT:Team {name: row.away_team})
        
        MERGE (fix)-[:HAS_HOME_TEAM]->(homeT)
        MERGE (fix)-[:HAS_AWAY_TEAM]->(awayT)
        """

if __name__ == "__main__":
    builder = FPLGraphBuilder()
    try:
        builder.setup_constraints()
        # Use the exact filename you provided
        csv_file = "C:/Users/omarf/FPL-Ai-Companion/src/kg/fpl_two_seasons.csv" 
        if os.path.exists(csv_file):
            builder.load_data(csv_file)
            print("Knowledge Graph construction complete.")
        else:
            print(f"Error: File {csv_file} not found.")
    finally:
        builder.close()