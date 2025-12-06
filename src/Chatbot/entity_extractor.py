import re
import json
import torch
import platform
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Lookup / Regex lists with examples ---
PLAYERS = ["Salah", "Haaland", "Son", "Watkins", "Saka", "Foden", "Pickford", "Raya", "Fernandes", "Kane"]
TEAMS = ["Liverpool", "Arsenal", "Man City", "Tottenham", "Chelsea", "Newcastle", "Brighton", "Man United"]
POSITIONS = ["GK", "Goalkeeper", "GKP", "Defender", "Defenders", "DEF",
             "Midfielder", "Midfielders", "MID", "Forward", "Forwards", "FWD"]
METRICS = ["goals", "assists", "points", "bonus points", "clean sheets", "ICT index", "minutes played", "fixtures", "form"]
GAMEWEEKS = [f"GW{i}" for i in range(1, 39)]
SEASON_PATTERN = r"\b(20\d{2})[-/](\d{2})\b"

# --- Normalization maps ---
POSITION_MAP = {
    "defender": "Defender",
    "defenders": "Defender",
    "midfielder": "Midfielder",
    "midfielders": "Midfielder",
    "forward": "Forward",
    "forwards": "Forward",
    "goalkeeper": "GK",
    "gkp": "GK",
    "gk": "GK",
    "def": "Defender",
    "mid": "Midfielder",
    "fwd": "Forward"
}

METRIC_MAP = {
    "bonus points": "bonus points",
    "points": "points",
    "goals": "goals",
    "assists": "assists",
    "clean sheets": "clean sheets",
    "ict index": "ICT index",
    "minutes played": "minutes played",
    "fixtures": "team",
    "form": "form"
}


class HybridEntityExtractor:
    def __init__(self, model_size="1.5B"):
        # ----------------------------
        # Load lightweight LLM for fallback
        # ----------------------------
        model_id = f"Qwen/Qwen2.5-{model_size}-Instruct"
        print(f"Loading LLM {model_id} for fallback...")

        is_windows = platform.system() == "Windows"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=None,  # no auto device mapping
            torch_dtype=torch.float16 if not is_windows else None,
            local_files_only=False
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"✔ LLM loaded on {self.device}.")

    # ----------------------------
    # Main entity extraction
    # ----------------------------
    def extract(self, query):
        entities = {}
        query_norm = query.lower()

        # --- Regex / Lookup extraction ---
        found_players = [p for p in PLAYERS if re.search(rf"\b{p}\b", query, re.I)]
        if found_players:
            entities["player"] = found_players

        found_teams = [t for t in TEAMS if re.search(rf"\b{t}\b", query, re.I)]
        if found_teams:
            entities["team"] = found_teams

        found_positions = [pos for pos in POSITIONS if re.search(rf"\b{pos}\b", query, re.I)]
        if found_positions:
            entities["position"] = [POSITION_MAP.get(pos.lower(), pos) for pos in found_positions]

        found_metrics = [m for m in METRICS if re.search(rf"\b{m}\b", query, re.I)]
        if found_metrics:
            entities["metric"] = [METRIC_MAP.get(m.lower(), m) for m in found_metrics]

        found_gw = [gw for gw in GAMEWEEKS if re.search(rf"\b{gw}\b", query, re.I)]
        if found_gw:
            entities["gameweek"] = found_gw

        found_seasons = re.findall(SEASON_PATTERN, query)
        if found_seasons:
            entities["season"] = [f"{y1}-{y2}" for y1, y2 in found_seasons]

        # --- LLM fallback for missing entities ---
        required_keys = ["player", "team", "position", "metric", "season", "gameweek"]
        if not all(k in entities for k in required_keys):
            llm_entities = self._llm_extract(query)
            for k, v in llm_entities.items():
                if k not in entities or not entities[k]:
                    entities[k] = v

        return entities

    # ----------------------------
    # LLM-based extraction fallback
    # ----------------------------
    def _llm_extract(self, query):
        system_prompt = """You are an entity extractor for Fantasy Premier League (FPL).

Extract the following entities from the user's query:
- player: any player mentioned
- team: any team mentioned
- position: goalkeepers, defenders, midfielders, forwards, etc.
- metric: stats like goals, assists, points, bonus points, clean sheets, ICT index, minutes played, form, fixtures
- season: format YYYY-YY (e.g., 2022-23)
- gameweek: format GW followed by a number (e.g., GW12)

Rules:
1. Only output entities that appear in the query.
2. Output exactly a JSON object. Do NOT invent players, teams, or metrics.
3. If an entity is not mentioned, omit it.
4. Do NOT include explanations or extra text."""

        examples_text = """
User: How many goals did Salah score?
Assistant: {"player": ["Salah"], "metric": ["goals"]}

User: Compare Haaland and Watkins.
Assistant: {"player": ["Haaland", "Watkins"]}

User: Top defenders for Arsenal next week?
Assistant: {"position": ["Defender"], "team": ["Arsenal"]}

User: Stats for Saka in 2022-23.
Assistant: {"player": ["Saka"], "season": ["2022-23"], "metric": ["stats"]}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": examples_text + f"\nUser: {query}"}
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False  # deterministic output for JSON
        )

        generated_only = generated_ids[0, inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_only, skip_special_tokens=True).strip()
        entities = self._clean_json(response)

        # normalize positions & metrics
        if "position" in entities:
            entities["position"] = [POSITION_MAP.get(p.lower(), p) for p in entities["position"]]
        if "metric" in entities:
            entities["metric"] = [METRIC_MAP.get(m.lower(), m) for m in entities["metric"]]

        return entities

    # ----------------------------
    # JSON cleanup helper
    # ----------------------------
    def _clean_json(self, raw_response):
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            return {}


# --- Test ---
if __name__ == "__main__":
    extractor = HybridEntityExtractor(model_size="1.5B")
    test_queries = [
        "Stats for Mount in 2022-23 season",
        "Top midfielders for Chelsea in GW12",
        "2022/23 season stats for Saka",
        "Best goalkeepers under 5m?",
        "Who scored most goals for Liverpool?",
        "Clean sheets for Man City defenders?"
    ]

    for q in test_queries:
        print(f"Query: {q}")
        print("Entities:", extractor.extract(q))
        print("-"*40)
