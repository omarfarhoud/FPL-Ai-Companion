import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- Lookup / Regex lists with synonyms & plurals ---
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
    def __init__(self, model_size="3B"):
        # Initialize LLM fallback
        if model_size == "1.5B":
            model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        else:
            model_id = "Qwen/Qwen2.5-3B-Instruct"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        print(f"Loading LLM {model_id} for fallback...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        print(f"✔ LLM loaded.")

    def extract(self, query):
        entities = {}
        query_norm = query.lower()

        # --- Regex / Lookup Extraction ---
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

        # --- If some entities are missing, use LLM to fill in missing ones ---
        if not all(k in entities for k in ["player", "team", "position", "metric", "season", "gameweek"]):
            llm_entities = self._llm_extract(query)
            for k, v in llm_entities.items():
                if k not in entities or not entities[k]:
                    entities[k] = v

        return entities

    def _llm_extract(self, query):
        system_prompt = """You are an entity extractor for a Fantasy Premier League (FPL) app.
        Identify and extract the following entities from the user's query:
        - player, team, position, metric, season, gameweek
        Output ONLY a valid JSON object. Do not add explanations. If missing, do not include the entity."""

        examples_text = """
        User: How many goals did Salah score?
        Assistant: {"player": ["Salah"], "metric": ["goals"]}

        User: Compare Haaland and Watkins.
        Assistant: {"player": ["Haaland", "Watkins"]}

        User: Top defenders for Arsenal next week?
        Assistant: {"position": ["Defender"], "team": ["Arsenal"]}

        User: Stats for Saka in 2022-23.
        Assistant: {"player": ["Saka"], "season": ["2022-23"], "metric": ["Stats"]}
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
            temperature=0.1
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        entities = self._clean_json(response)
        if "position" in entities:
            entities["position"] = [POSITION_MAP.get(p.lower(), p) for p in entities["position"]]
        if "metric" in entities:
            entities["metric"] = [METRIC_MAP.get(m.lower(), m) for m in entities["metric"]]

        return entities

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
    extractor = HybridEntityExtractor(model_size="3B")
    test_queries = [
        "Stats for Mount in 2022-23 season",
        "Top midfielders for Chelsea in GW12",
        "2022/23 season stats for Saka",
        "Best goalkeepers under 5m?"
    ]

    for q in test_queries:
        print(f"Query: {q}")
        print("Entities:", extractor.extract(q))
        print("-"*40)
