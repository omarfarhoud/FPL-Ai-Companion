import torch
import platform
from transformers import AutoTokenizer, AutoModelForCausalLM

class IntentClassifier:
    def __init__(self):
        # ----------------------------
        # Model selection: Qwen2.5-1.5B
        # ----------------------------
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"Loading {model_id}...")

        is_windows = platform.system() == "Windows"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=None,
            torch_dtype=torch.float16 if not is_windows else None,
            local_files_only=False
        )

        # Move to device
        self.device = torch.device("cpu")
        self.model.to(self.device)
        print(f"✔ {model_id} loaded on {self.device}.")

        # ----------------------------
        # Intents
        # ----------------------------
        self.valid_intents = [
            "get_player_stats",
            "compare_players",
            "get_recommendation",
            "get_leaderboard",
            "team_analysis",
            "fixture_query",
            "unknown"
        ]

        # ----------------------------
        # Optimized system prompt
        # ----------------------------
        self.system_prompt = f"""
You are a helpful Fantasy Premier League (FPL) assistant. Classify the user's query into **exactly one** of these intents:

{self.valid_intents}

Instructions:
- Output **ONLY** the intent name exactly as listed above.
- Never invent new intent names.
- If the query does not match any of the above intents, always output 'unknown'.
- Match the format of the examples exactly.
"""

        # ----------------------------
        # Expanded few-shot examples
        # ----------------------------
        self.examples_text = """
User: How many points did Haaland get last week?
Assistant: get_player_stats
User: How many goals does Salah have?
Assistant: get_player_stats
User: Show me Watkins' recent stats.
Assistant: get_player_stats
User: What is Fernandes' current form like?
Assistant: get_player_stats
User: Show me Mohamed Salah's stats
Assistant: get_player_stats
User: Tell me about Harry Kane
Assistant: get_player_stats
User: What are Haaland's numbers?
Assistant: get_player_stats
User: Stats for Saka
Assistant: get_player_stats

User: Should I start Pickford or Raya?
Assistant: compare_players
User: Who is better, Saka or Foden?
Assistant: compare_players
User: Pick one: Alvarez or Solanke.
Assistant: compare_players
User: Is Antonio better than Jimenez for this week?
Assistant: compare_players
User: Compare Haaland and Kane's performance
Assistant: compare_players
User: Compare Son vs Salah this season
Assistant: compare_players

User: Suggest a defender under 4.5m.
Assistant: get_recommendation
User: Who is the best captain for GW12?
Assistant: get_recommendation
User: I need a replacement for Trent.
Assistant: get_recommendation
User: Recommend a midfielder under 5.0m
Assistant: get_recommendation
User: Suggest a cheap forward for my team
Assistant: get_recommendation
User: Find me forwards in good form under 8 million
Assistant: get_recommendation
User: Show me budget defenders
Assistant: get_recommendation
User: Give me a good goalkeeper under 5 million
Assistant: get_recommendation
User: Find a goalkeeper under 6m
Assistant: get_recommendation
User: Show me cheap goalkeepers
Assistant: get_recommendation
User: Recommend a budget keeper
Assistant: get_recommendation
User: Give me a goalkeeper with at least 5 cleansheets
Assistant: get_recommendation
User: Find me a keeper with good clean sheet record
Assistant: get_recommendation
User: Show me goalkeepers with 10+ cleansheets
Assistant: get_recommendation

User: Who are the top 3 goal scorers?
Assistant: get_leaderboard
User: Show me the best point scorers this season
Assistant: get_leaderboard
User: Which players have the most assists?
Assistant: get_leaderboard
User: Who are the highest scoring midfielders?
Assistant: get_leaderboard
User: Top 5 defenders by points
Assistant: get_leaderboard

User: How are Arsenal performing recently?
Assistant: team_analysis
User: Which team has the most clean sheets?
Assistant: team_analysis
User: Are Liverpool strong defensively right now?
Assistant: team_analysis
User: How is Chelsea performing after their last three matches?
Assistant: team_analysis
User: Are Tottenham improving this season?
Assistant: team_analysis

User: Who does Liverpool play next?
Assistant: fixture_query
User: Does Man City have a double gameweek?
Assistant: fixture_query
User: What are the easiest fixtures coming up?
Assistant: fixture_query
User: Who does Brighton face next GW?
Assistant: fixture_query
User: Which teams have easy fixtures in the next two GWs?
Assistant: fixture_query

User: What is the capital of France?
Assistant: unknown
User: Translate this to Spanish.
Assistant: unknown
User: Explain quantum computing.
Assistant: unknown
User: What's the weather like in London?
Assistant: unknown
User: How do I cook pasta?
Assistant: unknown
User: Who won the 2018 World Cup?
Assistant: unknown
User: Tell me a joke
Assistant: unknown
User: What is the population of Japan?
Assistant: unknown
"""

        # ----------------------------
        # Pre-tokenize base prompt once
        # ----------------------------
        base_prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.examples_text}
            ],
            tokenize=False,
            add_generation_prompt=True
        )

        self.base_ids = self.tokenizer(
            base_prompt_text,
            return_tensors="pt",
            padding=True
        ).input_ids.to(self.device)

    # ----------------------------
    # Optimized prediction
    # ----------------------------
    def predict(self, user_query):
        # Rule-based fallback for common patterns
        query_lower = user_query.lower()
        
        # Comparison patterns (check FIRST)
        if 'compare' in query_lower or ' vs ' in query_lower or ' versus ' in query_lower:
            return "compare_players"
        
        # Team analysis patterns (check BEFORE player stats to avoid confusion)
        team_names = ['liverpool', 'arsenal', 'chelsea', 'manchester', 'city', 'united', 
                      'tottenham', 'spurs', 'brighton', 'newcastle', 'villa', 'west ham']
        if any(team in query_lower for team in team_names) or any(keyword in query_lower for keyword in ['team', 'teams', 'club', 'clubs', 'side']):
            if any(pattern in query_lower for pattern in [
                'how is', 'how are', 'doing', 'performing', 'performance', 'form', 
                'clean sheet', 'cleansheet', 'defensive', 'strong', 'weak', 'improving', 'struggling',
                'how many'
            ]):
                return "team_analysis"
        
        # Player stats patterns
        if any(pattern in query_lower for pattern in [
            'stats', 'statistics', 'numbers', 'performance', 'how is',
            'how has', 'how did', 'tell me about', 'show me about'
        ]):
            # Make sure it's not a team query or comparison
            if 'compare' not in query_lower and ' vs ' not in query_lower:
                if not any(keyword in query_lower for keyword in ['team', 'teams', 'which team']):
                    return "get_player_stats"
        
        # Fixture patterns (check BEFORE recommendations to avoid false positives)
        if any(keyword in query_lower for keyword in ['fixture', 'fixtures', 'match', 'matches', 'game', 'games']):
            return "fixture_query"
        if any(pattern in query_lower for pattern in ['play next', 'face next', 'upcoming', 'next gw', 'double gameweek']):
            return "fixture_query"
        
        # Captain queries (specific pattern for captain recommendations)
        if 'captain' in query_lower and any(word in query_lower for word in ['who', 'should', 'which', 'best']):
            return "get_recommendation"
        
        # Leaderboard patterns (for PLAYERS) - check BEFORE recommendations
        if any(pattern in query_lower for pattern in [
            'top', 'best', 'most', 'highest', 'leading', 'leaderboard',
            'who scored', 'who has', 'which player', 'show me', 'give me'
        ]) and any(stat in query_lower for stat in [
            'goal', 'assist', 'point', 'scorer', 'score', 'cleansheet', 'clean sheet'
        ]):
            # Make sure it's not about teams
            if not any(keyword in query_lower for keyword in ['team', 'teams', 'club', 'clubs']):
                # Check if it includes a position (defender, midfielder, etc.)
                if any(pos in query_lower for pos in ['defender', 'midfielder', 'forward', 'goalkeeper', 'keeper']):
                    return "get_leaderboard"
                # Or if it has leaderboard indicators
                if any(pattern in query_lower for pattern in ['top', 'best', 'most', 'highest', 'leading', 'leaderboard']):
                    return "get_leaderboard"
        
        # Recommendation patterns (find/suggest/recommend + position/budget)
        if any(keyword in query_lower for keyword in ['find', 'suggest', 'recommend', 'show me', 'need', 'give me', 'get me']):
            if any(indicator in query_lower for indicator in [
                'forward', 'midfielder', 'defender', 'goalkeeper', 'keeper', 'under', 'budget',
                'cheap', 'value', 'replacement', 'captain'
            ]):
                return "get_recommendation"
        
        # Continue with LLM-based classification
        user_line = f"User: {user_query}\nAssistant:"

        new_ids = self.tokenizer(
            user_line,
            return_tensors="pt",
            padding=True
        ).input_ids.to(self.device)

        model_inputs = torch.cat([self.base_ids, new_ids], dim=1)

        outputs = self.model.generate(
            input_ids=model_inputs,
            attention_mask=torch.ones_like(model_inputs),
            max_new_tokens=5,
            do_sample=False  # greedy decoding
        )

        generated_only = outputs[0, model_inputs.shape[1]:]

        raw_output = self.tokenizer.decode(
            generated_only,
            skip_special_tokens=True
        ).strip()

        # Map to valid intents (fallback to 'unknown')
        intent = next((i for i in self.valid_intents if i in raw_output), "unknown")
        return intent


# ----------------------------
# TEST BLOCK
# ----------------------------
if __name__ == "__main__":
    classifier = IntentClassifier()

    test_queries = [
        "How many assists did Son have last gameweek?",
        "Show me Kane's total points this season.",
        "What is Fernandes' current form like?",
        "Who should I pick: De Bruyne or Mount?",
        "Is Antonio better than Jimenez for this week?",
        "Between Cancelo and Alexander-Arnold, who is safer?",
        "I need a cheap midfielder under 5.5m.",
        "Who should I captain this week?",
        "Suggest a good budget forward for my team.",
        "How is Chelsea performing after their last three matches?",
        "Which team has the best defensive record currently?",
        "Are Tottenham improving this season?",
        "Who does Brighton face next GW?",
        "Does Liverpool have a double gameweek soon?",
        "Which teams have easy fixtures in the next two GWs?",
        "What's the weather like in London?",
        "How do I cook pasta?",
        "Who won the World Cup in 1998?"
    ]

    print("\n--- Testing Qwen2.5-1.5B with fully optimized prompt ---")
    for q in test_queries:
        intent = classifier.predict(q)
        print(f"Query: '{q}'\nIntent: {intent}\n")
