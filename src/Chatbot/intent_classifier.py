import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class IntentClassifier:
    def __init__(self, model_size="3B"):
        # Model selection
        if model_size == "1.5B":
            model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        else:
            model_id = "Qwen/Qwen2.5-3B-Instruct"
            
        print(f"Loading {model_id}...")

        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        print(f"✔ {model_id} loaded successfully.")

        # Valid intents
        self.valid_intents = [
            "get_player_stats",
            "compare_players",
            "get_recommendation",
            "team_analysis",
            "fixture_query",
            "unknown"
        ]

        # System prompt
        self.system_prompt = f"""You are a helpful assistant for Fantasy Premier League (FPL).
        Classify the user's query into EXACTLY one of these intents: {self.valid_intents}.
        Output ONLY the intent name. Do not output anything else."""

        # Examples: 3–4 per intent
        self.examples_text = """
        User: How many points did Haaland get last week?
        Assistant: get_player_stats
        User: How many goals does Salah have?
        Assistant: get_player_stats
        User: Show me Watkins’ recent stats.
        Assistant: get_player_stats
        User: Should I start Pickford or Raya?
        Assistant: compare_players
        User: Who is better, Saka or Foden?
        Assistant: compare_players
        User: Pick one: Alvarez or Solanke.
        Assistant: compare_players
        User: Suggest a defender under 4.5m.
        Assistant: get_recommendation
        User: Who is the best captain for GW12?
        Assistant: get_recommendation
        User: I need a replacement for Trent.
        Assistant: get_recommendation
        User: How are Arsenal performing recently?
        Assistant: team_analysis
        User: Which team has the most clean sheets?
        Assistant: team_analysis
        User: Are Liverpool strong defensively right now?
        Assistant: team_analysis
        User: Who does Liverpool play next?
        Assistant: fixture_query
        User: Does Man City have a double gameweek?
        Assistant: fixture_query
        User: What are the easiest fixtures coming up?
        Assistant: fixture_query
        User: What is the capital of France?
        Assistant: unknown
        User: Translate this to Spanish.
        Assistant: unknown
        User: Explain quantum computing.
        Assistant: unknown
        """

        # Pre-apply chat template once
        self.static_input_ids = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.examples_text}
            ],
            tokenize=True,   # now we pre-tokenize
            add_generation_prompt=True
        )

    def predict(self, user_query):
        # Append the user query only
        text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.examples_text + f"\nUser: {user_query}"}
            ],
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=10,
            temperature=0.1
        )

        # Decode
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response.strip()
        # --- Test Block ---
if __name__ == "__main__":
    classifier = IntentClassifier(model_size="3B") 

    test_queries = [
        # get_player_stats
        "How many assists did Son have last gameweek?",
        "Show me Kane's total points this season.",
        "What is Fernandes' current form like?",

        # compare_players
        "Who should I pick: De Bruyne or Mount?",
        "Is Antonio better than Jimenez for this week?",
        "Between Cancelo and Alexander-Arnold, who is safer?",

        # get_recommendation
        "I need a cheap midfielder under 5.5m.",
        "Who should I captain this week?",
        "Suggest a good budget forward for my team.",

        # team_analysis
        "How is Chelsea performing after their last three matches?",
        "Which team has the best defensive record currently?",
        "Are Tottenham improving this season?",

        # fixture_query
        "Who does Brighton face next GW?",
        "Does Liverpool have a double gameweek soon?",
        "Which teams have easy fixtures in the next two GWs?",

        # unknown
        "What's the weather like in London?",
        "How do I cook pasta?",
        "Who won the World Cup in 1998?"
    ]

    print("\n--- Testing Qwen2.5 ---")
    for q in test_queries:
        print(f"Query: '{q}'\nIntent: {classifier.predict(q)}\n")