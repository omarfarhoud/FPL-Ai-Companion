import streamlit as st
import time
import pandas as pd

# Import your custom modules
from retriever import FPLHybridRetriever
from llm_client import LLMClient

# -------------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="FPL Graph-RAG System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# Custom CSS
# -------------------------------------------------------------------
st.markdown("""
<style>
    .reportview-container { margin-top: -2em; }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Session State & Resource Caching
# -------------------------------------------------------------------
@st.cache_resource
def get_retriever():
    try:
        return FPLHybridRetriever()
    except Exception as e:
        st.error(f"Failed to initialize Retriever: {e}")
        return None


@st.cache_resource
def get_llm_client():
    try:
        return LLMClient()
    except Exception as e:
        st.error(f"Failed to initialize LLM Client: {e}")
        return None

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.title("⚽ System Config")

    st.subheader("LLM Settings")
    selected_llm = st.selectbox(
        "Select LLM",
        options=["qwen", "llama", "smollm"],
        format_func=lambda x: x.upper()
    )

    st.subheader("Embedding Settings")
    selected_embedding = st.selectbox(
        "Select Embedding Model",
        options=["model1", "model2"],
        index=1,
        format_func=lambda x: x.upper()
    )

    st.subheader("Retrieval Settings")
    use_semantic = st.toggle("Enable Semantic Search (Vector)", value=True)

    st.info("Ensure Neo4j is running.")

# -------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------
st.title("Fantasy Premier League Graph-RAG")
st.markdown(
    "Ask questions about player stats, fixtures, or recommendations "
    "using a transparent Graph-RAG pipeline."
)

# Suggested Queries
st.caption("Try asking:")
c1, c2, c3 = st.columns(3)
if c1.button("Top scoring defenders under 5m"):
    st.session_state.query_input = "Top scoring defenders under 5m"
if c2.button("Compare Haaland and Salah"):
    st.session_state.query_input = "Compare Haaland and Salah"
if c3.button("Liverpool fixtures"):
    st.session_state.query_input = "Show me the next fixtures for Liverpool"

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

query = st.text_input("Enter your query:", value=st.session_state.query_input)
submit_btn = st.button("Analyze & Answer", type="primary")

# -------------------------------------------------------------------
# Execution Logic
# -------------------------------------------------------------------
if submit_btn and query:
    retriever = get_retriever()
    llm_client = get_llm_client()

    if retriever and llm_client:
        status_container = st.empty()

        # ------------------ STEP 1: RETRIEVAL ------------------
        with status_container.status(
            "🔍 Retrieving context from Knowledge Graph...",
            expanded=True
        ) as status:

            start_time = time.time()

            retrieval_results = retriever.retrieve(
                query,
                use_embeddings=use_semantic,
                model_choice=selected_embedding
            )

            baseline_data = retrieval_results.get("baseline", {})
            semantic_data = retrieval_results.get("semantic_search", {})

            intent = baseline_data.get("intent", "Unknown")
            entities = baseline_data.get("entities", {})

            status.write(f"**Intent Detected:** `{intent}`")
            status.write(f"**Entities Extracted:** `{entities}`")

            status.update(
                label="Context Retrieved",
                state="complete",
                expanded=False
            )

        # ------------------ STEP 2: CONTEXT DISPLAY ------------------
        st.divider()
        st.subheader("1. Retrieved Context (Transparency Layer)")
        st.markdown(
            "Raw information retrieved **before** LLM reasoning."
        )

        col1, col2 = st.columns(2)

        # -------- BASELINE (Cypher) --------
        with col1:
            st.info("**Structured Data (Cypher Results)**")

            results_list = baseline_data.get("results", [])

            if results_list:
                for res in results_list:
                    data = res.get("data", [])
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, width="stretch", hide_index=True)
                    else:
                        st.write("Query executed but returned no rows.")
            else:
                st.warning("No baseline matches found.")

        # -------- EMBEDDINGS (FULL OUTPUT) --------
        with col2:
            st.success("**Vector Search Results (FULL OUTPUT)**")

            if semantic_data:
                for model_name, results in semantic_data.items():
                    st.markdown(f"### 🔹 Embedding Model: `{model_name}`")

                    if not results:
                        st.write("No results.")
                        continue

                    for i, item in enumerate(results, start=1):
                        with st.expander(
                            f"{i}. {item.get('name', 'Unknown')} "
                            f"({item.get('position', 'N/A')}) "
                            f"| Score: {item.get('score', 0):.4f}"
                        ):
                            st.json(item)
            else:
                st.markdown("*Semantic search disabled or no results returned.*")

        # ------------------ STEP 3: LLM ------------------
        st.divider()
        st.subheader("2. Final Answer")

        with st.spinner(f"Generating answer using {selected_llm.upper()}..."):
            llm_response = llm_client.generate_response(
                user_query=query,
                baseline_results=baseline_data,
                embedding_results=semantic_data,
                model_name=selected_llm
            )

        if llm_response.get("success"):
            st.markdown(f"### 🤖 {llm_response['answer']}")

            st.caption(
                f"⏱️ {llm_response['response_time']:.2f}s | "
                f"🪙 Tokens: {llm_response['token_count']} | "
                f"🧠 Model: {selected_llm.upper()}"
            )

            with st.expander("View Full Prompt Sent to LLM"):
                merged = llm_client.merger.merge_results(
                    baseline_data,
                    semantic_data
                )
                full_prompt = llm_client.prompt_builder.build_full_prompt(
                    user_query=query,
                    merged_context=merged
                )
                st.text(full_prompt)
        else:
            st.error(f"LLM failed: {llm_response.get('error')}")

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown("---")
st.markdown("Built with Streamlit, Neo4j, and HuggingFace Transformers.")
