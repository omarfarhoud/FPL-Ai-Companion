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

# Custom CSS for better readability
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
    """Initialize the retriever once and cache it."""
    try:
        return FPLHybridRetriever()
    except Exception as e:
        st.error(f"Failed to initialize Retriever. Error: {e}")
        return None

@st.cache_resource
def get_llm_client():
    """Initialize the LLM Client once and cache it."""
    try:
        return LLMClient()
    except Exception as e:
        st.error(f"Failed to initialize LLM Client. Error: {e}")
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
        index=0,
        format_func=lambda x: x.upper()
    )

    st.subheader("Embeddings Settings")
    selected_embedding = st.selectbox(
        "Select Embedding Model",
        options=["model1", "model2"],
        index=1,
        format_func=lambda x: x.upper()
    )

    st.subheader("Retrieval Settings")
    use_semantic = st.toggle("Enable Semantic Search (Vector)", value=True)

    st.info("Ensure your Neo4j database is running.")

# -------------------------------------------------------------------
# Main Interface
# -------------------------------------------------------------------
st.title("Fantasy Premier League Graph-RAG")
st.markdown("Ask questions about player stats, fixtures, or get recommendations based on the Knowledge Graph.")

# Suggested Queries
st.caption("Try asking:")
col_ex1, col_ex2, col_ex3 = st.columns(3)
if col_ex1.button("Top scoring defenders under 5m"):
    st.session_state.query_input = "Top scoring defenders under 5m"
if col_ex2.button("Compare Haaland and Salah"):
    st.session_state.query_input = "Compare Haaland and Salah"
if col_ex3.button("Liverpool fixtures"):
    st.session_state.query_input = "Show me the next fixtures for Liverpool"

# Input Area
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

query = st.text_input("Enter your query:", value=st.session_state.query_input)
submit_btn = st.button("Analyze & Answer", type="primary")

# -------------------------------------------------------------------
# Execution Logic
# -------------------------------------------------------------------
if submit_btn and query:
    # Initialize retriever and LLM client
    retriever = get_retriever()
    llm_client = get_llm_client()

    if retriever and llm_client:
        status_container = st.empty()

        # --- STEP 1: RETRIEVAL ---
        with status_container.status("🔍 Retrieving context from Knowledge Graph...", expanded=True) as status:
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
            status.update(label="Context Retrieved!", state="complete", expanded=False)

        # --- STEP 2: DISPLAY KG CONTEXT ---
        st.divider()
        st.subheader("1. Retrieved Context (Transparency Layer)")
        st.markdown("Raw facts retrieved from the Knowledge Graph *before* LLM processing.")

        col1, col2 = st.columns(2)

        # Structured Database Results
        with col1:
            st.info("**Structured Data (Cypher Query)**")
            results_list = baseline_data.get("results", [])
            if results_list:
                for res in results_list:
                    data = res.get("data", [])
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, width='stretch', hide_index=True)
                    else:
                        st.write("Query executed but returned no data.")
            else:
                st.warning("No direct structured matches found.")

        # Semantic Vector Search Results
        with col2:
            st.success("**Vector Search (Semantic Similarity)**")
            combined_semantic = []
            for model_results in semantic_data.values():
                combined_semantic.extend(model_results)

            if combined_semantic:
                seen = set()
                unique_semantic = []
                for item in combined_semantic:
                    if item['name'] not in seen:
                        unique_semantic.append(item)
                        seen.add(item['name'])
                for item in unique_semantic[:5]:
                    with st.expander(f"**{item['name']}** ({item['position']}) - Score: {item.get('score', 0):.3f}"):
                        st.write(f"Points: {item.get('total_points')} | Price: £{item.get('avg_value')}m")
                        st.write(f"Form: {item.get('avg_form')}")
            else:
                st.markdown("*No semantic matches requested or found.*")

        # --- STEP 3: LLM GENERATION ---
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
            st.caption(f"---\n**Metrics:** ⏱️ {llm_response['response_time']:.2f}s | 🪙 Tokens: {llm_response['token_count']} | 🧠 Model: {selected_llm.upper()}")

            with st.expander("View Full Prompt Sent to LLM"):
                merged = llm_client.merger.merge_results(baseline_data, semantic_data)
                context_str = llm_client.prompt_builder.build_full_prompt(
                    user_query=query,
                    merged_context=merged
                )
                st.text(context_str)
        else:
            st.error(f"LLM Generation Failed: {llm_response.get('error')}")

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown("---")
st.markdown("Built with Streamlit, Neo4j, and HuggingFace Transformers.")
