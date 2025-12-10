import streamlit as st
import time
import pandas as pd
import os
import json

# Import your custom modules
from retriever import FPLHybridRetriever
from llm_client import LLMClient

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FPL Graph-RAG System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better readability
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State & Resource Caching
# -----------------------------------------------------------------------------

@st.cache_resource
def get_retriever():
    """Initialize the retriever once and cache it."""
    try:
        return FPLHybridRetriever()
    except Exception as e:
        st.error(f"Failed to initialize Retriever. Check Neo4j connection/Config. Error: {e}")
        return None

@st.cache_resource
def get_llm_client():
    """Initialize the LLM Client once and cache it."""
    try:
        # Authentication is handled via env vars or local cache
        return LLMClient() 
    except Exception as e:
        st.error(f"Failed to initialize LLM Client. Error: {e}")
        return None

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚽ System Config")
    
    st.subheader("LLM Settings")
    
    selected_model = st.selectbox(
        "Select Model",
        options=["llama", "qwen", "smollm"],
        index=0,
        format_func=lambda x: x.upper()
    )
    
    st.subheader("Retrieval Settings")
    use_semantic = st.toggle("Enable Semantic Search (Vector)", value=True)
    
    st.info("Ensure your Neo4j database is running.")

# -----------------------------------------------------------------------------
# Main Interface
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Execution Logic
# -----------------------------------------------------------------------------
if submit_btn and query:
    # 1. Initialization
    retriever = get_retriever()
    llm_client = get_llm_client()

    if retriever and llm_client:
        status_container = st.empty()
        
        # --- STEP 1: RETRIEVAL ---
        with status_container.status("🔍 Retrieving context from Knowledge Graph...", expanded=True) as status:
            start_time = time.time()
            
            # Perform Retrieval
            retrieval_results = retriever.retrieve(query, use_embeddings=use_semantic)
            
            baseline_data = retrieval_results.get("baseline", {})
            semantic_data = retrieval_results.get("semantic_search", {})
            
            intent = baseline_data.get("intent", "Unknown")
            entities = baseline_data.get("entities", {})
            
            status.write(f"**Intent Detected:** `{intent}`")
            status.write(f"**Entities Extracted:** `{entities}`")
            status.update(label="Context Retrieved!", state="complete", expanded=False)

        # --- STEP 2: DISPLAY KG CONTEXT (Requirement A) ---
        st.divider()
        st.subheader("1. Retrieved Context (Transparency Layer)")
        st.markdown("These are the raw facts retrieved from the Knowledge Graph *before* LLM processing.")

        col1, col2 = st.columns(2)

        # Left Column: Structured Database Results (Cypher)
        with col1:
            st.info(f"**Structured Data (Cypher Query)**")
            results_list = baseline_data.get("results", [])
            
            if results_list:
                for res in results_list:
                    data = res.get("data", [])
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.write("Query executed but returned no data.")
            else:
                st.warning("No direct structured matches found.")

        # Right Column: Semantic Vector Search Results
        with col2:
            st.success(f"**Vector Search (Semantic Similarity)**")
            
            # Combine model results for display
            combined_semantic = []
            if "model1" in semantic_data:
                combined_semantic.extend(semantic_data["model1"])
            if "model2" in semantic_data:
                combined_semantic.extend(semantic_data["model2"])
            
            if combined_semantic:
                # Deduplicate for display
                seen = set()
                unique_semantic = []
                for item in combined_semantic:
                    if item['name'] not in seen:
                        unique_semantic.append(item)
                        seen.add(item['name'])
                
                # Create a nice display for embeddings
                for item in unique_semantic[:5]:
                    with st.expander(f"**{item['name']}** ({item['position']}) - Score: {item.get('similarity_score', 0):.3f}"):
                        st.write(f"Points: {item.get('total_points')} | Price: £{item.get('avg_value')}m")
                        st.write(f"Form: {item.get('avg_form')}")
            else:
                st.markdown("*No semantic matches requested or found.*")

        # --- STEP 3: LLM GENERATION (Requirement B) ---
        st.divider()
        st.subheader("2. Final Answer")
        
        with st.spinner(f"Generating answer using {selected_model.upper()}..."):
            llm_response = llm_client.generate_response(
                user_query=query,
                baseline_results=baseline_data,
                embedding_results=semantic_data,
                model_name=selected_model
            )
        
        if llm_response.get("success"):
            # Display Answer
            st.markdown(f"### 🤖 {llm_response['answer']}")
            
            # Metrics Footer
            st.caption(f"""
            ---
            **Metrics:** ⏱️ Response Time: {llm_response['response_time']:.2f}s | 
            🪙 Tokens Generated: {llm_response['token_count']} | 
            🧠 Model: {selected_model.upper()}
            """)
            
            # Debug Data (Optional)
            with st.expander("View Full Prompt Sent to LLM"):
                # Reconstruct prompt to show user
                merged = llm_client.merger.merge_results(baseline_data, semantic_data)
                context_str = llm_client.prompt_builder.build_full_prompt(    user_query=query,
    merged_context=merged)
                st.text(context_str)
        else:
            st.error(f"LLM Generation Failed: {llm_response.get('error')}")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("Built with Streamlit, Neo4j, and HuggingFace Transformers.")