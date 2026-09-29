import os
import streamlit as st
import requests

from src.config import PINECONE_INDEX_NAME, OPENAI_API_KEY, PINECONE_API_KEY, HOST, PORT
from src.graph import build_rag_graph

st.set_page_config(
    page_title="Agentic AI RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agentic AI RAG Assistant")
st.caption("Strictly Grounded Chatbot on the Agentic AI eBook using LangGraph & Pinecone")

# Sidebar Configuration and Status
with st.sidebar:
    st.header("⚙️ System Status")
    st.markdown(f"**Index Name:** `{PINECONE_INDEX_NAME}`")
    st.markdown(f"**OpenAI API Key:** {'✅ Configured' if OPENAI_API_KEY else '❌ Missing'}")
    st.markdown(f"**Pinecone API Key:** {'✅ Configured' if PINECONE_API_KEY else '❌ Missing'}")

    st.divider()
    st.subheader("💡 Sample Test Queries")
    sample_queries = [
        "What is Agentic AI according to the eBook?",
        "How do AI agents differ from traditional automation systems?",
        "What are the core components of an Agentic Architecture?",
        "What role does memory play in Agentic AI workflows?",
        "Who won the 2022 FIFA World Cup?",
    ]
    for q in sample_queries:
        if st.button(q, use_container_width=True):
            st.session_state["pending_query"] = q

    st.divider()
    st.info(
        "**Strict Grounding Rule:**\n"
        "The model is instructed to answer ONLY based on retrieved eBook chunks. "
        "Questions outside the context will be strictly refused."
    )

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Cached graph builder for direct execution
@st.cache_resource(show_spinner="Initializing LangGraph RAG Engine...")
def get_cached_graph():
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        return None
    return build_rag_graph()


def query_rag(prompt: str):
    """Query FastAPI endpoint if alive, otherwise fallback to local graph."""
    api_url = f"http://{HOST}:{PORT}/chat"
    try:
        res = requests.post(api_url, json={"query": prompt}, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return data["final_answer"], data["retrieved_context"], data["confidence_score"]
    except Exception:
        pass

    # Fallback to direct LangGraph invocation
    graph = get_cached_graph()
    if graph is None:
        raise ValueError("API Keys not configured in .env. Please set OPENAI_API_KEY and PINECONE_API_KEY.")
    
    result = graph.invoke({"question": prompt, "context": [], "answer": "", "score": 0.0})
    return result["answer"], result["context"], result["score"]


# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "score" in msg:
            score = msg["score"]
            badge_color = "green" if score >= 0.5 else "red"
            st.caption(f":{badge_color}[**Confidence Score:** {score * 100:.0f}%]")
        if "context" in msg and msg["context"]:
            with st.expander("📚 View Retrieved Context Chunks"):
                for idx, chunk in enumerate(msg["context"], 1):
                    st.markdown(f"**Chunk {idx}:**")
                    st.text(chunk)
                    st.divider()

# Handle input
user_prompt = st.chat_input("Ask a question about the Agentic AI eBook...")
if "pending_query" in st.session_state and st.session_state["pending_query"]:
    user_prompt = st.session_state.pop("pending_query")

if user_prompt:
    # Display user query
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Process response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving from Pinecone & generating grounded answer..."):
            try:
                answer, context, score = query_rag(user_prompt)
                st.markdown(answer)

                score_color = "green" if score >= 0.5 else "red"
                st.caption(f":{score_color}[**Confidence Score:** {score * 100:.0f}%]")

                if context:
                    with st.expander("📚 View Retrieved Context Chunks", expanded=True):
                        for idx, chunk in enumerate(context, 1):
                            st.markdown(f"**Chunk {idx}:**")
                            st.text(chunk)
                            st.divider()
                else:
                    st.caption("*(No context chunks retrieved)*")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "score": score,
                    "context": context,
                })
            except Exception as e:
                st.error(f"Error: {str(e)}")
