import os
from typing import List, Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    TOP_K,
)


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float


REFUSAL_PHRASE = "I cannot answer based on the provided document."


def is_refusal_response(answer: str) -> bool:
    """
    Check if the answer indicates that the information was missing or refused.
    """
    normalized = answer.lower()
    refusal_signals = [
        "cannot answer based on the provided document",
        "does not contain enough info",
        "does not contain sufficient info",
        "not mentioned in the provided context",
        "context does not mention",
        "context does not provide",
        "no information provided",
    ]
    return any(signal in normalized for signal in refusal_signals)


def build_rag_graph(
    index_name: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    pinecone_api_key: Optional[str] = None,
    top_k: int = TOP_K,
    llm_model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
):
    """
    Build and compile the LangGraph RAG execution workflow.
    
    Nodes:
    1. retrieve: Queries Pinecone for top-k relevant chunks based on user prompt.
    2. generate: LLM formulates response strictly grounded in context chunks and computes confidence score.
    """
    target_index = index_name or PINECONE_INDEX_NAME
    oai_key = openai_api_key or OPENAI_API_KEY
    pc_key = pinecone_api_key or PINECONE_API_KEY

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=oai_key,
    )

    vectorstore = PineconeVectorStore(
        index_name=target_index,
        embedding=embeddings,
        pinecone_api_key=pc_key,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    llm = ChatOpenAI(
        model=llm_model,
        temperature=temperature,
        openai_api_key=oai_key,
    )

    # Node 1: Retrieve context chunks
    def retrieve_node(state: AgentState):
        question = state.get("question", "")
        if not question.strip():
            return {"context": []}

        try:
            docs = retriever.invoke(question)
            context_texts = [d.page_content for d in docs]
        except Exception as e:
            print(f"[Graph:Retrieve] Warning: Retrieval encountered an error: {e}")
            context_texts = []

        return {"context": context_texts}

    # Node 2: Generate strictly grounded response
    def generate_node(state: AgentState):
        question = state.get("question", "")
        context_list = state.get("context", [])

        if not context_list:
            return {
                "answer": REFUSAL_PHRASE,
                "score": 0.0,
            }

        context_str = "\n\n--- CHUNK ---\n\n".join(context_list)

        system_prompt = (
            "You are a strict, factual assistant for an Agentic AI knowledge base. "
            "Your task is to answer the question using ONLY the provided context excerpts.\n\n"
            "STRICT RULES:\n"
            f"1. If the provided context does not contain enough direct information to answer the question, you MUST reply: '{REFUSAL_PHRASE}'.\n"
            "2. Do NOT use outside knowledge, speculation, or extrapolation.\n"
            "3. Answer concisely, accurately, and professionally based strictly on facts in the context."
        )

        user_prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        answer_text = response.content.strip()

        # Score computation:
        # If response refuses or context is missing -> 0.0
        # If successfully answered with context -> high confidence score (0.95)
        if is_refusal_response(answer_text):
            confidence = 0.0
        else:
            confidence = 0.95

        return {
            "answer": answer_text,
            "score": confidence,
        }

    # Assemble LangGraph StateGraph
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
