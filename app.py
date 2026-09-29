import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.graph import build_rag_graph
from src.config import PINECONE_INDEX_NAME, OPENAI_API_KEY, PINECONE_API_KEY

app = FastAPI(
    title="Agentic AI RAG API",
    description="Strictly grounded Retrieval-Augmented Generation (RAG) AI Chatbot powered by LangGraph, Pinecone, and OpenAI.",
    version="1.0.0",
)

# Global graph instance
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_rag_graph(index_name=os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index"))
    return _graph


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user question to ask the chatbot", example="What is Agentic AI according to the eBook?")


class QueryResponse(BaseModel):
    final_answer: str = Field(..., description="The final grounded answer produced by the chatbot")
    retrieved_context: List[str] = Field(default_factory=list, description="List of text chunks retrieved from Pinecone")
    confidence_score: float = Field(..., description="Confidence or relevance score between 0.0 and 1.0")


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint to verify API and environment readiness."""
    has_openai = bool(OPENAI_API_KEY)
    has_pinecone = bool(PINECONE_API_KEY)
    return {
        "status": "healthy" if (has_openai and has_pinecone) else "configuration_needed",
        "index_name": PINECONE_INDEX_NAME,
        "openai_configured": has_openai,
        "pinecone_configured": has_pinecone,
    }


@app.post("/chat", response_model=QueryResponse, tags=["Chat"])
async def chat_endpoint(request: QueryRequest):
    """
    RAG chat endpoint:
    1. Retrieves top-k chunks from Pinecone.
    2. Generates strictly grounded answer with LangGraph.
    3. Computes confidence score.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    graph = get_graph()
    initial_state = {
        "question": request.query.strip(),
        "context": [],
        "answer": "",
        "score": 0.0,
    }

    try:
        result = graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")

    return QueryResponse(
        final_answer=result["answer"],
        retrieved_context=result["context"],
        confidence_score=result["score"],
    )


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def web_ui():
    """Interactive web interface for testing the RAG chatbot directly in browser."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Agentic AI RAG Chatbot</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; min-height: 100vh; display: flex; flex-direction: column; }
    header { background: #1e293b; padding: 1.25rem 2rem; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }
    header h1 { font-size: 1.25rem; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 0.5rem; }
    header .badge { background: #0284c7; color: white; font-size: 0.75rem; padding: 0.25rem 0.6rem; border-radius: 9999px; text-decoration: none; }
    main { flex: 1; max-width: 1000px; width: 100%; margin: 0 auto; padding: 2rem 1.5rem; display: flex; flex-direction: column; gap: 1.5rem; }
    .hero { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.5rem; }
    .hero h2 { font-size: 1.1rem; color: #e2e8f0; margin-bottom: 0.5rem; }
    .hero p { font-size: 0.9rem; color: #94a3b8; line-height: 1.5; }
    .sample-queries { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; }
    .sample-btn { background: #334155; color: #cbd5e1; border: 1px solid #475569; padding: 0.4rem 0.8rem; border-radius: 8px; font-size: 0.8rem; cursor: pointer; transition: all 0.2s; }
    .sample-btn:hover { background: #0284c7; color: white; border-color: #38bdf8; }
    .chat-box { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; }
    .input-group { display: flex; gap: 0.75rem; }
    input[type="text"] { flex: 1; background: #0f172a; border: 1px solid #475569; border-radius: 8px; padding: 0.75rem 1rem; color: white; font-size: 0.95rem; outline: none; }
    input[type="text"]:focus { border-color: #38bdf8; }
    button.submit-btn { background: #0284c7; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 8px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
    button.submit-btn:hover { background: #0369a1; }
    button.submit-btn:disabled { background: #475569; cursor: not-allowed; }
    .response-card { background: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 1.25rem; display: none; flex-direction: column; gap: 1rem; }
    .response-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 0.75rem; }
    .score-badge { font-size: 0.8rem; font-weight: 700; padding: 0.25rem 0.75rem; border-radius: 9999px; }
    .score-high { background: #065f46; color: #34d399; border: 1px solid #059669; }
    .score-low { background: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }
    .answer-text { line-height: 1.6; color: #f1f5f9; font-size: 0.95rem; white-space: pre-wrap; }
    .context-accordion { display: flex; flex-direction: column; gap: 0.5rem; margin-top: 0.5rem; }
    .chunk-card { background: #1e293b; border-left: 3px solid #38bdf8; border-radius: 6px; padding: 0.75rem 1rem; font-size: 0.85rem; color: #cbd5e1; line-height: 1.4; }
    .chunk-header { font-size: 0.75rem; color: #38bdf8; font-weight: 700; margin-bottom: 0.3rem; }
    .spinner { display: inline-block; width: 1rem; height: 1rem; border: 2px solid #ffffff44; border-top-color: #ffffff; border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <header>
    <h1><span>🤖</span> Agentic AI RAG Assistant</h1>
    <a href="/docs" target="_blank" class="badge">Swagger API Docs</a>
  </header>
  <main>
    <div class="hero">
      <h2>Strict Grounding Knowledge Base: Agentic AI eBook</h2>
      <p>This assistant answers queries relying exclusively on context retrieved from the Agentic AI eBook. It strictly refuses ungrounded questions and out-of-scope inquiries.</p>
      <div class="sample-queries">
        <button class="sample-btn" onclick="fillQuery('What is Agentic AI according to the eBook?')">What is Agentic AI?</button>
        <button class="sample-btn" onclick="fillQuery('How do AI agents differ from traditional automation systems?')">AI Agents vs Automation</button>
        <button class="sample-btn" onclick="fillQuery('What are the core components of an Agentic Architecture?')">Core Architecture</button>
        <button class="sample-btn" onclick="fillQuery('What role does memory play in Agentic AI workflows?')">Role of Memory</button>
        <button class="sample-btn" onclick="fillQuery('Who won the 2022 FIFA World Cup?')">Validation Refusal Test</button>
      </div>
    </div>

    <div class="chat-box">
      <div class="input-group">
        <input type="text" id="queryInput" placeholder="Ask a question about Agentic AI..." onkeydown="if(event.key==='Enter') sendQuery()" />
        <button class="submit-btn" id="sendBtn" onclick="sendQuery()">Ask Assistant</button>
      </div>
    </div>

    <div class="response-card" id="responseCard">
      <div class="response-header">
        <strong style="color: #38bdf8;">Final Generated Answer</strong>
        <span class="score-badge" id="scoreBadge">Confidence: --</span>
      </div>
      <div class="answer-text" id="answerText"></div>
      <div>
        <h4 style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Retrieved Context Chunks</h4>
        <div class="context-accordion" id="chunksContainer"></div>
      </div>
    </div>
  </main>

  <script>
    function fillQuery(text) {
      document.getElementById('queryInput').value = text;
      sendQuery();
    }

    async function sendQuery() {
      const input = document.getElementById('queryInput');
      const btn = document.getElementById('sendBtn');
      const card = document.getElementById('responseCard');
      const answerEl = document.getElementById('answerText');
      const badgeEl = document.getElementById('scoreBadge');
      const chunksEl = document.getElementById('chunksContainer');

      const query = input.value.trim();
      if (!query) return;

      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Querying...';
      card.style.display = 'none';

      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: query })
        });
        const data = await res.json();

        if (!res.ok) {
          throw new Error(data.detail || 'Failed to fetch answer.');
        }

        card.style.display = 'flex';
        answerEl.innerText = data.final_answer;

        const score = data.confidence_score;
        badgeEl.innerText = `Confidence: ${(score * 100).toFixed(0)}%`;
        if (score >= 0.5) {
          badgeEl.className = 'score-badge score-high';
        } else {
          badgeEl.className = 'score-badge score-low';
        }

        chunksEl.innerHTML = '';
        if (data.retrieved_context && data.retrieved_context.length > 0) {
          data.retrieved_context.forEach((chunk, idx) => {
            const chunkDiv = document.createElement('div');
            chunkDiv.className = 'chunk-card';
            chunkDiv.innerHTML = `<div class="chunk-header">CHUNK ${idx + 1}</div><div>${chunk}</div>`;
            chunksEl.appendChild(chunkDiv);
          });
        } else {
          chunksEl.innerHTML = '<p style="color: #64748b; font-size: 0.85rem;">No context retrieved.</p>';
        }
      } catch (err) {
        card.style.display = 'flex';
        answerEl.innerHTML = `<span style="color: #f87171;">Error: ${err.message}</span>`;
        badgeEl.className = 'score-badge score-low';
        badgeEl.innerText = 'Error';
        chunksEl.innerHTML = '';
      } finally {
        btn.disabled = false;
        btn.innerText = 'Ask Assistant';
      }
    }
  </script>
</body>
</html>
"""
