import os
import sys
import json
import requests

from src.config import HOST, PORT, OPENAI_API_KEY, PINECONE_API_KEY
from src.graph import build_rag_graph

SAMPLE_QUERIES = [
    {
        "id": 1,
        "query": "What is Agentic AI according to the eBook?",
        "expected_refusal": False,
        "description": "Definition and concepts of Agentic AI.",
    },
    {
        "id": 2,
        "query": "How do AI agents differ from traditional automation systems?",
        "expected_refusal": False,
        "description": "Comparison between dynamic agents and rule-based automation.",
    },
    {
        "id": 3,
        "query": "What are the core components of an Agentic Architecture?",
        "expected_refusal": False,
        "description": "Architecture breakdown (perception, reasoning, memory, tools).",
    },
    {
        "id": 4,
        "query": "What role does memory play in Agentic AI workflows?",
        "expected_refusal": False,
        "description": "Short-term vs long-term memory in agents.",
    },
    {
        "id": 5,
        "query": "Who won the 2022 FIFA World Cup?",
        "expected_refusal": True,
        "description": "Validation refusal test (out-of-scope query, must refuse).",
    },
    {
        "id": 6,
        "query": "What are the primary challenges or limitations discussed in deploying Agentic AI systems?",
        "expected_refusal": False,
        "description": "Challenges, safety, or reliability considerations.",
    },
]


def test_via_api(base_url: str):
    """Run tests by sending HTTP requests to the FastAPI backend."""
    print(f"\n=======================================================")
    print(f" Running Test Suite via FastAPI Endpoint ({base_url}/chat)")
    print(f"=======================================================\n")

    passed_count = 0

    for item in SAMPLE_QUERIES:
        q_id = item["id"]
        query = item["query"]
        expected_refusal = item["expected_refusal"]
        desc = item["description"]

        print(f"-------------------------------------------------------")
        print(f"Test #{q_id}: {query}")
        print(f"Focus: {desc}")
        print(f"Expected Behavior: {'REFUSAL' if expected_refusal else 'GROUNDED ANSWER'}")
        print(f"-------------------------------------------------------")

        try:
            res = requests.post(f"{base_url}/chat", json={"query": query}, timeout=30)
            if res.status_code != 200:
                print(f"[FAIL] HTTP Status {res.status_code}: {res.text}\n")
                continue

            data = res.json()
            answer = data.get("final_answer", "")
            context = data.get("retrieved_context", [])
            score = data.get("confidence_score", 0.0)

            print(f"Answer:\n{answer}\n")
            print(f"Confidence Score: {score}")
            print(f"Retrieved Chunks: {len(context)}")

            # Check grounding & refusal validation
            is_refusal = (
                score == 0.0
                or "cannot answer" in answer.lower()
                or "does not contain" in answer.lower()
            )

            if expected_refusal:
                if is_refusal:
                    print("--> Result: [PASS] System correctly refused out-of-scope query.")
                    passed_count += 1
                else:
                    print("--> Result: [FAIL] System should have refused but answered.")
            else:
                if not is_refusal and len(answer) > 20:
                    print("--> Result: [PASS] Grounded answer generated successfully.")
                    passed_count += 1
                else:
                    print("--> Result: [FAIL] Expected grounded answer, but got refusal or empty response.")

        except Exception as e:
            print(f"[ERROR] Request failed: {e}")

        print("\n")

    print(f"=======================================================")
    print(f" Test Summary: {passed_count}/{len(SAMPLE_QUERIES)} Passed")
    print(f"=======================================================\n")
    return passed_count == len(SAMPLE_QUERIES)


def test_via_graph():
    """Run tests directly invoking the LangGraph workflow."""
    print(f"\n=======================================================")
    print(f" Running Test Suite via Direct LangGraph Invocation")
    print(f"=======================================================\n")

    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        print("[Notice] OPENAI_API_KEY or PINECONE_API_KEY not found in environment.")
        print("Please configure .env before running live tests with Pinecone and OpenAI.")
        return False

    graph = build_rag_graph()
    passed_count = 0

    for item in SAMPLE_QUERIES:
        q_id = item["id"]
        query = item["query"]
        expected_refusal = item["expected_refusal"]
        desc = item["description"]

        print(f"-------------------------------------------------------")
        print(f"Test #{q_id}: {query}")
        print(f"Focus: {desc}")
        print(f"Expected Behavior: {'REFUSAL' if expected_refusal else 'GROUNDED ANSWER'}")
        print(f"-------------------------------------------------------")

        try:
            result = graph.invoke({
                "question": query,
                "context": [],
                "answer": "",
                "score": 0.0,
            })

            answer = result["answer"]
            context = result["context"]
            score = result["score"]

            print(f"Answer:\n{answer}\n")
            print(f"Confidence Score: {score}")
            print(f"Retrieved Chunks: {len(context)}")

            is_refusal = (
                score == 0.0
                or "cannot answer" in answer.lower()
                or "does not contain" in answer.lower()
            )

            if expected_refusal:
                if is_refusal:
                    print("--> Result: [PASS] System correctly refused out-of-scope query.")
                    passed_count += 1
                else:
                    print("--> Result: [FAIL] System should have refused but answered.")
            else:
                if not is_refusal and len(answer) > 20:
                    print("--> Result: [PASS] Grounded answer generated successfully.")
                    passed_count += 1
                else:
                    print("--> Result: [FAIL] Expected grounded answer, but got refusal or empty response.")

        except Exception as e:
            print(f"[ERROR] Execution failed: {e}")

        print("\n")

    print(f"=======================================================")
    print(f" Test Summary: {passed_count}/{len(SAMPLE_QUERIES)} Passed")
    print(f"=======================================================\n")
    return passed_count == len(SAMPLE_QUERIES)


if __name__ == "__main__":
    # Check if API server is running on HOST:PORT
    base_url = f"http://{HOST}:{PORT}"
    server_alive = False
    try:
        res = requests.get(f"{base_url}/health", timeout=2)
        if res.status_code == 200:
            server_alive = True
    except Exception:
        server_alive = False

    if server_alive:
        print(f"[Info] Detected running FastAPI server at {base_url}.")
        success = test_via_api(base_url)
    else:
        print("[Info] FastAPI server not detected. Testing via direct LangGraph workflow...")
        success = test_via_graph()

    sys.exit(0 if success else 1)
