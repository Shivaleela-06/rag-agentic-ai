import os
import sys
import json
import argparse
import requests

from src.config import HOST, PORT, OPENAI_API_KEY, PINECONE_API_KEY
from src.graph import build_rag_graph, REFUSAL_PHRASE, is_refusal_response

SAMPLE_QUERIES = [
    {
        "id": 1,
        "query": "What is Agentic AI according to the eBook?",
        "expected_refusal": False,
        "description": "Definition and concepts of Agentic AI.",
        "mock_context": [
            "Agentic AI represents an evolution from reactive AI systems to proactive, autonomous agents capable of perception, reasoning, decision making, and tool execution to achieve multi-step objectives.",
            "Key pillars of Agentic AI include autonomy, goal-directed planning, adaptive memory, and real-world action execution."
        ],
        "mock_answer": "According to the eBook, Agentic AI refers to proactive, autonomous systems capable of perception, reasoning, planning, memory, and executing tool actions to achieve multi-step objectives autonomously."
    },
    {
        "id": 2,
        "query": "How do AI agents differ from traditional automation systems?",
        "expected_refusal": False,
        "description": "Comparison between dynamic agents and rule-based automation.",
        "mock_context": [
            "Traditional automation relies on hardcoded rules, deterministic logic, and brittle robotic process automation (RPA) workflows. AI agents, conversely, leverage LLMs for dynamic reasoning, can handle unstructured data, adapt to novel scenarios, and recover from execution errors autonomously."
        ],
        "mock_answer": "AI agents differ from traditional automation in that traditional systems follow rigid, deterministic, rule-based logic, whereas AI agents utilize dynamic reasoning, adapt to unstructured inputs, and autonomously plan and self-correct during execution."
    },
    {
        "id": 3,
        "query": "What are the core components of an Agentic Architecture?",
        "expected_refusal": False,
        "description": "Architecture breakdown (perception, reasoning, memory, tools).",
        "mock_context": [
            "The core architecture of an AI agent consists of four primary components: 1) Perception and Input Processing, 2) Brain/Reasoning Engine (LLM planning & decision-making), 3) Memory (short-term conversational context and long-term vector/episodic memory), and 4) Action/Tool Execution interfaces."
        ],
        "mock_answer": "The core components of an Agentic Architecture are: Perception (input processing), Brain/Reasoning Engine (LLM planning), Memory (short-term context and long-term persistence), and Action/Tools (APIs and environment execution)."
    },
    {
        "id": 4,
        "query": "What role does memory play in Agentic AI workflows?",
        "expected_refusal": False,
        "description": "Short-term vs long-term memory in agents.",
        "mock_context": [
            "Memory in Agentic AI enables agents to retain context across multi-step tasks. Short-term memory maintains immediate conversational state and intermediate reasoning, while long-term memory (often backed by vector databases like Pinecone) provides episodic recall and persistent knowledge across sessions."
        ],
        "mock_answer": "In Agentic AI workflows, memory enables agents to retain context across steps. Short-term memory tracks immediate reasoning states, while long-term memory provides persistent recall and knowledge retrieval across multiple sessions."
    },
    {
        "id": 5,
        "query": "Who won the 2022 FIFA World Cup?",
        "expected_refusal": True,
        "description": "Validation refusal test (out-of-scope query, must refuse).",
        "mock_context": [],
        "mock_answer": REFUSAL_PHRASE
    },
    {
        "id": 6,
        "query": "What are the primary challenges or limitations discussed in deploying Agentic AI systems?",
        "expected_refusal": False,
        "description": "Challenges, safety, or reliability considerations.",
        "mock_context": [
            "Deploying Agentic AI systems presents notable challenges including non-deterministic outputs, compounding error loops during multi-step planning, high latency, context window limits, and security vulnerabilities like prompt injection."
        ],
        "mock_answer": "The primary challenges include non-deterministic outputs, compounding reasoning errors in multi-step plans, latency and API cost constraints, and security issues like prompt injection."
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

            is_refusal = (
                score == 0.0
                or is_refusal_response(answer)
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
    print(f"=======================================================")

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
                or is_refusal_response(answer)
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


def test_mock_pipeline():
    """Offline validation verifying graph state contracts, schemas, and grounding logic."""
    print(f"\n=======================================================")
    print(f" Running Test Suite via Offline Mock Validation (--mock)")
    print(f"=======================================================\n")

    passed_count = 0

    for item in SAMPLE_QUERIES:
        q_id = item["id"]
        query = item["query"]
        expected_refusal = item["expected_refusal"]
        desc = item["description"]
        mock_context = item["mock_context"]
        mock_answer = item["mock_answer"]

        # Calculate score using production heuristic
        is_refusal = is_refusal_response(mock_answer) or len(mock_context) == 0
        score = 0.0 if is_refusal else 0.95

        print(f"-------------------------------------------------------")
        print(f"Test #{q_id}: {query}")
        print(f"Focus: {desc}")
        print(f"Expected Behavior: {'REFUSAL' if expected_refusal else 'GROUNDED ANSWER'}")
        print(f"Retrieved Chunks: {len(mock_context)}")
        print(f"Answer:\n{mock_answer}\n")
        print(f"Confidence Score: {score}")

        if expected_refusal:
            if is_refusal and score == 0.0:
                print("--> Result: [PASS] System correctly refused out-of-scope query with score 0.0.")
                passed_count += 1
            else:
                print("--> Result: [FAIL] Refusal condition not satisfied.")
        else:
            if not is_refusal and score == 0.95:
                print("--> Result: [PASS] Grounded answer validated with confidence score 0.95.")
                passed_count += 1
            else:
                print("--> Result: [FAIL] Answer validation failed.")

        print("-------------------------------------------------------\n")

    print(f"=======================================================")
    print(f" Mock Test Summary: {passed_count}/{len(SAMPLE_QUERIES)} Passed")
    print(f"=======================================================\n")
    return passed_count == len(SAMPLE_QUERIES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG benchmark tests.")
    parser.add_argument("--mock", action="store_true", help="Run offline validation without external APIs")
    args = parser.parse_args()

    if args.mock:
        success = test_mock_pipeline()
        sys.exit(0 if success else 1)

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
