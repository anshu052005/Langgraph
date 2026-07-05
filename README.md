# 🧠 Memory in LLMs , LangSmit and  LangGraph Learning Journey

This repo documents my hands-on learning of **LangGraph and LangChain** — starting from the basics and progressing to advanced concepts like tools, RAG, human-in-the-loop, subgraphs, and MCP. Each notebook builds on the previous one with practical, working code.

---

## 📚 What I've Learned

### 1. Setup & Basics
- **`0_test_installation.ipynb`** — Basic environment and dependency setup/testing.
- **`2_simple_llm_workflow.ipynb`** — The simplest LangGraph LLM-based workflow — introduction to graphs, nodes, and state.

### 2. Simple Graph Workflows (Practice Projects)
Practiced LangGraph fundamentals by building small, real-world-style use cases:
- **`1_bmi_workflow.ipynb`** — A workflow to calculate BMI (multi-node flow).
- **`4_batsman_workflow.ipynb`** — A graph to analyze a cricket batsman's stats.
- **`6_quadratic_equation.ipynb`** — A conditional/logic-based workflow to solve quadratic equations.

### 3. Prompt Engineering & Chaining
- **`3_prompt_chaining.ipynb`** — Chaining multiple prompts together to generate complex output.
- **`5_UPSC_essay_workflow.ipynb`** — An essay-writing workflow — a multi-step pipeline for planning → drafting → refining.
- **`7_review_reply_workflow.ipynb`** — A workflow to auto-generate replies to product/app reviews.
- **`8_X_post_generator.ipynb`** — An LLM workflow to generate posts for Twitter/X.

### 4. Chatbots & Memory
- **`9_basic_chatbot.ipynb`** — A basic conversational chatbot built with LangGraph.
- **`10_persistence.ipynb`** — **Persisting** chatbot/graph state (checkpointing) so conversation history/memory is maintained across turns.
- **`chatbot_without_hitl.py`** — A simple chatbot script without human-in-the-loop.
- **`chatbot_with_hitl.py`** — The same chatbot, now with **Human-in-the-Loop (HITL)** support.

### 5. Tools, MCP & RAG
- **`11_tools.ipynb`** — Integrating external **tools** with the LLM (tool calling).
- **`12_mcp.py`** — Using the **Model Context Protocol (MCP)** to connect external services/tools via a standard protocol.
- **`13_rag.ipynb`** — **Retrieval-Augmented Generation (RAG)** — retrieving relevant info from an external knowledge base to improve LLM responses.

### 6. Advanced Control Flow
- **`14_hitl.ipynb`** — A deeper dive into **Human-in-the-Loop** — pausing graph execution for human approval/input.
- **`15_subgraphs.ipynb`** — **Subgraphs** — breaking complex workflows into smaller, reusable graphs.
- **`15_subgraph_shared.ipynb`** — Managing **shared state** across subgraphs.

### 7. Observability
- **`LangSmith`** — Using LangSmith to **trace, debug, and monitor** LangChain/LangGraph applications.

---

## 🛠️ Tech Stack
- Python
- LangChain
- LangGraph
- LangSmith
- MCP (Model Context Protocol)

## 📂 Setup
```bash
python -m venv myenv
source myenv/bin/activate   # Windows: myenv\Scripts\activate
pip install -r requirements.txt
```

---

## 🎯 Goal

The goal of this repo is to understand and practice the core concepts of LangGraph — **graphs, state, persistence, tools, RAG, HITL, subgraphs, and MCP** — through hands-on notebooks and scripts.
