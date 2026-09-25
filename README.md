# Advanced Retrieval-Augmented Generation (RAG) Architecture Suite

A production-grade collection of advanced Retrieval-Augmented Generation (RAG) paradigms implemented using **LangGraph**, **LangChain**, **Groq LLM** (`openai/gpt-oss-120b`), and **HuggingFace Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`).

---

## Architecture Modules

### 1. Corrective RAG (CRAG) (`CorrectiveRAG/`)
- **Self-Correction & Web Fallback**: Uses LLM-as-a-judge to evaluate document relevance before generation.
- **Tavily Web Search Integration**: If retrieved documents are ambiguous or incorrect, fallback web queries retrieve fresh external context.
- **Artifacts**: [corrective_rag.ipynb](CorrectiveRAG/corrective_rag.ipynb), [corrective_rag.html](CorrectiveRAG/corrective_rag.html), [workflow_architecture.png](CorrectiveRAG/workflow_architecture.png).

### 2. Adaptive RAG (`AdaptiveRAG/`)
- **Query Classification & Multi-Route Orchestration**: Direct conversational LLM response for chitchat, local vectorstore similarity retrieval for domain knowledge, or web search for broad web queries.
- **Hallucination & Answer Grading**: Dual-stage post-generation validation ensuring answers are strictly grounded in context.
- **Artifacts**: [adaptive_rag.ipynb](AdaptiveRAG/adaptive_rag.ipynb), [adaptive_rag.py](AdaptiveRAG/adaptive_rag.py), [adaptive_rag.html](AdaptiveRAG/adaptive_rag.html), [workflow_architecture.png](AdaptiveRAG/workflow_architecture.png).

### 3. Agentic RAG Suite (`AgenticRAG/`)
- **1-basic_agentic_rag.ipynb**: Baseline autonomous retrieval agent with LangGraph state graph.
- **2-ReAct.ipynb**: Multi-source ReAct (Reason + Act) agent equipped with 5 heterogeneous tools (FAISS local vectorstore, Wikipedia, ArXiv academic research, proprietary tech specs, and internal research notes).
- **desion_maker_rag.ipynb**: Decision-Maker RAG featuring dynamic domain routing, document relevance grading, and self-correcting query reformulation loops.
- **Artifacts**: Complete interactive Jupyter notebooks with full live outputs and embedded HTML documentation.

### 4. RAG with Persistent Memory (`RAG with Persistant Memory/`)
- **Conversational Memory Integration**: LangGraph `MemorySaver` checkpointer for stateful multi-turn question answering across sessions.
- **Artifacts**: [ragmemory.ipynb](RAG%20with%20Persistant%20Memory/ragmemory.ipynb), [ragmemory.html](RAG%20with%20Persistant%20Memory/ragmemory.html), [workflow_architecture.png](RAG%20with%20Persistant%20Memory/workflow_architecture.png).

### 5. Cache-Augmented Generation (CAG) (`CAG/`)
- **Preloaded Context Cache**: Eliminates retrieval latency for static reference materials by preloading and structuring domain documents directly in-context.
- **Artifacts**: [cache_augment_generation.ipynb](CAG/cache_augment_generation.ipynb), [cache_augment_generation.html](CAG/cache_augment_generation.html), [workflow_architecture.png](CAG/workflow_architecture.png).

---

## Tech Stack & Setup

| Component | Specification |
|---|---|
| **LLM Engine** | Groq `openai/gpt-oss-120b` via `langchain.chat_models.init_chat_model` |
| **Embeddings** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | In-Memory FAISS (`langchain_community.vectorstores.FAISS`) |
| **Orchestration** | LangGraph `StateGraph` & `create_react_agent` |
| **Tools** | Wikipedia, ArXiv, Tavily Search, Local File Loaders |

### Environment Variables
Create a `.env` file in the root directory (see `.env.example`):
```bash
GROQ_API_KEY=your_groq_api_key_here
USER_AGENT=RAGWorkflow/1.0
TAVILY_API_KEY=your_tavily_api_key_here # optional for web search
```

### Installation
```bash
uv venv
source .venv/bin/activate # or .venv\Scripts\activate on Windows
pip install -r requirment.txt
```
