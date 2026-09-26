# Advanced Retrieval-Augmented Generation (RAG) Architecture Suite

A comprehensive, production-grade repository demonstrating modern **Retrieval-Augmented Generation (RAG)** paradigms, agentic retrieval workflows, semantic search techniques, multimodal document understanding, and document parsing pipelines.

Every module follows a unified, deterministic, and production-tested technology stack:
- **LLM Engine**: Groq `openai/gpt-oss-120b` via `langchain.chat_models.init_chat_model` (fast, deterministic, zero-shot structured outputs).
- **Embeddings Model**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (`langchain_huggingface.HuggingFaceEmbeddings` / `HuggingFaceEmbeddings`).
- **Vision Model**: Multimodal CLIP (`openai/clip-vit-base-patch32`) + GPT-4o / Vision LLMs.
- **Orchestration**: LangGraph `StateGraph`, `MessagesState`, `MemorySaver`, and `create_react_agent`.
- **Vector Storage**: FAISS (In-Memory Euclidean / Cosine) and ChromaDB (Persistent HNSW + SQLite).
- **Visuals & Reports**: Custom colorful Mermaid architecture workflows, exported diagrams, and self-contained interactive HTML companion reports with full live outputs.

---

## Topic-Wise Repository Architecture & Status Matrix

| # | Topic / Folder | Core Paradigm | Key Artifacts | Models & Embeddings | Diagram | Live Outputs | HTML Companion | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | [`CAG/`](CAG/) | **Cache-Augmented Generation** | `cache_augment_generation.ipynb` | Groq `gpt-oss-120b` + MiniLM | Yes | Yes | Yes | **Covered** |
| 2 | [`AdaptiveRAG/`](AdaptiveRAG/) | **Adaptive Routing & Self-Grading** | `adaptive_rag.ipynb`, `adaptive_rag.py` | Groq `gpt-oss-120b` + MiniLM | Yes | Yes | Yes | **Covered** |
| 3 | [`CorrectiveRAG/`](CorrectiveRAG/) | **CRAG + Web Search Fallback** | `corrective_rag.ipynb` | Groq `gpt-oss-120b` + MiniLM | Yes | Yes | Yes | **Covered** |
| 4 | [`AgenticRAG/`](AgenticRAG/) | **Multi-Source ReAct & Decision Agents** | `1-basic_agentic_rag.ipynb`, `2-ReAct.ipynb`, `desion_maker_rag.ipynb`, Studio | Groq `gpt-oss-120b` + MiniLM | Yes | Yes | Yes | **Covered** |
| 5 | [`PersistentMemoryRAG/`](PersistentMemoryRAG/) | **Stateful Multi-Turn MemorySaver** | `ragmemory.ipynb` | Groq `gpt-oss-120b` + MiniLM | Yes | Yes | Yes | **Covered** |
| 6 | [`AutonomousRAG/`](AutonomousRAG/) | **Self-Reflection, CoT & Iterative RAG** | 5 Notebooks (`self_reflection`, `chain_of_thoughts`, `iterative_retrieval`, `query_planner`, `answersynthesis`) | Groq `gpt-oss-120b` + MiniLM | Yes | Yes | Yes | **Covered** |
| 7 | [`AdvancedRetrieval/`](AdvancedRetrieval/) | **HyDE, Reranking, Hybrid Search & MMR** | 7 Notebooks (`hyde`, `query-expansion`, `querydecomposition`, `re-ranking_with_llm`, `hybrid_search`, `mmr`, `cosinesimilarity`) | Groq `gpt-oss-120b` + MiniLM | Yes | Yes | Yes | **Covered** |
| 8 | [`ChunkingStrategies/`](ChunkingStrategies/) | **Semantic Chunking & Text Splitters** | `loaders_and_splitters.ipynb`, `semantic_chunking.ipynb`, `textsplitter.ipynb` | HuggingFace MiniLM Embeddings | Yes | Yes | Yes | **Covered** |
| 9 | [`MultimodalRAG/`](MultimodalRAG/) | **Cross-Modal PDF (Text + Figures) RAG** | `1-multimodalopenai.ipynb`, Sample PDFs | CLIP `clip-vit-base-patch32` + Vision LLM | Yes | Yes | Yes | **Covered** |
| 10 | [`VectorStoresAndEmbeddings/`](VectorStoresAndEmbeddings/) | **FAISS, ChromaDB & Persistence** | `faisstest.ipynb`, `charomadb.ipynb` | Groq `gpt-oss-120b` + MiniLM | Yes | Yes | Yes | **Covered** |
| 11 | [`0-DataIngestParsing/`](0-DataIngestParsing/) | **Document Ingestion & Multi-Format Parsers** | 6 Notebooks (PDF, Word DOCX, CSV/Excel, JSON, SQL Databases) | PyPDF, PyMuPDF, Unstructured | Yes (SVG) | Yes | Yes | **Covered** |
| 12 | [`Langchain/`](Langchain/) | **LangChain Framework Foundations** | 8 Notebooks (Guardrails, Tools, Structured Output, Init Model, Middleware) | Multi-Provider (Groq, Google) | Yes | Yes | Yes | **Covered** |
| 13 | [`Langgraph/`](Langgraph/) | **StateGraph & Multi-Tool Graphs** | 6 Notebooks (StateGraph, Pydantic Schema, Reducers, Chains) | LangGraph Primitives | Yes | Yes | Yes | **Covered** |

> **Summary**: All 13 core modules across foundational parsing, advanced indexing, agentic decision-making, autonomous cognitive loops, and multimodal RAG have been **100% Covered** with full live execution outputs, custom architecture diagrams, and HTML reports.

---

## Detailed Topic Guides & Architecture Breakdowns

### 1. Cache-Augmented Generation ([`CAG/`](CAG/))
- **Concept**: Eliminates document chunking and vector retrieval overhead for static corpora by pre-loading and locking tokenized knowledge directly into the LLM's context window / prompt cache.
- **Workflow**: Pre-cached Documents -> Direct Context Extraction -> Low-Latency Generation.
- **Artifacts**:
  - Notebook: [`cache_augment_generation.ipynb`](CAG/cache_augment_generation.ipynb)
  - Interactive HTML: [`cache_augment_generation.html`](CAG/cache_augment_generation.html)
  - Diagram: [`workflow_architecture.png`](CAG/workflow_architecture.png)

---

### 2. Adaptive RAG ([`AdaptiveRAG/`](AdaptiveRAG/))
- **Concept**: Dynamically evaluates incoming queries to select the optimal retrieval path: direct response for chit-chat, vector store retrieval for internal domain documents, or live web search (via Tavily) for recent events. Employs dual-stage verification:
  1. *Hallucination Grader*: Checks if the generated answer is grounded in retrieved facts.
  2. *Answer Grader*: Checks if the answer fully addresses the user query; if not, triggers query reformulation loops.
- **Artifacts**:
  - Notebook: [`adaptive_rag.ipynb`](AdaptiveRAG/adaptive_rag.ipynb)
  - Standalone Script: [`adaptive_rag.py`](AdaptiveRAG/adaptive_rag.py)
  - Interactive HTML: [`adaptive_rag.html`](AdaptiveRAG/adaptive_rag.html)
  - Diagram: [`workflow_architecture.png`](AdaptiveRAG/workflow_architecture.png)

---

### 3. Corrective RAG (CRAG) ([`CorrectiveRAG/`](CorrectiveRAG/))
- **Concept**: Evaluates retrieved documents using an LLM-as-a-judge document relevance grader before generating an answer. If retrieved documents are ambiguous, low-quality, or irrelevant, CRAG automatically intervenes and conducts a corrective web search using Tavily.
- **Artifacts**:
  - Notebook: [`corrective_rag.ipynb`](CorrectiveRAG/corrective_rag.ipynb)
  - Interactive HTML: [`corrective_rag.html`](CorrectiveRAG/corrective_rag.html)
  - Diagram: [`workflow_architecture.png`](CorrectiveRAG/workflow_architecture.png)

---

### 4. Agentic RAG Suite ([`AgenticRAG/`](AgenticRAG/))
- **Concept**: Autonomous agents equipped with tools to query diverse heterogeneous sources (internal FAISS knowledge bases, live Wikipedia, ArXiv academic papers, and local technical docs) using dynamic reasoning loops.
- **Key Modules**:
  - `1-basic_agentic_rag.ipynb`: Baseline tool-calling agent using LangGraph.
  - `2-ReAct.ipynb`: Multi-source ReAct (Reasoning + Acting) agent synthesizing answers across 4 distinct knowledge tools.
  - `desion_maker_rag.ipynb`: Advanced decision-maker RAG with dynamic query classification, document evaluation, and query rewriting.
  - `ReAct_LangGraph_Studio/`: Production-ready LangGraph Studio application with streaming agent checkpoints.
- **Artifacts**: Matching notebooks, HTML reports, and Mermaid workflows.

---

### 5. Persistent Memory RAG ([`PersistentMemoryRAG/`](PersistentMemoryRAG/))
- **Concept**: Multi-turn conversational RAG system that maintains state across session turns using LangGraph's `MemorySaver` checkpointer and `MessagesState`. It intelligently decides when to search the knowledge base versus answering directly from prior context.
- **Artifacts**:
  - Notebook: [`ragmemory.ipynb`](PersistentMemoryRAG/ragmemory.ipynb)
  - Interactive HTML: [`ragmemory.html`](PersistentMemoryRAG/ragmemory.html)
  - Diagram: [`workflow_architecture.png`](PersistentMemoryRAG/workflow_architecture.png)

---

### 6. Autonomous RAG ([`AutonomousRAG/`](AutonomousRAG/))
- **Concept**: Advanced cognitive loops granting RAG pipelines self-correction, reasoning decomposition, and multi-step evidence gathering:
  - `self_reflection.ipynb`: Self-reflective loop evaluating factual faithfulness and refining answers.
  - `chain_of_thoughts_with_rag.ipynb`: Step-by-step reasoning interleaved with retrieval checkpoints.
  - `iterative_retrieval.ipynb`: Multi-hop iterative question answering with cumulative evidence accumulation.
  - `query_planner_and_decomposition.ipynb`: Decomposes complex queries into ordered sub-queries.
  - `7-answersynthesis.ipynb`: Synthesizes multi-perspective, cross-document answers.
- **Artifacts**: 5 fully executed notebooks, matching `.html` reports, and 5 dedicated architecture diagrams (`workflow_*.png`).

---

### 7. Advanced Retrieval ([`AdvancedRetrieval/`](AdvancedRetrieval/))
- **Concept**: Enhances search quality, resolves the lexical-semantic gap, and optimizes document ranking:
  - `hyde.ipynb`: Hypothetical Document Embeddings (HyDE) — generating hypothetical answers to retrieve geometrically closer documents.
  - `query-expansion.ipynb`: Multi-query generation to broaden retrieval coverage.
  - `querydecomposition.ipynb`: Recursive query decomposition for multi-part questions.
  - `re-ranking_with_llm.ipynb`: Two-stage retrieval pipeline with cross-encoder / LLM re-ranking.
  - `hybrid_search.ipynb`: Reciprocal Rank Fusion (RRF) combining dense vector search and sparse BM25 lexical search.
  - `mmr.ipynb`: Maximal Marginal Relevance for maximizing diversity and minimizing redundancy.
  - `cosinesimilarity.ipynb`: Deep geometric analysis of vector space distance metrics.
- **Artifacts**: 7 fully executed notebooks, matching `.html` reports, and 7 dedicated architecture diagrams (`workflow_*.png`).

---

### 8. Chunking Strategies ([`ChunkingStrategies/`](ChunkingStrategies/))
- **Concept**: Techniques for segmenting long documents while preserving semantic boundaries:
  - `loaders_and_splitters.ipynb`: Comprehensive comparison of document loaders and character/token splitters.
  - `semantic_chunking.ipynb`: Semantic boundary chunking based on embedding cosine distance percentiles.
  - `textsplitter.ipynb`: Recursive character text splitting benchmarks.
- **Artifacts**: 3 fully executed notebooks, matching `.html` reports, and 3 dedicated architecture diagrams (`workflow_*.png`).

---

### 9. Multimodal RAG ([`MultimodalRAG/`](MultimodalRAG/))
- **Concept**: Ingests, embeds, and reasons across both text and visual elements (charts, diagrams, tables) extracted from complex PDF documents:
  - **Extractor**: PyMuPDF extracts text passages and raster images.
  - **Shared Embedding Space**: OpenAI CLIP (`openai/clip-vit-base-patch32`) projects text and images into a single geometric vector space.
  - **Unified Index**: Single FAISS index for cross-modal similarity search.
  - **Multimodal Generation**: Constructs messages with text context and base64-encoded image payloads for Vision LLMs (GPT-4o / Vision OSS).
- **Artifacts**:
  - Notebook: [`1-multimodalopenai.ipynb`](MultimodalRAG/1-multimodalopenai.ipynb)
  - Interactive HTML: [`1-multimodalopenai.html`](MultimodalRAG/1-multimodalopenai.html)
  - Diagram: [`workflow_multimodal_rag.png`](MultimodalRAG/workflow_multimodal_rag.png)

---

### 10. Vector Stores & Embeddings ([`VectorStoresAndEmbeddings/`](VectorStoresAndEmbeddings/))
- **Concept**: Practical guide to vector indexing, semantic search mechanics, and database persistence:
  - `faisstest.ipynb`: FAISS in-memory index creation, L2/cosine similarity comparison, threshold scoring, metadata filtering, and LCEL RAG chains with Groq `openai/gpt-oss-120b`.
  - `charomadb.ipynb`: ChromaDB persistent HNSW indexing, SQLite metadata storage, and modern retrieval QA chains.
- **Artifacts**: 2 fully executed notebooks, matching `.html` reports, and architecture diagram [`workflow_vectorstores.png`](VectorStoresAndEmbeddings/workflow_vectorstores.png).

---

### 11. Document Ingestion & Parsing ([`0-DataIngestParsing/`](0-DataIngestParsing/))
- **Concept**: End-to-end parsing pipelines for multi-format enterprise data:
  - `1-dataingestion.ipynb`: Core ingestion pipelines and architecture.
  - `2-dataparsingpdf.ipynb`: PyPDF, PyMuPDF, and pdfplumber extraction.
  - `3-dataparsingdoc.ipynb`: Microsoft Word `.docx` parsing.
  - `4-csvexcelparsing.ipynb`: Structured tabular parsing (CSV, Excel).
  - `5-jsonparsing.ipynb`: Semi-structured JSON schema extraction.
  - `6-databaseparsing.ipynb`: Relational SQL database extraction.
- **Artifacts**: 6 notebooks with live execution outputs and architecture diagrams (`.svg`).

---

### 12. LangChain & LangGraph Foundations ([`Langchain/`](Langchain/), [`Langgraph/`](Langgraph/))
- **Concept**: Production building blocks for modern LLM applications:
  - Guardrails, structured Pydantic outputs, dynamic tool calling, and middleware.
  - LangGraph StateGraphs, typed state schemas, message reducers, and multi-agent coordination.
- **Artifacts**: 14 notebooks with live execution outputs.

---

## Environment Setup & Installation

### 1. Prerequisites & Dependencies
Clone the repository and install the verified dependencies:
```bash
git clone https://github.com/itsarpitgupta/RAG.git
cd RAG
pip install -r requirment.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```bash
# Groq API Key (Primary LLM engine)
GROQ_API_KEY=gsk_your_groq_api_key_here

# Tavily API Key (Corrective & Web Search fallback)
TAVILY_API_KEY=tvly-your_tavily_api_key_here

# OpenAI API Key (Multimodal Vision / Fallback)
OPENAI_API_KEY=sk-your_openai_api_key_here

# User Agent for Wikipedia & Web Search
USER_AGENT=RAGWorkflowApp/1.0
```

### 3. Running Notebooks & Viewing Reports
- **Interactive UI**: Launch Jupyter Lab or Notebook:
  ```bash
  jupyter lab
  ```
- **HTML Companion Reports**: All upgraded folders contain pre-rendered `.html` reports (e.g. `CAG/cache_augment_generation.html`, `AutonomousRAG/self_reflection.html`, `MultimodalRAG/1-multimodalopenai.html`) that can be opened directly in any browser with zero installation needed.
