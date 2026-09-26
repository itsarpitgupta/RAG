# Advanced Retrieval-Augmented Generation (RAG) Architecture Suite

A comprehensive, production-grade repository demonstrating modern **Retrieval-Augmented Generation (RAG)** paradigms, agentic retrieval workflows, semantic search techniques, and document parsing pipelines. 

Every upgraded module follows a unified, deterministic technology stack:
- **LLM Engine**: Groq `openai/gpt-oss-120b` via `langchain.chat_models.init_chat_model`
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (`langchain_huggingface.HuggingFaceEmbeddings`)
- **Orchestration**: LangGraph `StateGraph`, `MessagesState`, `MemorySaver`, and `create_react_agent`
- **Visuals & Reports**: Custom colorful Mermaid architecture workflows, exported diagrams, and self-contained HTML companion reports.

---

## Topic-Wise Repository Architecture & Status Matrix

| Folder | Core Paradigm / Topic | Key Artifacts | LLM & Embeddings | Diagram | Output | HTML Report | Status |
|---|---|---|---|---|---|---|---|
| [`CAG/`](CAG/) | **Cache-Augmented Generation** | `cache_augment_generation.ipynb` | Groq `gpt-oss-120b` + MiniLM | Yes | Yes | Yes | **Covered** |
| [`AdaptiveRAG/`](AdaptiveRAG/) | **Adaptive Query Routing & Verification** | `adaptive_rag.ipynb`, `adaptive_rag.py` | Groq `gpt-oss-120b` + MiniLM | Yes | Yes | Yes | **Covered** |
| [`CorrectiveRAG/`](CorrectiveRAG/) | **Self-Corrective RAG (CRAG) + Web Fallback** | `corrective_rag.ipynb` | Groq `gpt-oss-120b` + MiniLM | Yes | Yes | Yes | **Covered** |
| [`AgenticRAG/`](AgenticRAG/) | **Multi-Source ReAct & Decision Maker Agents** | `1-basic_agentic_rag.ipynb`, `2-ReAct.ipynb`, `desion_maker_rag.ipynb` | Groq `gpt-oss-120b` + MiniLM | Yes | Yes | Yes | **Covered** |
| [`PersistentMemoryRAG/`](PersistentMemoryRAG/) | **Conversational RAG with Stateful MemorySaver** | `ragmemory.ipynb` | Groq `gpt-oss-120b` + MiniLM | Yes | Yes | Yes | **Covered** |
| [`0-DataIngestParsing/`](0-DataIngestParsing/) | **Document Ingestion & Multi-Format Parsers** | 6 Notebooks (PDF, DOCX, CSV, JSON, SQL) | Text Loaders & Splitters | Yes (SVG) | Yes | N/A | **Covered** |
| [`Langchain/`](Langchain/) | **LangChain Foundations** | 6 Foundational Notebooks | Multi-Provider Basics | Yes | Yes | N/A | **Covered** |
| [`Langgraph/`](Langgraph/) | **LangGraph Workflows & StateGraphs** | 4 Foundational Graph Notebooks | LangGraph Primitives | Yes | Yes | N/A | **Covered** |
| [`AutonomousRAG/`](AutonomousRAG/) | **Self-Reflection, CoT & Iterative Retrieval** | `self_reflection.ipynb`, `chain_of_thoughts_with_rag.ipynb`, `iterative_retrieval.ipynb`, `query_planner_and_decomposition.ipynb`, `7-answersynthesis.ipynb` | Groq `gpt-oss-120b` + MiniLM | In Progress | In Progress | In Progress | **In Progress** |
| [`AdvancedRetrieval/`](AdvancedRetrieval/) | **HyDE, Query Expansion, Re-ranking & Hybrid Search** | `hyde.ipynb`, `query-expansion.ipynb`, `querydecomposition.ipynb`, `re-ranking_with_llm.ipynb`, `hybrid_search.ipynb`, `mmr.ipynb`, `cosinesimilarity.ipynb` | In Upgrade Pipeline | In Progress | In Progress | In Progress | **Pending** |
| [`ChunkingStrategies/`](ChunkingStrategies/) | **Semantic Chunking & Text Splitters** | `textsplitter.ipynb`, `semantic_chunking.ipynb`, `semanticchunk.ipynb`, `loaders_and_splitters.ipynb` | In Upgrade Pipeline | Pending | Pending | Pending | **Pending** |
| [`MultimodalRAG/`](MultimodalRAG/) | **Multimodal Vision & Document Understanding** | `1-multimodalopenai.ipynb`, sample PDFs | Vision LLMs + Text Extractors | Pending | Pending | Pending | **Pending** |
| [`VectorStoresAndEmbeddings/`](VectorStoresAndEmbeddings/) | **FAISS, ChromaDB & Vector Store Operations** | `faisstest.ipynb`, `charomadb.ipynb`, `docloader.ipynb`, `dataparsingandcleaning.ipynb` | Local Embeddings + Vector DBs | Pending | Pending | Pending | **Pending** |

---

## Detailed Topic Overviews

### 1. Cache-Augmented Generation (`CAG/`)
- **Concept**: Eliminates document retrieval latency for static reference corpora by preloading tokenized text directly into prompt cache.
- **Workflow**: Client Query -> In-Context Cache Extraction -> Single-Pass Groq LLM Generation.
- **Key Files**: 
  - [`cache_augment_generation.ipynb`](CAG/cache_augment_generation.ipynb)
  - [`cache_augment_generation.html`](CAG/cache_augment_generation.html)
  - [`workflow_architecture.png`](CAG/workflow_architecture.png)

### 2. Adaptive RAG (`AdaptiveRAG/`)
- **Concept**: Dynamically routes incoming queries based on intent: conversational chitchat (direct LLM), specialized domain questions (vector store retriever), or broad current events (web search). Includes dual-stage hallucination and answer grading.
- **Workflow**: Query Classifier -> Route Selection -> Retrieval / Direct / Web -> Hallucination Grader -> Answer Grader -> Final Synthesis.
- **Key Files**: 
  - [`adaptive_rag.ipynb`](AdaptiveRAG/adaptive_rag.ipynb)
  - [`adaptive_rag.py`](AdaptiveRAG/adaptive_rag.py)
  - [`adaptive_rag.html`](AdaptiveRAG/adaptive_rag.html)
  - [`workflow_architecture.png`](AdaptiveRAG/workflow_architecture.png)

### 3. Corrective RAG (`CorrectiveRAG/`)
- **Concept**: Uses an LLM-as-a-judge to evaluate retrieved document relevance before answering. Ambiguous or irrelevant context triggers a corrective web search via Tavily to ensure factuality.
- **Workflow**: Retrieve -> Grade Documents -> Conditional Web Fallback -> Context Consolidation -> Generation.
- **Key Files**: 
  - [`corrective_rag.ipynb`](CorrectiveRAG/corrective_rag.ipynb)
  - [`corrective_rag.html`](CorrectiveRAG/corrective_rag.html)
  - [`workflow_architecture.png`](CorrectiveRAG/workflow_architecture.png)

### 4. Agentic RAG Suite (`AgenticRAG/`)
- **Concept**: Autonomous agents capable of planning, tool execution, and self-directed retrieval across heterogeneous knowledge sources.
- **Modules**:
  - `1-basic_agentic_rag.ipynb`: Baseline tool-calling LangGraph agent.
  - `2-ReAct.ipynb`: Multi-source ReAct agent combining FAISS, Wikipedia, ArXiv, proprietary tech docs, and research notes.
  - `desion_maker_rag.ipynb`: Decision-Maker RAG featuring dynamic domain routing, document grading, and query reformulation loops.
  - `ReAct_LangGraph_Studio/`: LangGraph Studio application with streaming agent checkpoints.
- **Key Files**:
  - Interactive notebooks with live outputs and matching `.html` report companions.

### 5. Persistent Memory RAG (`PersistentMemoryRAG/`)
- **Concept**: Multi-turn conversational RAG maintaining state across conversation turns using LangGraph `MemorySaver` checkpointer and `MessagesState`.
- **Workflow**: User Turn -> Stateful Checkpoint Load -> LLM Tool Decision (`retrieve` or answer) -> FAISS Vector Store -> Synthesis -> Stateful Checkpoint Save.
- **Key Files**: 
  - [`ragmemory.ipynb`](PersistentMemoryRAG/ragmemory.ipynb)
  - [`ragmemory.html`](PersistentMemoryRAG/ragmemory.html)
  - [`workflow_architecture.png`](PersistentMemoryRAG/workflow_architecture.png)

### 6. Autonomous RAG (`AutonomousRAG/`)
- **Concept**: Advanced agentic cognitive loops: Self-Reflection, Chain-of-Thought (CoT) retrieval, iterative multi-step query expansion, query decomposition, and answer synthesis.
- **Modules**:
  - `self_reflection.ipynb`: Self-evaluating generation loop that inspects its own answer against retrieved evidence and refines if deficient.
  - `chain_of_thoughts_with_rag.ipynb`: Step-by-step reasoning interleaved with retrieval checkpoints.
  - `iterative_retrieval.ipynb`: Multi-hop iterative question answering with cumulative evidence accumulation.
  - `query_planner_and_decomposition.ipynb`: Breaking down complex multi-faceted queries into sub-questions.
  - `7-answersynthesis.ipynb`: Multi-perspective answer generation and cross-document reconciliation.

### 7. Advanced Retrieval (`AdvancedRetrieval/`)
- **Concept**: State-of-the-art search enhancement techniques to bridge the lexical-semantic gap.
- **Modules**:
  - `hyde.ipynb`: Hypothetical Document Embeddings (HyDE).
  - `query-expansion.ipynb`: Multi-query expansion and query generation.
  - `querydecomposition.ipynb`: Sub-query decomposition.
  - `re-ranking_with_llm.ipynb`: Two-stage retrieval with cross-encoder / LLM re-ranking.
  - `hybrid_search.ipynb`: Reciprocal Rank Fusion (RRF) combining dense vector search and sparse BM25 lexical search.
  - `mmr.ipynb`: Maximal Marginal Relevance for diversity in retrieval.
  - `cosinesimilarity.ipynb`: Deep dive into vector distance metrics.

### 8. Chunking Strategies (`ChunkingStrategies/`)
- **Concept**: Document splitting methodologies optimizing chunk boundary preservation and semantic coherence.
- **Modules**: Recursive character splitting, semantic boundary chunking, token splitters, and document loaders.

### 9. Multimodal RAG (`MultimodalRAG/`)
- **Concept**: Processing, embedding, and reasoning over mixed text, tables, and images from PDFs and documents.

### 10. Vector Stores & Embeddings (`VectorStoresAndEmbeddings/`)
- **Concept**: Deep dive into indexing, persistence, and querying using FAISS, ChromaDB, and HuggingFace embedding models.

---

## Environment Setup & Installation

### 1. Requirements
Ensure Python 3.10+ is installed. Then install dependencies:
```bash
pip install -r requirment.txt
```

### 2. Environment Variables
Create a `.env` file in the project root:
```bash
GROQ_API_KEY=gsk_your_groq_api_key_here
TAVILY_API_KEY=tvly-your_tavily_api_key_here
USER_AGENT=RAGWorkflowApp/1.0
```

### 3. Running Notebooks
Launch Jupyter or execute notebooks directly:
```bash
jupyter notebook
```
Or execute headless:
```bash
jupyter nbconvert --to notebook --execute --inplace <notebook_path>.ipynb
```
