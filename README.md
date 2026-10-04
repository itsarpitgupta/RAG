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
| 14 | [`VectorlessRAG/`](VectorlessRAG/) | **Vectorless RAG & Tree Reasoning** | `PageIndex_Vectorless_RAG_CrashCourse+(1).ipynb` | Groq `gpt-oss-120b` + MiniLM | Yes | Yes | Yes | **Covered** |
| 15 | [`LLMGateways/`](LLMGateways/) | **Enterprise LLM Gateways & Routing** | `llm_gateway_tutorial.ipynb` | Groq `gpt-oss-120b`, `gpt-oss-20b`, `qwen3.8-27b` + MiniLM | Yes | Yes | Yes | **Covered** |
| 16 | [`Guardrails/`](Guardrails/) | **Defense-in-Depth AI Guardrails** | `langchain_guardrails_crash_course.ipynb` | Groq `gpt-oss-120b` + MiniLM Vector Guardrail | Yes | Yes | Yes | **Covered** |
| 17 | [`ChatBotAndRagEvalution/`](ChatBotAndRagEvalution/) | **Chatbot & RAG Triad Evaluation** | `01_chatbot_evaluation.ipynb`, `02_rag_evaluation.ipynb` | Groq `gpt-oss-120b`, `qwen3.8-27b` + MiniLM | Yes (2) | Yes | Yes (2) | **Covered** |
| 18 | [`GraphDB/`](GraphDB/) | **Knowledge Graphs & Neo4j Hybrid RAG** | `01_neo4j_knowledge_graph_construction.ipynb`, `02_neo4j_graph_rag_and_cypher.ipynb` | Groq `gpt-oss-120b` + Neo4j Aura Cloud + MiniLM | Yes (2) | Yes | Yes (2) | **Covered** |
| 19 | [`GraphDBWithLLM/`](GraphDBWithLLM/) | **GraphCypherQAChain & Few-Shot Cypher** | `01_graph_cypher_qa_chain.ipynb`, `02_advanced_cypher_prompt_strategies.ipynb` | Groq `gpt-oss-120b` + Neo4j Aura Cloud | Yes (2) | Yes | Yes (2) | **Covered** |

> **Summary**: All 19 core modules across foundational parsing, advanced indexing, agentic decision-making, autonomous cognitive loops, vectorless tree reasoning, multimodal RAG, enterprise LLM gateways, multi-tier guardrails, LLM-as-a-judge evaluation suites, Neo4j Knowledge Graph RAG, and GraphCypherQAChain question-answering engines have been **100% Covered** with full live execution outputs, custom architecture diagrams, and HTML reports.


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

### 13. Vectorless RAG & Structural Tree Reasoning ([`VectorlessRAG/`](VectorlessRAG/))
- **Concept**: Reasoning-based retrieval over hierarchical document trees that completely eliminates arbitrary text chunking and vector databases for complex, structured documents (10-Ks, annual reports, academic syllabi, contracts):
  - **Hierarchical Indexing**: Parses natural document boundaries (chapters, sections, subtopics, page indexes) without destroying tables or multi-paragraph context.
  - **Groq LLM Tree Search**: Uses `openai/gpt-oss-120b` to reason step-by-step over compressed tree metadata, returning exact section paths and page citations.
  - **Zero-Shot Domain Expertise**: Injects domain routing rules directly into retrieval prompts without expensive embedding fine-tuning.
  - **Comparative Benchmark**: Empirically contrasts Vector RAG (HuggingFace MiniLM + FAISS) vs Vectorless RAG (PageIndex Tree + Groq) on identical queries.
  - **Hybrid Pattern**: 4-stage enterprise pattern using vector filtering for corpus narrowing and tree reasoning for precision synthesis.
- **Artifacts**:
  - Notebook: [`PageIndex_Vectorless_RAG_CrashCourse+(1).ipynb`](VectorlessRAG/PageIndex_Vectorless_RAG_CrashCourse+(1).ipynb)
  - Interactive HTML: [`PageIndex_Vectorless_RAG_CrashCourse+(1).html`](VectorlessRAG/PageIndex_Vectorless_RAG_CrashCourse+(1).html)
  - Architecture Diagram: [`workflow_vectorless_rag.png`](VectorlessRAG/workflow_vectorless_rag.png)
  - Tree Index Cache: [`pageindex_tree_cache.json`](VectorlessRAG/pageindex_tree_cache.json)

---

### 14. Enterprise LLM Gateways ([`LLMGateways/`](LLMGateways/))
- **Concept**: Reverse proxy and traffic management layer decoupling client applications from proprietary provider APIs:
  - **Unified Completion Interface**: Single `completion()` call across heterogeneous LLM providers (Groq `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`).
  - **Zero-Downtime Automatic Fallbacks**: Transparent failover chains rescuing rate-limited or unavailable model deployments.
  - **Automated Cost Tracking**: Real-time per-call token calculation and USD financial accounting via LiteLLM.
  - **Multi-Tier Caching**: Exact in-memory cache + **HuggingFace MiniLM (`sentence-transformers/all-MiniLM-L6-v2`) Semantic Vector Caching** (>0.80 cosine similarity) serving paraphrased queries in <5ms at $0 cost.
  - **Intelligent Routing & Load Balancing**: Router with aliases (`fast-cheap`, `smart-reasoning`), `simple-shuffle`, and `latency-based-routing`.
  - **LangChain Integration**: `ChatLiteLLM` drop-in chat models and `.with_fallbacks()` LCEL chains.
  - **Pre-Call Guardrails Inside Gateway**: Pure Python input callbacks for PII scrubbing (Email, Phone, Cards) and prompt injection blocking.
- **Artifacts**:
  - Notebook: [`llm_gateway_tutorial.ipynb`](LLMGateways/llm_gateway_tutorial.ipynb)
  - Interactive HTML: [`llm_gateway_tutorial.html`](LLMGateways/llm_gateway_tutorial.html)
  - Architecture Diagram: [`workflow_llm_gateways.png`](LLMGateways/workflow_llm_gateways.png)

---

### 15. Defense-in-Depth AI Guardrails ([`Guardrails/`](Guardrails/))
- **Concept**: Multi-layered security mesh protecting agentic workflows against adversarial prompt injections, sensitive data leaks, and unauthorized tool execution:
  - **Layer 1 (Deterministic Filter)**: Zero-cost regex and keyword filters blocking explicit attack terms in <1ms.
  - **Layer 2 (Semantic Vector Guardrail)**: Local HuggingFace MiniLM (`sentence-transformers/all-MiniLM-L6-v2`) dense embedding vectors detecting paraphrased exploits and jailbreaks that bypass keyword lists.
  - **Layer 3 (PII Sanitization)**: Built-in `PIIMiddleware` performing RFC-compliant email redaction, credit card masking, and API key blocking.
  - **Layer 4 (Human-in-the-Loop)**: `HumanInTheLoopMiddleware` with LangGraph `InMemorySaver` checkpointer, pausing execution before high-risk actions (sending emails, purging database tables) and requiring explicit cryptographic resume approvals (`Command(resume={"decisions": [{"type": "approve"}]})`).
  - **Layer 5 (Output Compliance)**: `after_agent` hooks ensuring mandatory regulatory disclaimers and hallucination mitigation.
  - **Production Case Study**: Complete end-to-end multi-tier healthcare chatbot demonstrating safe symptom consultation, PII scrubbing, and appointment booking approvals.
- **Artifacts**:
  - Notebook: [`langchain_guardrails_crash_course.ipynb`](Guardrails/langchain_guardrails_crash_course.ipynb)
  - Interactive HTML: [`langchain_guardrails_crash_course.html`](Guardrails/langchain_guardrails_crash_course.html)
  - Architecture Diagram: [`workflow_guardrails.png`](Guardrails/workflow_guardrails.png)

---

### 16. Chatbot & RAG Triad Evaluation ([`ChatBotAndRagEvalution/`](ChatBotAndRagEvalution/))
- **Concept**: Production-grade evaluation frameworks and LLM-as-a-judge observability pipelines:
  - **Chatbot & Prompt Evaluation** (`01_chatbot_evaluation.ipynb`):
    - Golden test dataset uploading and management via LangSmith SDK.
    - Candidate persona benchmarking (Concise vs Educational/Detailed) using Groq `qwen/qwen3.8-27b`.
    - Structured LLM-as-a-Judge grading with step-by-step reasoning, zero-cost rule-based concision heuristics, and polite tone analysis.
    - Comparative executive scorecard across accuracy pass rate, length compliance, and response latency.
  - **RAG Pipeline & The RAG Triad** (`02_rag_evaluation.ipynb`):
    - Local dense vector retrieval indexed with HuggingFace MiniLM (`sentence-transformers/all-MiniLM-L6-v2`).
    - `@traceable()` LangSmith tracking automatically instrumenting document chunks, retrieved context, and generated answers.
    - The 4 Pillar Evaluators: *Context Retrieval Relevance*, *Groundedness (Faithfulness)*, *Answer Relevance*, and *Factual Ground Truth Correctness*.
    - Zero-Hallucination production scorecard and diagnostic health checks.
- **Artifacts**:
  - Notebooks: [`01_chatbot_evaluation.ipynb`](ChatBotAndRagEvalution/01_chatbot_evaluation.ipynb), [`02_rag_evaluation.ipynb`](ChatBotAndRagEvalution/02_rag_evaluation.ipynb)
  - Interactive HTML: [`01_chatbot_evaluation.html`](ChatBotAndRagEvalution/01_chatbot_evaluation.html), [`02_rag_evaluation.html`](ChatBotAndRagEvalution/02_rag_evaluation.html)
  - Architecture Diagrams: [`workflow_chatbot_evaluation.png`](ChatBotAndRagEvalution/workflow_chatbot_evaluation.png), [`workflow_rag_evaluation.png`](ChatBotAndRagEvalution/workflow_rag_evaluation.png)

---

### 17. Enterprise Knowledge Graphs & Neo4j Hybrid RAG ([`GraphDB/`](GraphDB/))
- **Concept**: Unites graph databases with dense vector search to solve the multi-hop reasoning, causal dependency, and entity-linking bottlenecks of traditional RAG:
  - **Knowledge Graph Construction & Graph Analytics** (`01_neo4j_knowledge_graph_construction.ipynb`):
    - Structured Pydantic extraction of entities (`Entity`) and directional relationships (`Relationship`) using Groq `openai/gpt-oss-120b`.
    - Real-world AI & semiconductor supply chain corpus (ASML, TSMC, NVIDIA, OpenAI, Microsoft, Anthropic, Google DeepMind, Meta AI).
    - Parameterized Cypher `MERGE` ingestion into live **Neo4j Aura Cloud** database with zero duplicates.
    - NetworkX topological analysis: Degree centrality (connectivity) and Betweenness centrality (identifying single-point-of-failure bridge nodes like TSMC and NVIDIA).
    - Shortest path multi-hop discovery (e.g. `ASML` ➔ `TSMC` ➔ `NVIDIA` ➔ `Microsoft` ➔ `OpenAI`) and high-resolution visual graph plotting.
  - **Graph RAG & Text-to-Cypher Engine** (`02_neo4j_graph_rag_and_cypher.ipynb`):
    - Zero-shot Text-to-Cypher query generator translating user questions into valid Cypher queries against Neo4j Aura Cloud.
    - Multi-hop subgraph extractor retrieving 1-hop and 2-hop connected factual triplets.
    - HuggingFace MiniLM (`sentence-transformers/all-MiniLM-L6-v2`) in-memory vector store for dense semantic passage search.
    - **Hybrid Graph + Vector RAG Pipeline**: Fuses structured graph facts with unstructured narrative prose for complete, zero-hallucination synthesis.
    - Head-to-head empirical benchmark: **Pure Vector RAG vs Pure Graph RAG vs Hybrid Graph RAG** across multi-hop supply chain queries.
- **Artifacts**:
  - Notebooks: [`01_neo4j_knowledge_graph_construction.ipynb`](GraphDB/01_neo4j_knowledge_graph_construction.ipynb), [`02_neo4j_graph_rag_and_cypher.ipynb`](GraphDB/02_neo4j_graph_rag_and_cypher.ipynb)
  - Interactive HTML: [`01_neo4j_knowledge_graph_construction.html`](GraphDB/01_neo4j_knowledge_graph_construction.html), [`02_neo4j_graph_rag_and_cypher.html`](GraphDB/02_neo4j_graph_rag_and_cypher.html)
  - Architecture Diagrams: [`workflow_neo4j_knowledge_graph.png`](GraphDB/workflow_neo4j_knowledge_graph.png), [`workflow_graph_rag_and_cypher.png`](GraphDB/workflow_graph_rag_and_cypher.png)

---

### 18. Question Answering over Graph Databases with LLMs ([`GraphDBWithLLM/`](GraphDBWithLLM/))
- **Concept**: Translating natural language questions into executable Cypher queries against Neo4j property graphs using LangChain and Groq:
  - **GraphCypherQAChain Pipeline** (`01_graph_cypher_qa_chain.ipynb`):
    - Ingests heterogeneous real-world datasets into **Neo4j Aura Cloud**: The Movies Knowledge Graph (Movies, Actors, Directors, Genres) and a Social Network Graph (Users, Posts, FRIEND, LIKES).
    - Modern `langchain_neo4j.GraphCypherQAChain` powered by Groq `openai/gpt-oss-120b`.
    - Handles direct entity lookups ("Who directed Casino?"), multi-entity queries ("Who were the actors of Casino?"), numerical aggregations ("How many movies has Tom Hanks acted in?"), multi-hop social queries, and network centrality questions.
    - Transparent intermediate step inspection: extracts and audits the generated Cypher query, raw database records, and synthesized answer.
  - **Advanced Cypher Prompt Strategies & Self-Correction** (`02_advanced_cypher_prompt_strategies.ipynb`):
    - Overcomes the failure modes of zero-shot Cypher generation: schema hallucinations (guessing non-existent labels like `:Artist`), case-sensitivity mismatches, and multi-stage `WITH` aggregations.
    - Dynamically formats exemplar question-Cypher pairs using LangChain's `FewShotPromptTemplate` conditioned on the live Neo4j schema.
    - Automated query syntax validator with a self-correction feedback loop catching database errors and rewriting queries.
    - Empirical benchmark comparing Zero-Shot vs Few-Shot Cypher generation across 4 complex queries with a detailed performance scorecard.
- **Artifacts**:
  - Notebooks: [`01_graph_cypher_qa_chain.ipynb`](GraphDBWithLLM/01_graph_cypher_qa_chain.ipynb), [`02_advanced_cypher_prompt_strategies.ipynb`](GraphDBWithLLM/02_advanced_cypher_prompt_strategies.ipynb)
  - Interactive HTML: [`01_graph_cypher_qa_chain.html`](GraphDBWithLLM/01_graph_cypher_qa_chain.html), [`02_advanced_cypher_prompt_strategies.html`](GraphDBWithLLM/02_advanced_cypher_prompt_strategies.html)
  - Architecture Diagrams: [`workflow_graph_cypher_qa.png`](GraphDBWithLLM/workflow_graph_cypher_qa.png), [`workflow_fewshot_cypher_prompting.png`](GraphDBWithLLM/workflow_fewshot_cypher_prompting.png)

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

# PageIndex API Key (Vectorless RAG Tree Generation)
PAGEINDEX_API_KEY=your_pageindex_api_key_here

# User Agent for Wikipedia & Web Search
USER_AGENT=RAGWorkflowApp/1.0
```

### 3. Running Notebooks & Viewing Reports
- **Interactive UI**: Launch Jupyter Lab or Notebook:
  ```bash
  jupyter lab
  ```
- **HTML Companion Reports**: All upgraded folders contain pre-rendered `.html` reports (e.g. `CAG/cache_augment_generation.html`, `AutonomousRAG/self_reflection.html`, `MultimodalRAG/1-multimodalopenai.html`) that can be opened directly in any browser with zero installation needed.
