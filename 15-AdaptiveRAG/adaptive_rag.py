r"""
Adaptive RAG (Adaptive Retrieval-Augmented Generation) Pipeline.

This module implements the Adaptive RAG pattern using LangGraph, Groq LLM,
and HuggingFace Embeddings.

================================================================================
ADAPTIVE RAG WORKFLOW ARCHITECTURE
================================================================================

                        ┌──────────────────────────────┐
                        │     START: User Question     │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                       /────────────────────────────────\
                      <   ROUTER: route_question (LLM)   >
                       \────────────────────────────────/
                                       │
                ┌──────────────────────┴──────────────────────┐
   [datasource == "vectorstore"]                 [datasource == "web_search"]
                │                                             │
                ▼                                             ▼
     ┌──────────────────────┐                      ┌──────────────────────┐
     │  NODE 1: retrieve    │                      │  NODE 5: web_search  │
     │  (FAISS Vectorstore) │                      │  (Tavily Live API)   │
     └──────────┬───────────┘                      └──────────┬───────────┘
                │                                             │
                ▼                                             │
     ┌──────────────────────┐                                 │
     │NODE 2:grade_documents│                                 │
     │  (Doc Relevance LLM) │                                 │
     └──────────┬───────────┘                                 │
                │                                             │
                ▼                                             │
      /────────────────────\                                  │
     <   decide_to_generate >                                 │
      \────────────────────/                                  │
        │                │                                    │
 [All Irrelevant]  [Relevant Docs]                            │
        │                │                                    │
        ▼                │                                    │
 ┌───────────────┐       │                                    │
 │    NODE 4:    │       │                                    │
 │transform_query│       │                                    │
 └──────┬────────┘       │                                    │
        │ (retry loop)   │                                    │
        └───────► retrieve                                    │
                         │                                    │
                         ▼                                    │
              ┌──────────────────────┐                        │
              │   NODE 3: generate   │ ◄──────────────────────┘
              │  (Grounded RAG LLM)  │
              └──────────┬───────────┘
                         │
                         ▼
             /────────────────────────\
            <   Hallucination Grader   > ──[Not Grounded]──► (Retry NODE 3: generate)
             \────────────────────────/
                         │ [Grounded: "yes"]
                         ▼
             /────────────────────────\
            <      Answer Grader       > ──[Not Useful]───► (Loop to NODE 4: transform_query)
             \────────────────────────/
                         │ [Addresses Question: "yes"]
                         ▼
                        ┌─────┐
                        │ END │
                        └─────┘

================================================================================
CELL & WORKFLOW MAPPING DIRECTORY
================================================================================
  [CELL 1]  Environment Setup & Dependencies
  [CELL 2]  Shared Graph State (GraphState)
  ──────────────────────────────────────────────────────────────────────────────
  LANGGRAPH NODES (in execution order):
  [CELL 3]  NODE 1: retrieve                --> Workflow: FAISS vector retrieval
  [CELL 4]  NODE 2: grade_documents         --> Workflow: Filter irrelevant docs
  [CELL 5]  NODE 3: generate                --> Workflow: Synthesize grounded QA
  [CELL 6]  NODE 4: transform_query         --> Workflow: Rewrite query for retry
  [CELL 7]  NODE 5: web_search              --> Workflow: Live internet search
  ──────────────────────────────────────────────────────────────────────────────
  CONDITIONAL EDGES (routing decisions):
  [CELL 8]  EDGE 1: route_question          --> Workflow: START -> vector vs web
  [CELL 9]  EDGE 2: decide_to_generate      --> Workflow: After grading -> gen vs rewrite
  [CELL 10] EDGE 3: dual_grade_evaluator    --> Workflow: Post-gen hallucination & utility
  ──────────────────────────────────────────────────────────────────────────────
  GRAPH COMPILATION:
  [CELL 11] Workflow Graph Builder          --> Workflow: LangGraph StateGraph assembly
  ──────────────────────────────────────────────────────────────────────────────
  REUSABLE SUB-METHODS (called by nodes & edges):
  [CELL 12] Sub-Methods: LLM & Doc Helper   --> Shared Groq LLM & text formatter
  [CELL 13] Sub-Methods: Vectorstore Index  --> HuggingFace Embeddings & FAISS index
  [CELL 14] Sub-Methods: Question Router    --> Structured Pydantic query classifier
  [CELL 15] Sub-Methods: Retrieval Grader   --> Structured Pydantic doc relevance check
  [CELL 16] Sub-Methods: Grounded Generator --> RAG prompt & Groq text generator
  [CELL 17] Sub-Methods: Hallucination Check--> Structured Pydantic factuality check
  [CELL 18] Sub-Methods: Answer Grader      --> Structured Pydantic utility check
  [CELL 19] Sub-Methods: Query Rewriter     --> LLM semantic query reformulation
  [CELL 20] Sub-Methods: Web Search Tool    --> Tavily Search API wrapper
  ──────────────────────────────────────────────────────────────────────────────
  MAIN EXECUTION:
  [CELL 21] Main Execution Demo            --> End-to-end multi-path test runs
================================================================================
"""

# %% [CELL 1: ENVIRONMENT SETUP & IMPORTS]
# ==============================================================================
# WORKFLOW MAP:
#   [Prerequisites] -> Prepares API keys, embeddings, tools, and models for all steps.
# ==============================================================================

import os
import sys
from typing import List, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

# Ensure environment variables are loaded and USER_AGENT is configured
load_dotenv()
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "AdaptiveRAG/1.0")

from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START


# %% [CELL 2: SHARED GRAPH STATE]
# ==============================================================================
# WORKFLOW MAP:
#   [Shared State Bus] Passed across every Node and Conditional Edge in the workflow:
#   ┌──────────────────────────────────────────────────────────────────────────┐
#   │ GraphState: question (str) | generation (str) | documents (List[Doc])   │
#   └──────────────────────────────────────────────────────────────────────────┘
# ==============================================================================

class GraphState(TypedDict):
    """
    Represents the shared state of our Adaptive RAG graph.

    Attributes:
        question: The user's input question (or reformulated query).
        generation: The synthesized answer generated by the LLM.
        documents: List of retrieved or web-searched document chunks.
    """

    question: str
    generation: str
    documents: List[Document]


# ==============================================================================
# SECTION 2: LANGGRAPH NODES (IN PIPELINE ORDER)
# ==============================================================================

# %% [CELL 3: NODE 1 - RETRIEVE]
# ==============================================================================
# WORKFLOW MAP:
#   [START]
#      │
#      ▼
#   [Question Router: datasource == "vectorstore"]
#      │
#      ▼
#   >>> [NODE 1: retrieve] <<<  (CURRENT NODE)
#      │
#      ▼
#   [NODE 2: grade_documents]
#
#   * Note: Also receives re-routed queries looping back from [NODE 4: transform_query]
#
# Inbound From : route_question (EDGE 1) OR transform_query (NODE 4 retry loop)
# Outbound To  : grade_documents (NODE 2)
# Delegated To : Sub-method `retrieve_documents()` [Cell 13]
# ==============================================================================

def retrieve(state: GraphState) -> dict:
    """
    Node 1: Retrieve documents from the local FAISS vector store.

    Delegates to sub-method: `retrieve_documents(query)`
    """
    print("\n--- [NODE 1: RETRIEVE] Fetching documents from vector store ---")
    question = state["question"]
    documents = retrieve_documents(question)
    return {"documents": documents, "question": question}


# %% [CELL 4: NODE 2 - GRADE_DOCUMENTS]
# ==============================================================================
# WORKFLOW MAP:
#   [NODE 1: retrieve]
#          │
#          ▼
#   >>> [NODE 2: grade_documents] <<<  (CURRENT NODE)
#          │
#          ▼
#   /────────────────────\
#  <  decide_to_generate  >  (EDGE 2: Any docs relevant?)
#   \────────────────────/
#     │                │
#  [Relevant Docs]   [All Irrelevant]
#     │                │
#     ▼                ▼
#  [NODE 3: gen]     [NODE 4: transform_query]
#
# Inbound From : retrieve (NODE 1)
# Outbound To  : decide_to_generate (EDGE 2)
# Delegated To : Sub-method `grade_document_relevance()` [Cell 15]
# ==============================================================================

def grade_documents(state: GraphState) -> dict:
    """
    Node 2: Evaluate candidate documents for relevance to the question.
    Filters out irrelevant chunks to prevent context pollution.

    Delegates to sub-method: `grade_document_relevance(question, doc_content)`
    """
    print("\n--- [NODE 2: GRADE_DOCUMENTS] Checking document relevance ---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    for d in documents:
        is_relevant = grade_document_relevance(question, d.page_content)
        if is_relevant:
            print("  -> GRADE: DOCUMENT RELEVANT")
            filtered_docs.append(d)
        else:
            print("  -> GRADE: DOCUMENT NOT RELEVANT (Filtered out)")

    return {"documents": filtered_docs, "question": question}


# %% [CELL 5: NODE 3 - GENERATE]
# ==============================================================================
# WORKFLOW MAP:
#   [NODE 5: web_search] ───┐
#                           ▼
#   [EDGE 2: Relevant Docs] ──► >>> [NODE 3: generate] <<<  (CURRENT NODE)
#                                      │
#                                      ▼
#                          /────────────────────────\
#                         <   Hallucination Grader   >  (EDGE 3)
#                          \────────────────────────/
#
# Inbound From : decide_to_generate (EDGE 2) OR web_search (NODE 5) OR self (retry)
# Outbound To  : grade_generation_v_documents_and_question (EDGE 3)
# Delegated To : Sub-method `generate_grounded_answer()` [Cell 16]
# ==============================================================================

def generate(state: GraphState) -> dict:
    """
    Node 3: Synthesize a concise, grounded answer using retrieved context.

    Delegates to sub-method: `generate_grounded_answer(question, documents)`
    """
    print("\n--- [NODE 3: GENERATE] Synthesizing grounded answer ---")
    question = state["question"]
    documents = state["documents"]

    generation = generate_grounded_answer(question, documents)
    return {"documents": documents, "question": question, "generation": generation}


# %% [CELL 6: NODE 4 - TRANSFORM_QUERY]
# ==============================================================================
# WORKFLOW MAP:
#   [EDGE 2: All Irrelevant] ──┐
#                              ▼
#   [EDGE 3: Answer Not Useful]─► >>> [NODE 4: transform_query] <<<  (CURRENT NODE)
#                                       │
#                                       ▼ (Retry Loop)
#                              [NODE 1: retrieve]
#
# Inbound From : decide_to_generate (EDGE 2) OR Answer Grader (EDGE 3)
# Outbound To  : retrieve (NODE 1) -> triggers re-retrieval with improved query
# Delegated To : Sub-method `rewrite_query_for_vectorstore()` [Cell 19]
# ==============================================================================

def transform_query(state: GraphState) -> dict:
    """
    Node 4: Rewrite the question to better optimize vectorstore retrieval.

    Delegates to sub-method: `rewrite_query_for_vectorstore(question)`
    """
    print("\n--- [NODE 4: TRANSFORM_QUERY] Reformulating question for retrieval ---")
    question = state["question"]
    documents = state.get("documents", [])

    better_question = rewrite_query_for_vectorstore(question)
    print(f"  -> Original Query : {question}")
    print(f"  -> Rewritten Query: {better_question}")
    return {"documents": documents, "question": better_question}


# %% [CELL 7: NODE 5 - WEB_SEARCH]
# ==============================================================================
# WORKFLOW MAP:
#   [START: User Question]
#          │
#          ▼
#   [Question Router: datasource == "web_search"]
#          │
#          ▼
#   >>> [NODE 5: web_search] <<<  (CURRENT NODE)
#          │
#          ▼
#   [NODE 3: generate]
#
# Inbound From : route_question (EDGE 1)
# Outbound To  : generate (NODE 3)
# Delegated To : Sub-method `perform_web_search()` [Cell 20]
# ==============================================================================

def web_search(state: GraphState) -> dict:
    """
    Node 5: Execute web search for general queries outside the local corpus.

    Delegates to sub-method: `perform_web_search(query)`
    """
    print("\n--- [NODE 5: WEB_SEARCH] Executing live Tavily web search ---")
    question = state["question"]

    web_doc = perform_web_search(question)
    return {"documents": [web_doc], "question": question}


# ==============================================================================
# SECTION 3: CONDITIONAL ROUTING EDGES
# ==============================================================================

# %% [CELL 8: CONDITIONAL EDGE 1 - ROUTE_QUESTION]
# ==============================================================================
# WORKFLOW MAP:
#   [START: User Question]
#          │
#          ▼
#   >>> < ROUTER: route_question > <<<  (CURRENT CONDITIONAL EDGE)
#          │
#     ┌────┴───────────────────────────┐
#     ▼                                ▼
#   [NODE 1: retrieve]            [NODE 5: web_search]
#   ("vectorstore")               ("web_search")
#
# Purpose     : Assesses question domain at graph entry.
# Outbound To : 'retrieve' (NODE 1) OR 'web_search' (NODE 5)
# Delegated To: Sub-method `classify_query_source()` [Cell 14]
# ==============================================================================

def route_question(state: GraphState) -> str:
    """
    Conditional Edge 1 (at START):
    Routes the query to either the vectorstore retrieval node or web search node.

    Delegates to sub-method: `classify_query_source(question)`

    Returns:
        "vectorstore" -> routes to 'retrieve'
        "web_search"  -> routes to 'web_search'
    """
    print("\n--- [EDGE 1: ROUTE_QUESTION] Determining data source ---")
    question = state["question"]
    datasource = classify_query_source(question)

    if datasource == "web_search":
        print("  -> DECISION: Route question to WEB SEARCH")
        return "web_search"
    else:
        print("  -> DECISION: Route question to VECTORSTORE RETRIEVAL")
        return "vectorstore"


# %% [CELL 9: CONDITIONAL EDGE 2 - DECIDE_TO_GENERATE]
# ==============================================================================
# WORKFLOW MAP:
#   [NODE 2: grade_documents]
#          │
#          ▼
#   >>> < DECISION: decide_to_generate > <<<  (CURRENT CONDITIONAL EDGE)
#          │
#     ┌────┴───────────────────────────┐
#     ▼                                ▼
#   [NODE 3: generate]            [NODE 4: transform_query]
#   (Relevant docs found)         (All docs irrelevant)
#
# Purpose     : Prevents empty or completely irrelevant context from reaching generation.
# Outbound To : 'generate' (NODE 3) OR 'transform_query' (NODE 4)
# ==============================================================================

def decide_to_generate(state: GraphState) -> str:
    """
    Conditional Edge 2 (after grade_documents):
    Determines whether to proceed to generation or transform query for re-retrieval.

    Returns:
        "generate"        -> if at least one relevant document chunk remains.
        "transform_query" -> if all document chunks were deemed irrelevant.
    """
    print("\n--- [EDGE 2: DECIDE_TO_GENERATE] Evaluating document relevance sufficiency ---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        print("  -> DECISION: All documents irrelevant. Route to TRANSFORM_QUERY.")
        return "transform_query"
    else:
        print("  -> DECISION: Relevant documents available. Route to GENERATE.")
        return "generate"


# %% [CELL 10: CONDITIONAL EDGE 3 - DUAL GRADE EVALUATOR]
# ==============================================================================
# WORKFLOW MAP:
#   [NODE 3: generate]
#          │
#          ▼
#   >>> < Hallucination Grader > ──[Not Grounded]──► (Retry NODE 3: generate)
#          │ [Grounded: "yes"]
#          ▼
#   >>> < Answer Grader > ─────────[Not Useful]────► (Loop to NODE 4: transform_query)
#          │ [Useful: "yes"]
#          ▼
#        [END]
#
# Purpose     : Dual-stage guardrail:
#               1. Grounds answer against facts (anti-hallucination)
#               2. Enforces question resolution (utility check)
# Outbound To : END (done), 'generate' (retry synthesis), OR 'transform_query' (re-retrieve)
# Delegated To: `check_groundedness_hallucination()` [Cell 17], `check_answer_addresses_question()` [Cell 18]
# ==============================================================================

def grade_generation_v_documents_and_question(state: GraphState) -> str:
    """
    Conditional Edge 3 (after generate):
    Performs dual validation:
      1. Hallucination check: Is generation grounded in retrieved facts?
      2. Answer utility check: Does generation directly address the question?

    Delegates to sub-methods:
      - `check_groundedness_hallucination(documents, generation)`
      - `check_answer_addresses_question(question, generation)`

    Returns:
        "useful"        -> Grounded and answers question -> END
        "not supported" -> Hallucinated / not grounded -> retry 'generate'
        "not useful"    -> Grounded but fails question -> 'transform_query'
    """
    print("\n--- [EDGE 3: DUAL_GRADE] Checking Hallucinations & Answer Utility ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # 1. Hallucination evaluation
    is_grounded = check_groundedness_hallucination(documents, generation)
    if not is_grounded:
        print("  -> DECISION: Generation NOT grounded in documents. Re-trying GENERATE.")
        return "not supported"

    print("  -> DECISION: Generation IS grounded in documents.")

    # 2. Answer utility evaluation
    is_useful = check_answer_addresses_question(question, generation)
    if is_useful:
        print("  -> DECISION: Generation ADDRESSES the question. Route to END.")
        return "useful"
    else:
        print("  -> DECISION: Generation DOES NOT address question. Route to TRANSFORM_QUERY.")
        return "not useful"


# ==============================================================================
# SECTION 4: GRAPH ASSEMBLY & COMPILATION
# ==============================================================================

# %% [CELL 11: WORKFLOW GRAPH BUILDER]
# ==============================================================================
# WORKFLOW MAP:
#   Connects all Nodes (Cells 3-7) and Conditional Edges (Cells 8-10) into a compiled
#   executable StateGraph runtime matching the master architecture diagram.
# ==============================================================================

def build_adaptive_rag_graph():
    """
    Assembles and compiles the full Adaptive RAG StateGraph.

    Returns:
        CompiledStateGraph: The runnable application.
    """
    workflow = StateGraph(GraphState)

    # 1. Register all 5 nodes
    workflow.add_node("web_search", web_search)            # Node 5
    workflow.add_node("retrieve", retrieve)                # Node 1
    workflow.add_node("grade_documents", grade_documents)  # Node 2
    workflow.add_node("generate", generate)                # Node 3
    workflow.add_node("transform_query", transform_query)  # Node 4

    # 2. Add entry conditional edge: START -> route_question (EDGE 1)
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        },
    )

    # 3. Add path from web_search to generate
    workflow.add_edge("web_search", "generate")

    # 4. Add path from retrieve to grade_documents
    workflow.add_edge("retrieve", "grade_documents")

    # 5. Add post-grading conditional edge: grade_documents -> decide_to_generate (EDGE 2)
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )

    # 6. Add loop from transform_query back to retrieve (Retry Loop)
    workflow.add_edge("transform_query", "retrieve")

    # 7. Add post-generation dual-validation conditional edge (EDGE 3)
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    return workflow.compile()


# ==============================================================================
# SECTION 5: REUSABLE TASK SUB-METHODS & LLM CHAINS
# ==============================================================================

# Global caches for shared resources
_RETRIEVER_INSTANCE = None
_LLM_INSTANCE = None
_WEB_SEARCH_TOOL = None


# %% [CELL 12: SUB-METHODS - LLM & DOC FORMATTING]
# ==============================================================================
# WORKFLOW MAP:
#   Shared utility powering NODE 2, NODE 3, NODE 4, EDGE 1, and EDGE 3.
#   Initializes Groq 'openai/gpt-oss-120b' LLM and handles document text concatenation.
# ==============================================================================

def get_llm():
    """
    Sub-method: Lazily initializes and returns the primary Groq LLM.
    Uses 'groq:openai/gpt-oss-120b' via init_chat_model.
    """
    global _LLM_INSTANCE
    if _LLM_INSTANCE is None:
        _LLM_INSTANCE = init_chat_model(model="groq:openai/gpt-oss-120b")
    return _LLM_INSTANCE


def format_docs(docs) -> str:
    """
    Sub-method: Formats a list of Document objects or strings into a clean text block.
    Used by NODE 3 (generate) and EDGE 3 (hallucination grader).
    """
    if isinstance(docs, str):
        return docs
    if hasattr(docs, "page_content"):
        return docs.page_content
    return "\n\n".join(
        doc.page_content if hasattr(doc, "page_content") else str(doc)
        for doc in docs
    )


# %% [CELL 13: SUB-METHODS - VECTORSTORE & RETRIEVAL]
# ==============================================================================
# WORKFLOW MAP:
#   Powers [NODE 1: retrieve]
#   - Embeddings: HuggingFace sentence-transformers/all-MiniLM-L6-v2 (local 384-d)
#   - Vectorstore: FAISS index over Lilian Weng blog posts
# ==============================================================================

def build_vectorstore_retriever():
    """
    Sub-method: Builds FAISS index from authoritative URLs and returns a retriever.
    Uses HuggingFace 'sentence-transformers/all-MiniLM-L6-v2' embeddings.
    """
    print("Initializing HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    print(f"Loading {len(urls)} source articles...")
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print(f"Indexing {len(doc_splits)} chunks into FAISS...")

    vectorstore = FAISS.from_documents(documents=doc_splits, embedding=embedding)
    return vectorstore.as_retriever()


def retrieve_documents(query: str) -> List[Document]:
    """
    Sub-method: Queries the FAISS retriever for candidate documents.
    Directly invoked by [NODE 1: retrieve].
    """
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        _RETRIEVER_INSTANCE = build_vectorstore_retriever()
    return _RETRIEVER_INSTANCE.invoke(query)


# %% [CELL 14: SUB-METHODS - ROUTER CHAIN]
# ==============================================================================
# WORKFLOW MAP:
#   Powers [EDGE 1: route_question]
#   - Pydantic schema RouteQuery enforces Literal["vectorstore", "web_search"]
#   - Groq structured LLM evaluates domain vs broad intent
# ==============================================================================

class RouteQuery(BaseModel):
    """Pydantic schema to route user queries to the optimal datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question, choose to route it to web search or vectorstore.",
    )


def classify_query_source(question: str) -> str:
    """
    Sub-method: Analyzes the question and returns 'vectorstore' or 'web_search'.
    Directly invoked by [EDGE 1: route_question].
    """
    llm = get_llm()
    structured_router = llm.with_structured_output(RouteQuery)

    system_router = (
        "You are an expert at routing a user question to a vectorstore or web search.\n"
        "The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks on LLMs.\n"
        "Use the vectorstore for questions on these topics. For all other questions, use web_search."
    )
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_router),
            ("human", "{question}"),
        ]
    )
    router_chain = route_prompt | structured_router
    result: RouteQuery = router_chain.invoke({"question": question})
    return result.datasource


# %% [CELL 15: SUB-METHODS - RETRIEVAL GRADER]
# ==============================================================================
# WORKFLOW MAP:
#   Powers [NODE 2: grade_documents]
#   - Evaluates each retrieved chunk for semantic relevance
#   - Pydantic schema GradeDocuments enforces binary_score: 'yes' | 'no'
# ==============================================================================

class GradeDocuments(BaseModel):
    """Pydantic schema for binary relevance grading."""
    binary_score: str = Field(
        description="Document relevance to the question: 'yes' or 'no'"
    )


def grade_document_relevance(question: str, document_text: str) -> bool:
    """
    Sub-method: Assesses if a single document chunk is semantically relevant to the question.
    Directly invoked by [NODE 2: grade_documents].
    """
    llm = get_llm()
    structured_grader = llm.with_structured_output(GradeDocuments)

    system_grader = (
        "You are a grader assessing relevance of a retrieved document to a user question.\n"
        "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.\n"
        "Give a binary score 'yes' or 'no' to indicate whether the document is relevant."
    )
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_grader),
            ("human", "Retrieved document:\n\n {document}\n\nUser question: {question}"),
        ]
    )
    grader_chain = grade_prompt | structured_grader
    result: GradeDocuments = grader_chain.invoke(
        {"question": question, "document": document_text}
    )
    return result.binary_score.lower() == "yes"


# %% [CELL 16: SUB-METHODS - RAG GENERATOR CHAIN]
# ==============================================================================
# WORKFLOW MAP:
#   Powers [NODE 3: generate]
#   - Synthesizes grounded answer strictly adhering to retrieved context chunks
# ==============================================================================

def generate_grounded_answer(question: str, documents: List[Document]) -> str:
    """
    Sub-method: Synthesizes a grounded answer using the RAG prompt and Groq LLM.
    Directly invoked by [NODE 3: generate].
    """
    llm = get_llm()
    rag_prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""
    )
    chain = rag_prompt | llm | StrOutputParser()
    return chain.invoke({"context": format_docs(documents), "question": question})


# %% [CELL 17: SUB-METHODS - HALLUCINATION GRADER]
# ==============================================================================
# WORKFLOW MAP:
#   Powers Stage 1 of [EDGE 3: grade_generation_v_documents_and_question]
#   - Pydantic schema GradeHallucinations enforces binary_score: 'yes' | 'no'
#   - Checks that every statement in generation is grounded in facts
# ==============================================================================

class GradeHallucinations(BaseModel):
    """Pydantic schema for groundedness / hallucination detection."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


def check_groundedness_hallucination(documents: List[Document], generation: str) -> bool:
    """
    Sub-method: Verifies that the LLM generation does not invent unsupported facts.
    Directly invoked by [EDGE 3: dual_grade_evaluator].
    """
    llm = get_llm()
    structured_grader = llm.with_structured_output(GradeHallucinations)

    system_hallucination = (
        "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.\n"
        "Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in / supported by the set of facts."
    )
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_hallucination),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
    chain = hallucination_prompt | structured_grader
    result: GradeHallucinations = chain.invoke(
        {"documents": format_docs(documents), "generation": generation}
    )
    return result.binary_score.lower() == "yes"


# %% [CELL 18: SUB-METHODS - ANSWER GRADER]
# ==============================================================================
# WORKFLOW MAP:
#   Powers Stage 2 of [EDGE 3: grade_generation_v_documents_and_question]
#   - Pydantic schema GradeAnswer enforces binary_score: 'yes' | 'no'
#   - Verifies whether the answer actually resolves the user's specific question
# ==============================================================================

class GradeAnswer(BaseModel):
    """Pydantic schema for answer relevance check."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


def check_answer_addresses_question(question: str, generation: str) -> bool:
    """
    Sub-method: Assesses whether the generation directly addresses the user's question.
    Directly invoked by [EDGE 3: dual_grade_evaluator].
    """
    llm = get_llm()
    structured_grader = llm.with_structured_output(GradeAnswer)

    system_answer = (
        "You are a grader assessing whether an answer addresses / resolves a question.\n"
        "Give a binary score 'yes' or 'no'. 'yes' means that the answer resolves the question."
    )
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_answer),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )
    chain = answer_prompt | structured_grader
    result: GradeAnswer = chain.invoke(
        {"question": question, "generation": generation}
    )
    return result.binary_score.lower() == "yes"


# %% [CELL 19: SUB-METHODS - QUERY RE-WRITER]
# ==============================================================================
# WORKFLOW MAP:
#   Powers [NODE 4: transform_query]
#   - Converts the question to an improved semantic formulation for vector retrieval
# ==============================================================================

def rewrite_query_for_vectorstore(question: str) -> str:
    """
    Sub-method: Reformulates a user query to enhance semantic retrieval precision.
    Directly invoked by [NODE 4: transform_query].
    """
    llm = get_llm()
    system_rewriter = (
        "You are a question re-writer that converts an input question to a better version that is optimized\n"
        "for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."
    )
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_rewriter),
            (
                "human",
                "Here is the initial question:\n\n {question} \n Formulate an improved question.",
            ),
        ]
    )
    chain = re_write_prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})


# %% [CELL 20: SUB-METHODS - TAVILY WEB SEARCH]
# ==============================================================================
# WORKFLOW MAP:
#   Powers [NODE 5: web_search]
#   - Queries Tavily live search API when Router flags an out-of-domain query
# ==============================================================================

def perform_web_search(query: str) -> Document:
    """
    Sub-method: Queries Tavily web search and wraps top results into a Document.
    Directly invoked by [NODE 5: web_search].
    """
    global _WEB_SEARCH_TOOL
    if _WEB_SEARCH_TOOL is None:
        _WEB_SEARCH_TOOL = TavilySearchResults(k=3)

    search_results = _WEB_SEARCH_TOOL.invoke({"query": query})
    web_content = "\n".join([r["content"] for r in search_results])
    return Document(page_content=web_content)


# ==============================================================================
# SECTION 6: MAIN EXECUTION DEMO
# ==============================================================================

# %% [CELL 21: MAIN EXECUTION DEMO]
# ==============================================================================
# WORKFLOW MAP:
#   Executes the two primary workflow paths through the compiled graph:
#   1. Out-of-Domain Path: START -> Router [web] -> web_search -> generate -> Graders -> END
#   2. In-Domain Path    : START -> Router [vector] -> retrieve -> grade -> generate -> Graders -> END
# ==============================================================================

if __name__ == "__main__":
    # Configure UTF-8 encoding for console printing on Windows
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    print("=" * 80)
    print("ADAPTIVE RAG PIPELINE DEMO")
    print("=" * 80)

    # 1. Compile the graph
    print("\nBuilding and compiling Adaptive RAG workflow graph...")
    app = build_adaptive_rag_graph()
    print("LangGraph workflow compiled successfully!\n")

    # 2. Test Case 1: Out-of-domain query (Routes to Web Search)
    print("=" * 80)
    print("TEST CASE 1: Query outside vectorstore domain -> Web Search")
    print("  -> Workflow Path: START -> route_question -> web_search -> generate -> dual_grade -> END")
    print("=" * 80)
    query_1 = "What is machine learning"
    result_1 = app.invoke({"question": query_1})

    print("\n" + "=" * 80)
    print(f"FINAL RESULT FOR: '{query_1}'")
    print("=" * 80)
    print(result_1["generation"])

    # 3. Test Case 2: In-domain query (Routes to Vectorstore Retrieval)
    print("\n" + "=" * 80)
    print("TEST CASE 2: Query inside vectorstore domain -> FAISS Retrieval")
    print("  -> Workflow Path: START -> route_question -> retrieve -> grade_documents -> generate -> dual_grade -> END")
    print("=" * 80)
    query_2 = "What is agent memory"
    result_2 = app.invoke({"question": query_2})

    print("\n" + "=" * 80)
    print(f"FINAL RESULT FOR: '{query_2}'")
    print("=" * 80)
    print(result_2["generation"])
    print("=" * 80)
