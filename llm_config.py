"""
Centralized LLM Configuration & Multi-Model Fallback Manager
============================================================
Provides a unified, configurable interface for initializing LLMs across
all notebooks and scripts in the workspace.

Key Features:
- Single source of truth configured via .env (e.g. LLM_MODEL)
- Curated list of verified free models with rate-limit tracking
- Automatic zero-downtime failover (LangChain .with_fallbacks)
- Seamless tool binding for agentic workflows
"""

import os
import sys
from typing import Any, Dict, List, Optional
from dotenv import find_dotenv, load_dotenv

# Ensure stdout/stderr handle UTF-8 safely on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure environment variables are loaded from the root .env
load_dotenv(find_dotenv(usecwd=True))

# ---------------------------------------------------------------------------
# Curated Free Models Catalog
# ---------------------------------------------------------------------------
FREE_MODELS: Dict[str, Dict[str, Any]] = {
    "openai/gpt-oss-120b": {
        "provider": "groq",
        "description": "Flagship 120B open-weights model. Best reasoning, agent architectures, and complex tool calls.",
        "context_window": 131072,
        "recommended_for": "Agents, Complex Reasoning, Production Tasks",
        "supports_tools": True,
    },
    "qwen/qwen3.8-27b": {
        "provider": "groq",
        "description": "High-efficiency 27B model. Fast, balanced latency and strong function calling.",
        "context_window": 131072,
        "recommended_for": "General RAG, Fast Tool Calling, Structured Output",
        "supports_tools": True,
    },
    "openai/gpt-oss-20b": {
        "provider": "groq",
        "description": "Ultra-fast 20B model. Low rate-limit consumption and minimal latency.",
        "context_window": 131072,
        "recommended_for": "High-throughput tasks, Semantic Chunking, Simple Chains",
        "supports_tools": True,
    },
    "allam-2-7b": {
        "provider": "groq",
        "description": "Compact 7B model. Ultra-fast responses with low resource usage.",
        "context_window": 4096,
        "recommended_for": "Emergency fallback, lightweight classification",
        "supports_tools": True,
    },
    "gemini-3.8-flash": {
        "provider": "google_genai",
        "description": "Google Gemini 3.8 Flash. Large 1M token context window via Google API Key.",
        "context_window": 1048576,
        "recommended_for": "Long context retrieval, Multimodal tasks",
        "supports_tools": True,
    },
}

# Default settings from .env with robust fallbacks
DEFAULT_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
DEFAULT_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
DEFAULT_ENABLE_FALLBACKS: bool = os.getenv("LLM_ENABLE_FALLBACKS", "true").lower() in ("true", "1", "yes")

# Default fallback hierarchy: ordered chain when a model hits rate-limits (HTTP 429)
DEFAULT_FALLBACK_ORDER: List[str] = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
]


def _instantiate_model(
    model_name: str,
    temperature: float = 0.7,
    **kwargs: Any,
) -> Any:
    """Helper to instantiate a specific provider model."""
    meta = FREE_MODELS.get(model_name, {})
    provider = meta.get("provider", "groq")

    if provider == "groq" or "/" in model_name or "groq" in model_name:
        from langchain_groq import ChatGroq

        clean_model = model_name.replace("groq:", "")
        api_key = os.getenv("GROQ_API_KEY")
        return ChatGroq(
            model=clean_model,
            temperature=temperature,
            groq_api_key=api_key,
            **kwargs,
        )

    elif provider == "google_genai" or "gemini" in model_name:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
            **kwargs,
        )

    elif provider == "openai" or "gpt-4" in model_name:
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            **kwargs,
        )

    else:
        # Default fallback to ChatGroq
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model_name,
            temperature=temperature,
            **kwargs,
        )


def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    with_fallbacks: Optional[bool] = None,
    tools: Optional[List[Any]] = None,
    custom_fallbacks: Optional[List[str]] = None,
    **kwargs: Any,
) -> Any:
    """
    Get a configured LLM with automatic multi-model fallback capability.

    Parameters:
    -----------
    model : str, optional
        Model ID (e.g., 'openai/gpt-oss-120b', 'qwen/qwen3.8-27b').
        Defaults to LLM_MODEL in .env or 'openai/gpt-oss-120b'.
    temperature : float, optional
        Sampling temperature (0.0 to 1.0). Defaults to 0.7.
    with_fallbacks : bool, optional
        If True, automatically wraps the model with LangChain fallbacks
        to seamlessly survive rate-limits (HTTP 429) and outages.
        Defaults to True.
    tools : list, optional
        List of LangChain tools to bind across primary and fallback models.
    custom_fallbacks : list of str, optional
        Override the default list of fallback model IDs.

    Returns:
    --------
    BaseChatModel or Runnable
        A runnable chat model instance ready for invoke(), ainvoke(), or agents.
    """
    primary_model_name = model or DEFAULT_MODEL
    # Strip any erroneous prefix if present
    primary_model_name = primary_model_name.replace("groq:", "")

    temp = temperature if temperature is not None else DEFAULT_TEMPERATURE
    enable_fallbacks = with_fallbacks if with_fallbacks is not None else DEFAULT_ENABLE_FALLBACKS

    # Instantiate primary model
    primary_llm = _instantiate_model(primary_model_name, temperature=temp, **kwargs)

    if tools:
        primary_llm = primary_llm.bind_tools(tools)

    if not enable_fallbacks:
        return primary_llm

    # Determine fallback model list (excluding the primary model)
    fallback_names = custom_fallbacks or [
        m for m in DEFAULT_FALLBACK_ORDER if m != primary_model_name
    ]

    fallback_models = []
    for fb_name in fallback_names:
        try:
            fb_llm = _instantiate_model(fb_name, temperature=temp, **kwargs)
            if tools:
                fb_llm = fb_llm.bind_tools(tools)
            fallback_models.append(fb_llm)
        except Exception:
            # Skip if an optional fallback provider lacks an API key
            continue

    if fallback_models:
        return primary_llm.with_fallbacks(fallback_models)

    return primary_llm


def list_free_models() -> None:
    """Print an easy-to-read summary of verified free models."""
    print("=" * 80)
    print("[*] VERIFIED FREE & ACTIVE MODELS REGISTRY")
    print("=" * 80)
    current_default = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    print(f">> Current Active Model in .env: {current_default}\n")

    for model_id, info in FREE_MODELS.items():
        is_active = " [ACTIVE]" if model_id == current_default else ""
        print(f"* Model: {model_id}{is_active}")
        print(f"  - Provider:    {info['provider'].upper()}")
        print(f"  - Context:     {info['context_window']:,} tokens")
        print(f"  - Best For:    {info['recommended_for']}")
        print(f"  - Description: {info['description']}")
        print("-" * 80)

    print("\nHow to switch models:")
    print("  1. In your .env file: set LLM_MODEL=\"qwen/qwen3.8-27b\"")
    print("  2. In Python/Notebook: get_llm(model=\"qwen/qwen3.8-27b\")")
    print("  3. Fallbacks are automatic: If your model hits 429 rate limit, it auto-switches!\n")


def visualize_graph(graph: Any, title: Optional[str] = None) -> None:
    """
    Renders and displays a compiled LangGraph workflow.
    - In Jupyter Notebooks: renders the visual Mermaid PNG image flowchart.
    - Also prints the ASCII layout diagram for immediate readable structure.
    """
    if title:
        print(f"\n--- LangGraph Architecture: {title} ---")
    
    displayed_image = False
    try:
        from IPython.display import Image, display
        img_bytes = graph.get_graph().draw_mermaid_png()
        display(Image(img_bytes))
        displayed_image = True
    except Exception:
        pass

    try:
        ascii_repr = graph.get_graph().draw_ascii()
        print(ascii_repr)
    except Exception:
        if not displayed_image:
            try:
                print(graph.get_graph().draw_mermaid())
            except Exception:
                pass


if __name__ == "__main__":
    list_free_models()
    print("Testing get_llm() with automatic fallback...")
    test_llm = get_llm()
    try:
        res = test_llm.invoke("Hi! Tell me your model name in one short sentence.")
        print(f"\n[+] Connection Success!\nResponse: {res.content.strip()}")
    except Exception as e:
        print(f"\n[-] Test failed: {e}")
