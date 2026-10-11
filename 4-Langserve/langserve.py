import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Avoid shadowing installed 'langserve' package when this file is named langserve.py
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).parent.resolve():
    _script_dir = sys.path.pop(0)
    from langserve import add_routes
    sys.path.insert(0, _script_dir)
else:
    from langserve import add_routes

# Load environment variables from .env
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(api_key=groq_api_key, model_name="openai/gpt-oss-120b")
prompt = ChatPromptTemplate.from_template("Translate the following text to {target_language}: {text}")
chain = prompt | model | StrOutputParser()

app = FastAPI(title="Langchain demo", version="1.0.0", description="demo")

# -----------------------------------------------------------------------------
# What is LangServe?
# LangServe is a library that helps developers deploy LangChain Runnables and 
# chains as production-ready REST APIs. Built on top of FastAPI and Pydantic,
# it automatically provides schema validation, streaming, and concurrency.
# -----------------------------------------------------------------------------

# add_routes() binds the LangChain Runnable (chain) to the FastAPI app.
# By setting path="/chain", LangServe automatically generates the following routes:
#
# 1. POST /chain/invoke
#    - Invokes the chain synchronously with a single input:
#      Body: {"input": {"target_language": "French", "text": "Hello, how are you?"}}
#
# 2. POST /chain/stream
#    - Streams tokens in real time using Server-Sent Events (SSE).
#
# 3. POST /chain/batch
#    - Processes multiple inputs in parallel:
#      Body: {"inputs": [{"target_language": "French", "text": "Hi"}, ...]}
#
# 4. GET /chain/playground/
#    - Interactive built-in web UI to test inputs, streaming, and view traces in the browser.
#
# 5. GET /chain/input_schema & /chain/output_schema
#    - OpenAPI/JSON schemas derived from the chain's input and output types.
#
# Client Usage:
# You can interact with this API using regular HTTP requests (curl, requests, fetch)
# or via LangServe's RemoteRunnable:
#   from langserve import RemoteRunnable
#   remote_chain = RemoteRunnable("http://127.0.0.1:8000/chain/")
#   remote_chain.invoke({"target_language": "French", "text": "Hello"})
# -----------------------------------------------------------------------------
add_routes(
    app,
    chain,
    path="/chain",
)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)