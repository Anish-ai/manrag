#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import chromadb
import requests
import uvicorn
import asyncio
import aiohttp
from typing import List, Optional, Dict, Any, Iterator
from sse_starlette.sse import EventSourceResponse
import os
import json
import time
import re
from contextlib import asynccontextmanager

# Configuration
COLLECTION_NAME = "manpages"
DB_PATH = "data/chroma_db"
OLLAMA_HOST = "http://axl:11434"  # Ollama server address
OLLAMA_MODEL = "nomic-embed-text"  # Default embedding model for search
CHAT_MODEL = "deepseek-coder:6.7b-instruct"  # Default model for answer generation

class OllamaEmbeddingFunction:
    """Custom embedding function for querying."""

    def __init__(self, model_name=OLLAMA_MODEL, ollama_base_url=OLLAMA_HOST):
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url

    def __call__(self, input):
        """Generate embeddings for a list of texts."""
        if isinstance(input, str):
            input = [input]

        results = []
        for text in input:
            try:
                response = requests.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": text},
                    timeout=30
                )

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error from Ollama API: {response.status_code} - {response.text}"
                    )

                result = response.json()
                if "embedding" not in result:
                    raise HTTPException(
                        status_code=500,
                        detail="No embedding found in response"
                    )

                results.append(result["embedding"])
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error getting embedding: {str(e)}")

        return results

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure data directory exists
    os.makedirs(DB_PATH, exist_ok=True)
    
    # Initialize ChromaDB client
    print("Initializing ChromaDB client...")
    app.client = chromadb.PersistentClient(path=DB_PATH)
    app.ef = OllamaEmbeddingFunction()
    try:
        app.collection = app.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=app.ef
        )
        print(f"Successfully connected to collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        # Create empty collection if it doesn't exist
        if not os.path.exists(DB_PATH):
            os.makedirs(DB_PATH, exist_ok=True)

        try:
            app.collection = app.client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=app.ef
            )
            print(f"Created new collection: {COLLECTION_NAME}")
        except Exception as e:
            print(f"Failed to create collection: {e}")
            raise

    yield

    # Clean up (if needed)
    print("Shutting down RAG server...")

# Define the FastAPI app with lifespan
app = FastAPI(
    title="ManPage RAG API",
    description="API for querying man pages using RAG",
    lifespan=lifespan
)

class GenerateRequest(BaseModel):
    query: str
    contexts: List[Dict[str, Any]]
    model: Optional[str] = CHAT_MODEL
    stream: Optional[bool] = False

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    elapsed_time: float

class GenerateResponse(BaseModel):
    answer: str
    model: str

@app.get("/health")
async def health_check():
    return {"status": "ok", "server": "ManPage RAG API"}

@app.get("/search", response_model=SearchResponse)
async def search(
    query: str,
    n_results: int = Query(5, ge=1, le=200),
    command: Optional[str] = None,
    section: Optional[str] = None
):
    start_time = time.time()

    # Prepare filters
    metadata_filters = {}
    if command:
        metadata_filters["command"] = command
    if section:
        metadata_filters["section"] = section

    try:
        # Perform the query
        results = app.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=metadata_filters if metadata_filters else None
        )

        # Format results
        formatted_results = []
        if results and "documents" in results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]

                formatted_results.append({
                    "content": doc,
                    "command": metadata.get("command", "Unknown"),
                    "section": metadata.get("section", "Unknown"),
                    "chunk_id": metadata.get("chunk_id", "Unknown"),
                    "distance": distance,
                    "relevance": 1.0 - distance  # Convert distance to relevance score
                })

        elapsed_time = time.time() - start_time

        return {"results": formatted_results, "elapsed_time": elapsed_time}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying database: {str(e)}")

async def generate_stream(request: GenerateRequest) -> Iterator[str]:
    """Generate a streaming response from Ollama."""
    # Format context
    context = "\n\n".join([
        f"From man page {c['command']}({c['section']}):\n{c['content']}"
        for c in request.contexts
    ])

    # Add an instruction to the prompt to use thinking
    prompt = f"""You are ManPageGPT, a helpful AI assistant focusing on Unix/Linux manual pages.

I will provide you with content from relevant man pages, and you should use this information to answer the query concisely.

When you need to work through a complex problem, please use <think>...</think>
tags to show your reasoning process. You should sort the data in the context by
what is most relevant to the query, and ignore things of lesser relevance. If
more than one answer in the context is relevance, provide a compare/contrast on
different ways to answer the query.

QUERY: {request.query}

RELEVANT MAN PAGE CONTENT:
{context}

Please provide a clear, accurate answer based only on the provided man page content.
If the answer isn't contained in the man pages provided, say so.
Format command names and options in backticks like `command`.
Include key examples where helpful.
"""

    # Add special headers to identify thinking tags in the stream
    thinking_tag_open = b"<<THINKING_START>>"
    thinking_tag_close = b"<<THINKING_END>>"

    # Connect to Ollama with streaming enabled
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": request.model,
                "prompt": prompt,
                "stream": True
            }
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise HTTPException(status_code=response.status, detail=f"Error from Ollama: {error_text}")

            # Process the streaming response
            in_thinking = False
            thinking_buffer = ""

            async for line in response.content:
                if not line:
                    continue

                try:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue

                    data = json.loads(line)

                    if "response" in data:
                        chunk = data["response"]
                        if not chunk:
                            continue

                        # Check for thinking tags
                        if "<think>" in chunk and not in_thinking:
                            # Start of thinking section
                            parts = chunk.split("<think>", 1)
                            if parts[0]:  # Text before the thinking tag
                                yield parts[0]
                            in_thinking = True
                            thinking_buffer = parts[1]
                            yield thinking_tag_open.decode()
                        elif "</think>" in chunk and in_thinking:
                            # End of thinking section
                            parts = chunk.split("</think>", 1)
                            thinking_buffer += parts[0]
                            yield thinking_buffer
                            yield thinking_tag_close.decode()
                            in_thinking = False
                            thinking_buffer = ""
                            if parts[1]:  # Text after the thinking tag
                                yield parts[1]
                        elif in_thinking:
                            # Inside thinking section
                            thinking_buffer += chunk
                            yield chunk
                        else:
                            # Normal text
                            yield chunk

                    # End of response
                    if data.get("done", False):
                        # If we're still in a thinking section, close it
                        if in_thinking:
                            yield thinking_buffer
                            yield thinking_tag_close.decode()
                        break

                except Exception as e:
                    print(f"Error processing stream: {e}")
                    continue

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate an answer based on retrieved chunks using Ollama."""
    if request.stream:
        raise HTTPException(status_code=400, detail="Use /generate_stream for streaming responses")

    # Format context
    context = "\n\n".join([
        f"From man page {c['command']}({c['section']}):\n{c['content']}"
        for c in request.contexts
    ])

    prompt = f"""You are ManPageGPT, a helpful AI assistant focusing on Unix/Linux manual pages.

I will provide you with content from relevant man pages, and you should use this information to answer the query concisely.

When you need to work through a complex problem, please use <think>...</think> tags to show your reasoning process.

QUERY: {request.query}

RELEVANT MAN PAGE CONTENT:
{context}

Please provide a clear, accurate answer based only on the provided man page content.
If the answer isn't contained in the man pages provided, say so.
Format command names and options in backticks like `command`.
Include key examples where helpful.
"""

    # Call Ollama to generate the response
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": request.model, "prompt": prompt, "stream": False}
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Error calling Ollama: {response.status_code} - {response.text}"
            )

        result = response.json()
        if "response" not in result:
            raise HTTPException(status_code=500, detail="No response found in Ollama result")

        # Remove thinking tags from final response
        answer = result["response"]
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)

        return {"answer": answer, "model": request.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.post("/generate_stream")
async def generate_stream_endpoint(request: GenerateRequest):
    """Generate a streaming response."""
    return StreamingResponse(
        generate_stream(request),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
