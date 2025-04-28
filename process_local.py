#!/usr/bin/env python3
import json
import os
from pathlib import Path
import re
import time
import requests
import chromadb
from tqdm import tqdm

# Configuration
COLLECTION_NAME = "manpages"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
OLLAMA_HOST = "http://localhost:11434"  # Use localhost directly
BATCH_SIZE = 4  # Very small batch size to avoid errors

class OllamaEmbeddingFunction:
    """Synchronous embedding function for Ollama."""
    
    def __init__(self, model_name="nomic-embed-text", ollama_base_url=OLLAMA_HOST):
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        
        # Test the connection
        try:
            response = requests.get(f"{self.ollama_base_url}/api/version")
            if response.status_code == 200:
                print(f"Connected to Ollama server at {self.ollama_base_url}")
                print(f"Server version: {response.json().get('version')}")
            else:
                raise Exception(f"Failed to connect to Ollama: {response.status_code}")
        except Exception as e:
            raise Exception(f"Failed to connect to Ollama: {e}")
    
    def __call__(self, input):
        """Generate embeddings for a list of texts."""
        if isinstance(input, str):
            input = [input]

        results = []
        for text in tqdm(input, desc="Generating embeddings", leave=False):
            try:
                response = requests.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": text},
                    timeout=30
                )

                if response.status_code != 200:
                    raise Exception(f"Error from Ollama API: {response.status_code} - {response.text}")

                result = response.json()
                if "embedding" not in result:
                    raise Exception("No embedding found in response")

                results.append(result["embedding"])
                
                # Short sleep to avoid overwhelming the server
                time.sleep(0.05)
                
            except Exception as e:
                print(f"Error getting embedding: {str(e)}")
                # Return a zero vector as a fallback
                import numpy as np
                results.append(np.zeros(4096).tolist())  # Typical embedding size

        return results

def chunk_text(text, command, section):
    """Split text into chunks with metadata."""
    chunks = []
    chunk_ids = []
    metadatas = []
    
    # Simple chunking strategy by paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    
    current_chunk = ""
    chunk_counter = 0
    
    for paragraph in paragraphs:
        cleaned_paragraph = paragraph.strip()
        if not cleaned_paragraph:
            continue
            
        # If adding this paragraph exceeds chunk size, finalize current chunk
        if len(current_chunk) + len(cleaned_paragraph) > CHUNK_SIZE and current_chunk:
            chunks.append(current_chunk.strip())
            chunk_id = f"{command}_{section}_{chunk_counter}"
            chunk_ids.append(chunk_id)
            metadatas.append({
                "command": command,
                "section": section,
                "chunk_id": chunk_counter
            })
            chunk_counter += 1
            current_chunk = ""
        
        # Add paragraph to current chunk
        if current_chunk:
            current_chunk += "\n\n" + cleaned_paragraph
        else:
            current_chunk = cleaned_paragraph
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
        chunk_id = f"{command}_{section}_{chunk_counter}"
        chunk_ids.append(chunk_id)
        metadatas.append({
            "command": command,
            "section": section,
            "chunk_id": chunk_counter
        })
    
    return chunks, chunk_ids, metadatas

def main():
    # Create paths
    data_path = "data/manpages.json"
    db_path = "data/chroma_db"
    os.makedirs(db_path, exist_ok=True)
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: {data_path} does not exist. Run extract_manpages.py or create_sample_data.py first.")
        return False
    
    # Initialize ChromaDB client with synchronous embedding function
    print("Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path=db_path)
    ef = OllamaEmbeddingFunction()
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef
    )
    
    # Get existing command/section pairs
    results = collection.get()
    processed_pairs = set()
    for metadata in results.get("metadatas", []):
        if metadata and "command" in metadata and "section" in metadata:
            processed_pairs.add((metadata["command"], metadata["section"]))
    
    # Load man pages from JSON
    print(f"Loading man pages from {data_path}...")
    with open(data_path, 'r') as f:
        manpages = json.load(f)
    
    # Filter to only unprocessed man pages
    unprocessed_manpages = [
        page for page in manpages 
        if (page["command"], page["section"]) not in processed_pairs
    ]
    
    print(f"Found {len(unprocessed_manpages)} unprocessed man pages out of {len(manpages)} total.")
    
    if not unprocessed_manpages:
        print("No new man pages to process. Exiting.")
        return True
    
    # Process man pages one by one (slower but more reliable)
    all_chunks = []
    all_ids = []
    all_metadatas = []
    
    for manpage in tqdm(unprocessed_manpages, desc="Processing man pages"):
        command = manpage["command"]
        section = manpage["section"]
        content = manpage["content"]
        
        chunks, chunk_ids, metadatas = chunk_text(content, command, section)
        
        all_chunks.extend(chunks)
        all_ids.extend(chunk_ids)
        all_metadatas.extend(metadatas)
    
    print(f"\nChunked {len(unprocessed_manpages)} man pages into {len(all_chunks)} chunks.")
    
    # Add chunks to the database in small batches
    print(f"Adding {len(all_chunks)} chunks to database...")
    
    # Process in small batches to avoid issues
    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="Adding to vector database"):
        end = min(i + BATCH_SIZE, len(all_chunks))
        try:
            collection.add(
                documents=all_chunks[i:end],
                ids=all_ids[i:end],
                metadatas=all_metadatas[i:end]
            )
            # Short sleep to avoid overwhelming the server
            time.sleep(0.1)
        except Exception as e:
            print(f"\nError adding batch {i}-{end} to database: {e}")
            print("Trying one by one...")
            
            # Try one by one as last resort
            for k in range(i, end):
                try:
                    collection.add(
                        documents=[all_chunks[k]],
                        ids=[all_ids[k]],
                        metadatas=[all_metadatas[k]]
                    )
                    time.sleep(0.2)  # Longer sleep between individual items
                except Exception as e:
                    print(f"Skipping document {k} due to error: {e}")
    
    print(f"\nSuccessfully processed man pages. You can now run the RAG server.")
    
    # Record processing timestamp
    with open(Path(db_path) / "last_processed.txt", "w") as f:
        f.write(f"{int(time.time())}")
    
    return True

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds") 