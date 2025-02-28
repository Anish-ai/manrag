#!/usr/bin/env python3
import json
import os
from pathlib import Path
import re
import time
import multiprocessing
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import torch
from tqdm import tqdm
import chromadb

# Check for GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Configuration
COLLECTION_NAME = "manpages"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CONCURRENT_REQUESTS = 16  # Maximum number of concurrent requests to Ollama
NUM_WORKERS = multiprocessing.cpu_count()  # Number of worker processes
OLLAMA_HOST = "http://axl:11434"  # Ollama server address

class AsyncOllamaEmbeddingFunction:
    """Async embedding function for Ollama with parallel GPU processing."""
    
    def __init__(self, model_name="nomic-embed-text", ollama_base_url=OLLAMA_HOST):
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        # Test the connection
        import requests
        try:
            response = requests.get(f"{self.ollama_base_url}/api/version")
            if response.status_code == 200:
                print(f"Connected to Ollama server at {self.ollama_base_url}")
                print(f"Server version: {response.json().get('version')}")
            else:
                raise Exception(f"Failed to connect to Ollama: {response.status_code}")
        except Exception as e:
            raise Exception(f"Failed to connect to Ollama: {e}")
    
    async def _get_embedding_async(self, session, text):
        """Get embedding for a single text using async request."""
        async with self.semaphore:
            async with session.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text}
            ) as response:
                if response.status != 200:
                    text_response = await response.text()
                    raise Exception(f"Error from Ollama API: {response.status} - {text_response}")
                result = await response.json()
                if "embedding" not in result:
                    raise Exception("No embedding found in response")
                return result["embedding"]
    
    async def _get_embeddings_async(self, texts):
        """Get embeddings for multiple texts in parallel."""
        async with aiohttp.ClientSession() as session:
            tasks = [self._get_embedding_async(session, text) for text in texts]
            return await asyncio.gather(*tasks)
    
    # FIX: Changed parameter name from 'texts' to 'input' to match ChromaDB's expectation
    def __call__(self, input):
        """Interface for ChromaDB to get embeddings."""
        loop = asyncio.new_event_loop()
        try:
            embeddings = loop.run_until_complete(self._get_embeddings_async(input))
            return embeddings
        finally:
            loop.close()

class ManPageProcessor:
    def __init__(self, data_path="data/manpages.json", db_path="data/chroma_db"):
        self.data_path = data_path
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Use our async Ollama embedding function
        self.ef = AsyncOllamaEmbeddingFunction()
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef
        )
    
    def chunk_text(self, text, command, section):
        """Split text into chunks with metadata."""
        chunks = []
        chunk_ids = []
        metadatas = []
        
        # Simple chunking strategy by paragraphs first
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
    
    def get_processed_commands(self):
        """Get a set of command/section pairs that are already in the database."""
        try:
            # Query the collection for all entries (limit to just getting IDs)
            results = self.collection.get()
            
            # Extract command and section from metadatas
            processed_pairs = set()
            for metadata in results.get("metadatas", []):
                if metadata and "command" in metadata and "section" in metadata:
                    processed_pairs.add((metadata["command"], metadata["section"]))
                    
            return processed_pairs
        except Exception as e:
            print(f"Error getting processed commands: {e}")
            return set()
    
    def process_manpage(self, manpage):
        """Process a single manpage and return chunks."""
        command = manpage["command"]
        section = manpage["section"]
        content = manpage["content"]
        
        chunks, chunk_ids, metadatas = self.chunk_text(content, command, section)
        
        return chunks, chunk_ids, metadatas
    
    def load_and_process(self, force_reprocess=False):
        """Load man pages, process them, and add to vector DB with GPU acceleration."""
        start_time = time.time()
        
        # Check if file exists
        if not os.path.exists(self.data_path):
            print(f"Error: {self.data_path} does not exist. Run extract_manpages.py first.")
            return False
        
        # Load man pages from JSON
        print(f"Loading man pages from {self.data_path}...")
        with open(self.data_path, 'r') as f:
            manpages = json.load(f)
            
        # Get already processed command/section pairs
        processed_pairs = set() if force_reprocess else self.get_processed_commands()
        
        # Filter to only unprocessed man pages
        unprocessed_manpages = [
            page for page in manpages 
            if (page["command"], page["section"]) not in processed_pairs
        ]
        
        print(f"Found {len(unprocessed_manpages)} unprocessed man pages out of {len(manpages)} total.")
        
        if not unprocessed_manpages:
            print("No new man pages to process. Exiting.")
            return True
        
        # Create master progress bar
        master_pbar = tqdm(total=3, desc="Overall Progress", position=0)
        master_pbar.set_description("Step 1/3: Chunking man pages")
        
        # Process man pages in parallel using ThreadPoolExecutor
        print(f"\nProcessing {len(unprocessed_manpages)} man pages using {NUM_WORKERS} workers...")
        all_chunks = []
        all_ids = []
        all_metadatas = []
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for manpage in unprocessed_manpages:
                futures.append(executor.submit(self.process_manpage, manpage))
                
            # Process results as they complete
            for future in tqdm(
                futures, 
                total=len(futures), 
                desc="Chunking man pages", 
                position=1, 
                leave=True
            ):
                chunks, chunk_ids, metadatas = future.result()
                all_chunks.extend(chunks)
                all_ids.extend(chunk_ids)
                all_metadatas.extend(metadatas)
        
        master_pbar.update(1)
        master_pbar.set_description("Step 2/3: Preparing database")
        
        print(f"\nChunked {len(unprocessed_manpages)} man pages into {len(all_chunks)} chunks.")
        
        # Add chunks to the database in batches optimized for GPU
        print(f"Adding {len(all_chunks)} chunks to database...")
        
        # Determine optimal batch size based on available GPU memory
        # For Ollama API with parallel requests, larger batch sizes are better
        batch_size = 64  # Default batch size
        if len(all_chunks) < batch_size:
            batch_size = len(all_chunks)
        
        master_pbar.update(1)
        master_pbar.set_description("Step 3/3: Adding to vector database")
        
        # Process in batches
        batch_count = (len(all_chunks) + batch_size - 1) // batch_size
        
        for i in tqdm(
            range(0, len(all_chunks), batch_size),
            total=batch_count,
            desc="Adding to vector database",
            position=1,
            leave=True
        ):
            end = min(i + batch_size, len(all_chunks))
            try:
                self.collection.add(
                    documents=all_chunks[i:end],
                    ids=all_ids[i:end],
                    metadatas=all_metadatas[i:end]
                )
            except Exception as e:
                print(f"\nError adding batch {i}-{end} to database: {e}")
                print("Trying with smaller batch size...")
                
                # Fall back to smaller batches
                smaller_batch = 8
                for j in range(i, end, smaller_batch):
                    sub_end = min(j + smaller_batch, end)
                    try:
                        self.collection.add(
                            documents=all_chunks[j:sub_end],
                            ids=all_ids[j:sub_end],
                            metadatas=all_metadatas[j:sub_end]
                        )
                    except Exception as e:
                        print(f"Error adding smaller batch {j}-{sub_end}: {e}")
                        # Try one by one as last resort
                        for k in range(j, sub_end):
                            try:
                                self.collection.add(
                                    documents=[all_chunks[k]],
                                    ids=[all_ids[k]],
                                    metadatas=[all_metadatas[k]]
                                )
                            except Exception as e:
                                print(f"Skipping document {k} due to error: {e}")
        
        master_pbar.update(1)
        master_pbar.close()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\nSuccessfully added {len(all_chunks)} chunks from {len(unprocessed_manpages)} man pages to ChromaDB.")
        print(f"Total processing time: {elapsed_time:.2f} seconds")
        print(f"Average time per chunk: {elapsed_time/len(all_chunks):.4f} seconds")
        print(f"Average time per man page: {elapsed_time/len(unprocessed_manpages):.4f} seconds")
        
        # Record processing timestamp
        with open(Path(self.db_path) / "last_processed.txt", "w") as f:
            f.write(f"{int(time.time())}")
            
        return True

def gpu_info():
    """Get information about available GPUs."""
    if not torch.cuda.is_available():
        print("No CUDA devices available")
        return
    
    print("\n=== GPU Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of devices: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nDevice {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"  Memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        
        if hasattr(torch.cuda, 'get_device_properties'):
            prop = torch.cuda.get_device_properties(i)
            print(f"  Total memory: {prop.total_memory / 1024**3:.2f} GB")
            print(f"  Compute capability: {prop.major}.{prop.minor}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process man pages for RAG with GPU acceleration")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all man pages")
    parser.add_argument("--gpu-info", action="store_true", help="Display GPU information")
    parser.add_argument("--concurrency", type=int, help=f"Number of concurrent requests (default: {MAX_CONCURRENT_REQUESTS})")
    args = parser.parse_args()
    
    if args.gpu_info:
        gpu_info()
        
    if args.concurrency:
        MAX_CONCURRENT_REQUESTS = args.concurrency
        print(f"Using {MAX_CONCURRENT_REQUESTS} concurrent requests")
        
    processor = ManPageProcessor()
    processor.load_and_process(force_reprocess=args.force)