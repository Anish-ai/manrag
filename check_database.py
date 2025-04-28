#!/usr/bin/env python3
import chromadb
import os
from pathlib import Path

# Check if data directory exists
data_dir = Path("data")
if not os.path.exists(data_dir):
    print("❌ Data directory doesn't exist. You need to run extract_manpages.py first.")
    exit(1)

# Check if manpages.json exists
json_path = data_dir / "manpages.json"
if not os.path.exists(json_path):
    print("❌ manpages.json doesn't exist. You need to run extract_manpages.py first.")
    exit(1)
else:
    import json
    try:
        with open(json_path, 'r') as f:
            manpages = json.load(f)
        print(f"✅ manpages.json exists with {len(manpages)} entries.")
    except Exception as e:
        print(f"❌ Error reading manpages.json: {e}")

# Check if ChromaDB directory exists
db_path = data_dir / "chroma_db"
if not os.path.exists(db_path):
    print("❌ ChromaDB directory doesn't exist. You need to run process_and_load.py first.")
    exit(1)
else:
    print(f"✅ ChromaDB directory exists at {db_path}")

# Check ChromaDB collection
try:
    client = chromadb.PersistentClient(path=str(db_path))
    collections = client.list_collections()
    
    if not collections:
        print("❌ No collections found in ChromaDB.")
    else:
        print(f"✅ Found {len(collections)} collections in ChromaDB:")
        for coll in collections:
            collection = client.get_collection(name=coll.name)
            count = collection.count()
            print(f"   - {coll.name}: {count} items")
except Exception as e:
    print(f"❌ Error checking ChromaDB: {e}")

print("\nTo populate the database:")
print("1. Run python extract_manpages.py to extract man pages")
print("2. Run python process_and_load.py to process and load into the vector database")
print("3. Run python rag_server.py to start the server")
print("4. Run python query_manpages.py 'your query here' to search") 