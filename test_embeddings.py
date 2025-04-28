#!/usr/bin/env python3
import requests
import sys
import os

MODEL = "nomic-embed-text"
TEST_TEXT = "This is a test to check if the embedding API works."

# Try both "axl" and "localhost"
urls = [
    "http://axl:11434",
    "http://localhost:11434"
]

# Add or update the hosts file entry
def suggest_hosts_edit():
    if os.name == 'posix':  # Linux/Mac
        print("\n--- SOLUTION ---")
        print("Add 'axl' to your hosts file by running:")
        print("sudo sh -c 'echo \"127.0.0.1 axl\" >> /etc/hosts'")
    else:  # Windows
        print("\n--- SOLUTION ---")
        print("Add 'axl' to your hosts file:")
        print("1. Run Notepad as administrator")
        print("2. Open C:\\Windows\\System32\\drivers\\etc\\hosts")
        print("3. Add this line: 127.0.0.1 axl")
        print("4. Save the file")

for url in urls:
    print(f"\nTesting embeddings API at {url}...")
    try:
        # Test embeddings API
        response = requests.post(
            f"{url}/api/embeddings",
            json={"model": MODEL, "prompt": TEST_TEXT},
            timeout=10
        )
        
        if response.status_code == 200:
            embedding = response.json().get("embedding")
            embedding_length = len(embedding) if embedding else 0
            print(f"✅ Success! Got embedding of length {embedding_length}")
        else:
            print(f"❌ API returned error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")

suggest_hosts_edit() 