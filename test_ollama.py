#!/usr/bin/env python3
import requests
import sys

# Try different URLs to find which one works
urls = [
    "http://localhost:11434",
    "http://127.0.0.1:11434",
    "http://axl:11434"  # The original URL from the codebase
]

for url in urls:
    print(f"Testing connection to Ollama at {url}...")
    try:
        # Test basic connection
        response = requests.get(f"{url}/api/version")
        if response.status_code == 200:
            version = response.json().get("version")
            print(f"✅ Successfully connected to Ollama version {version} at {url}")
            
            # Test models
            print(f"Checking available models at {url}...")
            response = requests.get(f"{url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                if models:
                    print(f"Available models: {', '.join(m.get('name') for m in models)}")
                else:
                    print("No models found. You need to pull some models.")
                    print("Run: ollama pull nomic-embed-text")
            else:
                print(f"❌ Failed to get models list: {response.status_code} - {response.text}")
        else:
            print(f"❌ Connection failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Error connecting to {url}: {e}")
        
    print("-" * 50)

print("If none of the connections work, you need to install and run Ollama.")
print("Visit: https://ollama.com/download")
print("Then pull the required models: ollama pull nomic-embed-text") 