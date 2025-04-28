#!/usr/bin/env python3
import requests
from flask import Flask, request, Response, stream_with_context
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OllamaProxy")

app = Flask(__name__)

# Ollama target URL
TARGET_URL = "http://localhost:11434"

@app.route('/', defaults={'path': ''}, methods=["GET", "POST", "PUT", "DELETE"])
@app.route('/<path:path>', methods=["GET", "POST", "PUT", "DELETE"])
def proxy(path):
    """Forward all requests to the local Ollama server."""
    target_url = f"{TARGET_URL}/{path}"
    logger.info(f"Forwarding request to: {target_url}")
    
    # Forward the request headers
    headers = {key: value for key, value in request.headers if key != 'Host'}
    
    try:
        if request.method == 'GET':
            response = requests.get(target_url, headers=headers, params=request.args, stream=True)
        elif request.method == 'POST':
            json_data = request.get_json(silent=True)
            if json_data:
                response = requests.post(target_url, headers=headers, json=json_data, stream=True)
            else:
                response = requests.post(target_url, headers=headers, data=request.get_data(), stream=True)
        elif request.method == 'PUT':
            response = requests.put(target_url, headers=headers, data=request.get_data(), stream=True)
        elif request.method == 'DELETE':
            response = requests.delete(target_url, headers=headers, params=request.args, stream=True)
        else:
            return Response("Method not supported", status=405)
        
        # Stream the response back to the client
        def generate():
            for chunk in response.iter_content(chunk_size=4096):
                yield chunk
        
        proxy_response = Response(stream_with_context(generate()), status=response.status_code)
        
        # Forward the response headers
        for key, value in response.headers.items():
            if key.lower() not in ('content-length', 'transfer-encoding', 'connection'):
                proxy_response.headers[key] = value
                
        return proxy_response
        
    except requests.RequestException as e:
        logger.error(f"Error forwarding request: {e}")
        return Response(f"Error: {str(e)}", status=500)

if __name__ == '__main__':
    logger.info(f"Starting Ollama proxy: {TARGET_URL} -> http://axl:11434")
    app.run(host='0.0.0.0', port=11434) 