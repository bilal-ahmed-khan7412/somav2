import uvicorn
import os
import argparse
import sys

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def main():
    parser = argparse.ArgumentParser(description="SOMA V2 Swarm OS Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--model", default="ollama/qwen2.5:3b", help="LLM model string")
    
    args = parser.parse_args()
    
    os.environ["SOMA_MODEL"] = args.model
    
    print(f"Starting SOMA V2 Server on {args.host}:{args.port}...")
    uvicorn.run("soma_v2.api.server:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
