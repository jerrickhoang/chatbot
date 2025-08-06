#!/usr/bin/env python3
"""
TinyLlama Chat Server Runner

This script helps you run the API server with different model configurations.

Usage:
    python run_server.py                    # Use TinyLlama-1.1B-Chat (CPU/GPU)
    python run_server.py --model MODEL_NAME # Use custom model
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Run TinyLlama Chat API Server')
    parser.add_argument('--model', type=str, 
                       help='Custom model name to use')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to run server on (default: 8000)')
    
    args = parser.parse_args()
    
    # Set model based on arguments
    if args.model:
        os.environ['MODEL_NAME'] = args.model
        print(f"🔧 Using custom model: {args.model}")
    else:
        os.environ['MODEL_NAME'] = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
        print("🚀 Using TinyLlama-1.1B-Chat model (works on CPU and GPU)")
        print("   This model is lightweight and fast!")
    
    # Set port
    os.environ['PORT'] = str(args.port)
    
    print(f"🌐 Starting server on http://localhost:{args.port}")
    print("🔍 Check http://localhost:{}/docs for API documentation".format(args.port))
    print("💻 Open frontend.html in your browser to test the API")
    print("---")
    
    # Import and run the app
    try:
        import uvicorn
        uvicorn.run("app:app", host="0.0.0.0", port=args.port, reload=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()