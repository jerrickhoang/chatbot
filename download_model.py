#!/usr/bin/env python3
"""
Pre-download TinyLlama model for Docker image

This script downloads the model and tokenizer during Docker build
to embed them in the image for faster startup.
"""

import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    cache_dir = "/app/model_cache"
    
    print(f"📥 Downloading {model_name} to {cache_dir}")
    print("This will embed the model in the Docker image for faster startup...")
    
    try:
        # Set cache directory
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        
        # Download tokenizer
        print("📄 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print("✅ Tokenizer downloaded successfully")
        
        # Download model
        print("🤖 Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,  # Use float32 for broad compatibility
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ Model downloaded successfully")
        
        # Test that everything works
        print("🧪 Testing model and tokenizer...")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        test_input = "Hello"
        inputs = tokenizer(test_input, return_tensors="pt")
        print("✅ Model and tokenizer test passed")
        
        # Clean up memory
        del model
        del tokenizer
        
        print(f"🎉 Successfully downloaded {model_name}")
        print(f"📦 Model cached in: {cache_dir}")
        
        # List cache contents for verification
        print("\n📋 Cache contents:")
        for root, dirs, files in os.walk(cache_dir):
            level = root.replace(cache_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:3]:  # Show first 3 files only
                print(f"{subindent}{file}")
            if len(files) > 3:
                print(f"{subindent}... and {len(files) - 3} more files")
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()