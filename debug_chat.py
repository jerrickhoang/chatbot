#!/usr/bin/env python3
"""Debug script to test TinyLlama chat template formatting"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_chat_template():
    print("Loading TinyLlama tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Test messages
    messages = [
        {"role": "user", "content": "Hi"}
    ]
    
    print(f"\nOriginal messages: {messages}")
    
    # Test if chat template works
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"\nFormatted with chat template:")
        print(repr(formatted_prompt))
        print(f"\nActual prompt:")
        print(formatted_prompt)
    except Exception as e:
        print(f"\nChat template failed: {e}")
        
        # Fallback formatting
        prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant:"
        
        print(f"\nFallback prompt:")
        print(repr(prompt))
        print(f"\nActual fallback:")
        print(prompt)
    
    # Check tokenizer properties
    print(f"\nTokenizer info:")
    print(f"Chat template exists: {hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None}")
    print(f"EOS token: {tokenizer.eos_token}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"Pad token: {tokenizer.pad_token}")
    print(f"Pad token ID: {tokenizer.pad_token_id}")

if __name__ == "__main__":
    test_chat_template()