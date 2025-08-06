from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import Optional, List
import os
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
tokenizer = None
model_name = None

class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message")

class CompletionRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    max_tokens: Optional[int] = Field(default=512, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    reasoning_level: Optional[str] = Field(default="medium", description="Reasoning level: low, medium, or high")

class CompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: List[dict]
    usage: dict

async def load_model():
    global model, tokenizer, model_name
    try:
        # Use TinyLlama by default - works on both CPU and GPU
        model_name = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        logger.info(f"Loading model: {model_name}")
        
        # Check if GPU is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Set up cache directory for models
        cache_dir = os.getenv('TRANSFORMERS_CACHE', None)
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate settings based on device
        if device == "cuda":
            logger.info("Loading model with GPU support")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir
            )
        else:
            logger.info("Loading model with CPU support")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir
            )
            model = model.to(device)
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Make sure you have sufficient memory and a stable internet connection")
        logger.info("Set MODEL_NAME environment variable to use a different model")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_model()
    yield

app = FastAPI(
    title="TinyLlama Chat API Server",
    description="API server for TinyLlama-1.1B-Chat model with CPU/GPU support",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

@app.get("/")
async def root():
    return {
        "message": "GPT-OSS-20B API Server",
        "endpoints": {
            "health": "/health",
            "chat": "/v1/chat/completions",
            "docs": "/docs"
        }
    }

@app.post("/v1/chat/completions", response_model=CompletionResponse)
async def chat_completions(request: CompletionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        conversation = []
        for msg in request.messages:
            conversation.append({"role": msg.role, "content": msg.content})
        
        # Use TinyLlama-specific chat format
        if "TinyLlama" in model_name:
            # TinyLlama uses a specific chat format
            prompt = "<|system|>\nYou are a helpful AI assistant.</s>\n"
            for msg in conversation:
                if msg["role"] == "user":
                    prompt += f"<|user|>\n{msg['content']}</s>\n"
                elif msg["role"] == "assistant":
                    prompt += f"<|assistant|>\n{msg['content']}</s>\n"
            prompt += "<|assistant|>\n"
        else:
            # Try standard chat template first
            try:
                prompt = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                # Generic fallback format
                prompt = ""
                for msg in conversation:
                    if msg["role"] == "user":
                        prompt += f"User: {msg['content']}\n"
                    elif msg["role"] == "assistant":
                        prompt += f"Assistant: {msg['content']}\n"
                prompt += "Assistant:"
        
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        generation_kwargs = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "do_sample": request.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_kwargs
            )
        
        response_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return CompletionResponse(
            id=f"chatcmpl-{hash(prompt) % 1000000}",
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": inputs["input_ids"].shape[1],
                "completion_tokens": len(tokenizer.encode(response_text)),
                "total_tokens": inputs["input_ids"].shape[1] + len(tokenizer.encode(response_text))
            }
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)