from fastapi import FastAPI
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

app = FastAPI()

print("Downloading model (first run only)...")

model_path = hf_hub_download(
    repo_id="TheBloke/Phi-3-mini-GGUF",
    filename="phi-3-mini-instruct.Q4_K_M.gguf"
)

print("Model loaded at:", model_path)

llm = Llama(
    model_path=model_path,
    n_ctx=512,
    n_threads=2,
    n_batch=128
)

@app.get("/")
def home():
    return {"status": "LLM is running"}

@app.get("/generate")
def generate(prompt: str):
    output = llm(
        prompt,
        max_tokens=100,
        temperature=0.7,
        stop=["</s>"]
    )

    return {
        "prompt": prompt,
        "response": output["choices"][0]["text"]
    }