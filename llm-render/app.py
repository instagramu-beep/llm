# from fastapi import FastAPI
# from llama_cpp import Llama
# from huggingface_hub import hf_hub_download

# app = FastAPI()

# print("Downloading small model...")

# model_path = hf_hub_download(
#     repo_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
#     filename="qwen1_5-0_5b-chat-q4_0.gguf"
# )

# print("Model ready:", model_path)

# llm = Llama(
#     model_path=model_path,
#     n_ctx=256,        # small context to save RAM
#     n_threads=1,      # low CPU (Render free)
#     n_batch=64
# )

# @app.get("/")
# def home():
#     return {"status": "LLM is running"}

# @app.get("/generate")
# def generate(prompt: str):
#     output = llm(
#         prompt,
#         max_tokens=50,
#         temperature=0.7
#     )

#     return {
#         "prompt": prompt,
#         "response": output["choices"][0]["text"]
#     }
from fastapi import FastAPI
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

app = FastAPI()

print("Downloading small model...")

# Download a small GGUF model (fits Render free tier)
model_path = hf_hub_download(
    repo_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
    filename="qwen1_5-0_5b-chat-q4_0.gguf"
)

print("Model ready:", model_path)

# Load model
llm = Llama(
    model_path=model_path,
    n_ctx=256,        # small context (important for low RAM)
    n_threads=1,      # low CPU usage
    n_batch=64
)

@app.get("/")
def home():
    return {"status": "LLM is running"}


@app.get("/generate")
def generate(prompt: str):
    # Proper chat format (VERY IMPORTANT)
    formatted_prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{prompt}\n<|assistant|>\n"

    output = llm(
        formatted_prompt,
        max_tokens=80,
        temperature=0.7,
        stop=["<|end|>", "<|user|>"]
    )

    return {
        "prompt": prompt,
        "response": output["choices"][0]["text"].strip()
    }
