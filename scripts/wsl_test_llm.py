import os, sys

BLOB = "/home/omnichat/models/ollama-archive/blobs/sha256-3e4cb14174460404e7a233e531675303b2fbf7749c02f91864fe311ab6344e4f"

print("exists:", os.path.isfile(BLOB), "size GB:", round(os.path.getsize(BLOB)/1e9, 2) if os.path.isfile(BLOB) else "NA", flush=True)
with open(BLOB, "rb") as f:
    magic = f.read(4)
print("magic:", magic, flush=True)
if magic != b"GGUF":
    print("NOT a GGUF file, abort", flush=True)
    sys.exit(1)

try:
    from llama_cpp import Llama
except ImportError:
    print("llama_cpp not installed", flush=True)
    sys.exit(2)

import time
print("loading qwen3:4b (n_gpu_layers=99, ctx=8192)...", flush=True)
t0 = time.time()
llm = Llama(
    model_path=BLOB,
    n_ctx=8192,
    n_gpu_layers=99,   # offload all layers to GPU (2.5GB model fits easily in 12GB)
    n_batch=512,
    verbose=False,
)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

msgs = [{"role": "user", "content": "用一句话解释什么是长期记忆。"}]
t0 = time.time()
out = llm.create_chat_completion(messages=msgs, max_tokens=60, temperature=0.0)
dt = time.time() - t0
text = out["choices"][0]["message"]["content"]
print(f"gen {dt:.1f}s:", text, flush=True)
print("OK", flush=True)
