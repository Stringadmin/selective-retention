import os, glob, json

home = os.path.expanduser("~")
arch = os.path.join(home, "models/ollama-archive")

# 1. find big blobs
print("=== big files (>1.5G) under ollama-archive ===")
big = []
for root, dirs, files in os.walk(arch):
    for fn in files:
        p = os.path.join(root, fn)
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        if sz > 1.5e9:
            big.append((sz, p))
for sz, p in sorted(big, reverse=True):
    print(f"  {sz/1e9:.2f}GB  {p}")

# 2. resolve qwen3:4b and omnichat:4b digests -> actual file
print("=== resolve manifests ===")
for lib, tag in [("qwen3","4b"), ("omnichat","4b")]:
    mp = os.path.join(arch, "manifests/registry.ollama.ai/library", lib, tag)
    if not os.path.isfile(mp):
        print(f"  {lib}:{tag} manifest MISSING"); continue
    m = json.load(open(mp))
    for l in m.get("layers", []):
        if l.get("size",0) > 5e8:
            digest = l["digest"].replace(":", "-")
            cand = os.path.join(arch, "blobs", digest)
            print(f"  {lib}:{tag} digest={l['digest'][:25]} size={l['size']/1e9:.2f}GB exists={os.path.isfile(cand)}")
            if os.path.isfile(cand):
                print(f"    -> {cand}")
