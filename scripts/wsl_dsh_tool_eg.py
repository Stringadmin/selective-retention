import os

D = "/home/omnichat/node/node-linux/lib/node_modules/@deepseek-ai/dsh/node_modules/@deepseek-ai"

def head(fname, n=60):
    p = os.path.join(D, fname)
    if not os.path.isfile(p):
        print(f"  MISSING {fname}")
        return
    src = open(p, encoding="utf-8", errors="replace").read()
    print(f"=== {fname} ({len(src)}b) ===")
    for i, line in enumerate(src.splitlines()[:n], 1):
        print(f"{i:3} {line[:130]}")
    print()

for f in ("dsh-tool-ask-user/lib/index.js", "dsh-tool-todo/lib/index.js"):
    head(f)
