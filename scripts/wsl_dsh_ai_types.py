import os

BASE = "/mnt/d/DevTools/npm-global/node_modules/@deepseek-ai/dsh/node_modules/@deepseek-ai/dsh-agent-instructions/lib/types"
for f in ("config.d.ts", "state.d.ts", "index.d.ts"):
    p = os.path.join(BASE, f)
    if not os.path.isfile(p):
        continue
    src = open(p, encoding="utf-8", errors="replace").read()
    print(f"=== {f} ({len(src)}b) ===")
    print(src[:1800])
    print()
