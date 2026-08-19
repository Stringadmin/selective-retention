import json, os, re

D = "/mnt/d/DevTools/npm-global/node_modules/@deepseek-ai/dsh"
pj = os.path.join(D, "package.json")
print("=== package.json ===")
with open(pj) as f:
    p = json.load(f)
for k in ("name", "version", "description", "bin", "main", "exports"):
    if k in p:
        print(f"  {k}: {p[k]}")

for name in ("README.zh.md", "README.md"):
    rp = os.path.join(D, name)
    if not os.path.isfile(rp):
        continue
    print(f"\n=== {name} 插件/记忆/钩子相关行 ===")
    with open(rp, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if re.search(r"插件|扩展|plugin|extension|hook|记忆|memory|skill|mcp|MCP", line):
                print(f"  {i}: {line.rstrip()[:120]}")
    break
