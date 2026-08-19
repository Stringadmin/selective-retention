import os

D = "/home/omnichat/node/node-linux/lib/node_modules/@deepseek-ai/dsh/node_modules/@deepseek-ai/dsh-tools"

print("=== dsh-tools 结构 ===")
for root, dirs, files in os.walk(D):
    dirs[:] = [d for d in dirs if d != "node_modules"]
    depth = root.replace(D, "").count(os.sep)
    if depth > 2:
        dirs[:] = []
        continue
    for f in files:
        print(f"  {os.path.relpath(os.path.join(root, f), D)}")

# find the defineTool type/impl
print("\n=== defineTool 定义位置 ===")
import re
for root, dirs, files in os.walk(D):
    dirs[:] = [d for d in dirs if d != "node_modules"]
    for f in files:
        if f.endswith(".d.ts") or f.endswith(".js"):
            p = os.path.join(root, f)
            try:
                txt = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            if "defineTool" in txt:
                for line in txt.splitlines():
                    if "defineTool" in line:
                        print(f"  {os.path.relpath(p, D)}: {line.strip()[:120]}")
