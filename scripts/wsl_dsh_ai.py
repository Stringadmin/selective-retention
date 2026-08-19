import os

BASE = "/mnt/d/DevTools/npm-global/node_modules/@deepseek-ai/dsh/node_modules/@deepseek-ai/dsh-agent-instructions"
print("=== 包结构 ===")
for root, dirs, files in os.walk(BASE):
    dirs[:] = [d for d in dirs if d != "node_modules"]
    depth = root.replace(BASE, "").count(os.sep)
    if depth > 2:
        dirs[:] = []
        continue
    for f in files:
        p = os.path.join(root, f)
        if os.path.getsize(p) < 200000:
            print(f"  {os.path.relpath(p, BASE)}")

print("\n=== lib 里的 JS 文件 ===")
import glob
for f in glob.glob(os.path.join(BASE, "lib", "*.js")):
    print(f"  {os.path.basename(f)} ({os.path.getsize(f)}b)")

# print main entry head to find the apply() and write API
pj = os.path.join(BASE, "package.json")
if os.path.isfile(pj):
    import json
    d = json.load(open(pj))
    print("\n=== package.json main/exports ===")
    print(" main:", d.get("main"))
    print(" exports:", json.dumps(d.get("exports"))[:200])
