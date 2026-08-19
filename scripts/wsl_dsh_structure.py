import os, re

ROOT = "/mnt/d/DevTools/npm-global/node_modules/@deepseek-ai"
KW = re.compile(r"memory|记忆|remember|context.*file|CLAUDE|MEMORY|persist", re.I)

print("=== @deepseek-ai 下的包 ===")
for d in sorted(os.listdir(ROOT)):
    full = os.path.join(ROOT, d)
    if os.path.isdir(full):
        print(" ", d)

# Search for memory-related source files (limit depth, skip node_modules)
print("\n=== 含 memory/记忆 的源文件 ===")
hits = []
for dirpath, dirs, files in os.walk(ROOT):
    dirs[:] = [x for x in dirs if x != "node_modules" and not x.startswith(".")]
    depth = dirpath.replace(ROOT, "").count(os.sep)
    if depth > 6:
        dirs[:] = []
        continue
    for fn in files:
        if fn.endswith((".ts", ".js", ".md")) and re.search(r"memory|remember", fn, re.I):
            hits.append(os.path.join(dirpath, fn))
for h in hits[:20]:
    print("  ", h.replace(ROOT, "@deepseek-ai"))
if not hits:
    print("  (no memory-named files)")
