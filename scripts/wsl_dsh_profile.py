import json, os

PROF = os.path.expanduser("~/.dsh/profiles/web")
print("=== web profile package.json ===")
pj = os.path.join(PROF, "package.json")
if os.path.isfile(pj):
    with open(pj) as f:
        p = json.load(f)
    print(" name:", p.get("name"))
    print(" dsh.profile:", json.dumps(p.get("dsh.profile") or p.get("dsh"), ensure_ascii=False)[:400])
    print(" deps:", list((p.get("dependencies") or {}).keys()))

print("\n=== profile 目录内容 ===")
for x in sorted(os.listdir(PROF)):
    print("  ", x)

print("\n=== 已装的 deepseek 组合包 ===")
nm = os.path.join(PROF, "node_modules", "@deepseek-ai")
if os.path.isdir(nm):
    for x in sorted(os.listdir(nm)):
        print("  ", x)

# look for memory/context handling in installed bundles
print("\n=== 组合包里的 memory/context 线索 ===")
import re
KW = re.compile(r"memory|remember|MEMORY|长期|记忆|历史", re.I)
for dirpath, dirs, files in os.walk(nm):
    dirs[:] = [d for d in dirs if d != "node_modules" and not d.startswith(".")]
    if dirpath.replace(nm, "").count(os.sep) > 3:
        dirs[:] = []
        continue
    for fn in files:
        if re.search(r"memory|context|remember", fn, re.I):
            print("  ", os.path.join(dirpath, fn).replace(nm, "@dsh"))
