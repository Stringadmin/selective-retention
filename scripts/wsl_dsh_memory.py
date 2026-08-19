import json
import os
import re

HOME = os.path.expanduser("~")
DSH = os.path.join(HOME, ".dsh")

print("=== .dsh 下所有文件（排除 node_modules）===")
for root, dirs, files in os.walk(DSH):
    dirs[:] = [d for d in dirs if d not in ("node_modules", "sessions", "attachments", "profiles/web/node_modules")]
    if root.replace(DSH, "").count(os.sep) > 2:
        dirs[:] = []
        continue
    for f in files:
        p = os.path.join(root, f)
        print(f"  {os.path.relpath(p, DSH)}")

print("\n=== .agent-presets 结构 ===")
ap = os.path.join(DSH, ".agent-presets")
for root, dirs, files in os.walk(ap):
    for f in files:
        print(f"  {os.path.relpath(os.path.join(root, f), ap)}")

print("\n=== 搜记忆/上下文关键字的文件内容 ===")
KW = re.compile(r"memory|记忆|context|CLAUDE|MEMORY|系统提示|长期", re.I)
for root, dirs, files in os.walk(DSH):
    dirs[:] = [d for d in dirs if d != "node_modules"]
    if root.replace(DSH, "").count(os.sep) > 2:
        dirs[:] = []
        continue
    for f in files:
        if not f.endswith((".yaml", ".yml", ".json", ".md", ".txt")):
            continue
        p = os.path.join(root, f)
        try:
            txt = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        if KW.search(txt):
            print(f"\n--- {os.path.relpath(p, DSH)} (含关键字) ---")
            for line in txt.splitlines():
                if KW.search(line):
                    print(f"  {line.strip()[:120]}")
