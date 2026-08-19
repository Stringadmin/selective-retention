import os

BASE = os.path.expanduser("~/.dsh/profiles/web/node_modules/dsh-routing-suite")
for rel in ("lib/index.js",):
    p = os.path.join(BASE, rel)
    if not os.path.isfile(p):
        print("missing", rel); continue
    src = open(p, encoding="utf-8", errors="replace").read()
    print(f"=== {rel} ({len(src)} bytes) ===")
    # print first 120 lines to see the plugin shape
    for i, line in enumerate(src.splitlines()[:120], 1):
        print(f"{i:3} {line}")
