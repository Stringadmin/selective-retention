import json
import os

R = os.path.expanduser("~/.dsh/profiles/web/node_modules/dsh-routing-suite")
PROF = os.path.expanduser("~/.dsh/profiles/web")

print("=== dsh-routing-suite/cordis.patch.yml ===")
p = os.path.join(R, "cordis.patch.yml")
print(open(p, encoding="utf-8", errors="replace").read()[:800] if os.path.isfile(p) else "MISSING")

print("\n=== package.json ===")
pj = os.path.join(R, "package.json")
if os.path.isfile(pj):
    d = json.load(open(pj))
    for k in ("main", "exports", "keywords", "peerDependencies", "dependencies"):
        if k in d:
            print(f"{k}: {json.dumps(d[k], ensure_ascii=False)[:400]}")

print("\n=== profile cordis.yml (前50行) ===")
cy = os.path.join(PROF, "cordis.yml")
print("\n".join(open(cy, encoding="utf-8", errors="replace").read().splitlines()[:50]) if os.path.isfile(cy) else "MISSING")

print("\n=== profile cordis.patch.yml (前50行) ===")
cpy = os.path.join(PROF, "cordis.patch.yml")
print("\n".join(open(cpy, encoding="utf-8", errors="replace").read().splitlines()[:50]) if os.path.isfile(cpy) else "MISSING")

print("\n=== dsh-routing-suite README 前40行 ===")
rd = os.path.join(R, "README.md")
print("\n".join(open(rd, encoding="utf-8", errors="replace").read().splitlines()[:40]) if os.path.isfile(rd) else "MISSING")
