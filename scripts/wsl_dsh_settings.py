import os, json, glob

home = os.path.expanduser("~")
print("=== settings.yaml ===")
sp = os.path.join(home, ".dsh/settings.yaml")
if os.path.isfile(sp):
    print(open(sp, encoding="utf-8", errors="replace").read()[:1500])

print("\n=== sessions 结构 ===")
sd = os.path.join(home, ".dsh/sessions")
for root, dirs, files in os.walk(sd):
    depth = root.replace(sd, "").count(os.sep)
    if depth > 2:
        dirs[:] = []
        continue
    for f in files[:8]:
        p = os.path.join(root, f)
        print(f"  {'  '*depth}{f}  ({os.path.getsize(p)}b)")

print("\n=== .dsh-logs 最新 ===")
ld = os.path.join(home, ".dsh-logs")
for f in sorted(os.listdir(ld))[-5:]:
    print("  ", f)
