import os

AP = os.path.expanduser("~/.dsh/.agent-presets/routing-suite")

print("=== preset.yml ===")
p = os.path.join(AP, "preset.yml")
if os.path.isfile(p):
    print(open(p, encoding="utf-8", errors="replace").read()[:2500])

print("\n=== agent.cordis.yml (前80行) ===")
a = os.path.join(AP, "agent.cordis.yml")
if os.path.isfile(a):
    lines = open(a, encoding="utf-8", errors="replace").read().splitlines()
    print("\n".join(lines[:80]))
