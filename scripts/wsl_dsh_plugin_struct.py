import json, os, glob

PROF = os.path.expanduser("~/.dsh/profiles/web")
NM = os.path.join(PROF, "node_modules")

def pkg_info(pkg_dir):
    pj = os.path.join(pkg_dir, "package.json")
    if not os.path.isfile(pj):
        return None
    with open(pj) as f:
        p = json.load(f)
    return p

print("=== web profile 已装插件包 ===")
for name in ("dsh-plugin-marketplace", "dsh-routing-suite"):
    d = os.path.join(NM, name)
    if not os.path.isdir(d):
        print(f"  {name}: NOT in node_modules")
        continue
    p = pkg_info(d)
    print(f"\n--- {name} ---")
    if p:
        print("  main:", p.get("main"), "| exports:", json.dumps(p.get("exports"))[:150])
        print("  keywords:", p.get("keywords"))
    # find entry source files
    for pat in ("lib/**/*.js", "src/**/*.ts", "dist/**/*.js", "*.js", "lib/*.js"):
        fs = glob.glob(os.path.join(d, pat), recursive=True)
        if fs:
            print("  files:", [os.path.relpath(x, d) for x in fs[:6]])
            break
