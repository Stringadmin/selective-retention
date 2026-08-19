import json
import os

D = "/home/omnichat/node/node-linux/lib/node_modules/@deepseek-ai/dsh/node_modules/@deepseek-ai"

print("=== dsh 自带的插件包 ===")
if os.path.isdir(D):
    for x in sorted(os.listdir(D)):
        if x.startswith("dsh-"):
            print("  ", x)

print("\n=== 找 tool 相关插件 ===")
for x in sorted(os.listdir(D)):
    if "tool" in x.lower():
        pj = os.path.join(D, x, "package.json")
        if os.path.isfile(pj):
            d = json.load(open(pj))
            print(f"\n--- {x} ---")
            print("  main:", d.get("main"))
            print("  keywords:", d.get("keywords"))
