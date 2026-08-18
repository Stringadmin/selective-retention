import glob
import os

import cairosvg

DOCS = "/mnt/f/FIP-Transformer/docs"
jobs = [
    ("fig1_rag_vs_igm.svg", "fig1_rag_vs_igm.png"),
    ("fig2_decisive_experiment.svg", "fig2_decisive_experiment.png"),
]
for src, dst in jobs:
    s = os.path.join(DOCS, src)
    d = os.path.join(DOCS, dst)
    cairosvg.svg2png(url=s, write_to=d, scale=2.5, background_color="#1a1a18")
    print(f"{dst}: {os.path.getsize(d)//1024} KB")
print("done")
