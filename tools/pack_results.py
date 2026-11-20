"""Zip a results folder for sharing, keeping its structure and leaving out model weights (*.pth).

    python tools/pack_results.py results/ham10000_desc2 ../para_claude_desc2.zip
"""
import os
import sys
import zipfile

src, out = sys.argv[1], sys.argv[2]
n = 0
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(src):
        for f in files:
            if f.endswith('.pth'):
                continue
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, os.path.dirname(src.rstrip('/\\'))))
            n += 1
print('{} files -> {}'.format(n, out))
