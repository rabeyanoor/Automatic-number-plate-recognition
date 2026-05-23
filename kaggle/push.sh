#!/usr/bin/env bash
# Build the Kaggle kernel (train_anpr.py with main.py embedded) and push it to run on a Kaggle GPU.
# Needs KAGGLE_USERNAME / KAGGLE_KEY in the environment (or ~/.kaggle/kaggle.json).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="${BUILD_DIR:-$(mktemp -d)}"
mkdir -p "$BUILD"
python3 - "$HERE" "$BUILD" <<'PY'
import base64, sys
here, build = sys.argv[1:]
src = open(f'{here}/train_anpr.py').read()
b64 = base64.b64encode(open(f'{here}/../main.py', 'rb').read()).decode()
open(f'{build}/train_anpr.py', 'w').write(src.replace('__MAIN_PY_B64__', b64))
PY
sed "s/__USER__/${KAGGLE_USERNAME}/" "$HERE/kernel-metadata.json" > "$BUILD/kernel-metadata.json"
kaggle kernels push -p "$BUILD"
echo "Built in $BUILD"
