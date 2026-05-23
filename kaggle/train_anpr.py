"""Kaggle GPU pipeline: train YOLOv9 (GELAN-C) on license plates, validate, then run main.py (YOLOv9 + EasyOCR)
on test images and a traffic video. Everything lands in /kaggle/working/out.

Inputs (Kaggle datasets): ronakgohil/license-plate-dataset, farzadnekouei/license-plate-recognition-for-red-light-violation
"""
import base64
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys

EPOCHS = int(os.environ.get('EPOCHS', 40))
SUBSET = int(os.environ.get('SUBSET', 0))  # >0: train/val on this many images only (quick pipeline check)
TRAIN_TIMEOUT = int(os.environ.get('TRAIN_TIMEOUT', 6 * 3600))  # stop early and keep best.pt so the session limit is never hit
MAIN_PY_B64 = '__MAIN_PY_B64__'  # main.py from the repo, embedded by kaggle/push.sh

W = '/kaggle/working'
OUT = f'{W}/out'
Y9 = f'{W}/yolov9'
os.makedirs(OUT, exist_ok=True)
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'  # yolov9 checkpoints pickle full models (torch>=2.6 default changed)


def sh(cmd, cwd=None, check=True):
    print(f'\n$ {cmd}', flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if check and r.returncode:
        raise SystemExit(f'command failed ({r.returncode}): {cmd}')
    return r.returncode


def find_input(name):
    hits = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    if not hits:
        raise SystemExit(f'{name} not found under /kaggle/input')
    return hits[0]


# ---------------------------------------------------------------- setup
# Works on a GPU session or a TPU-VM session (YOLOv9 is plain PyTorch, so on TPU it trains on the large host CPU)
try:
    import torch
except ImportError:
    sh('pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu')
    import torch
GPU = torch.cuda.is_available()
DEVICE = '0' if GPU else 'cpu'
CPUS = os.cpu_count()
torch.set_num_threads(CPUS)
print(f'torch {torch.__version__}, cuda={GPU}, cpus={CPUS}', flush=True)
if GPU:
    sh('nvidia-smi')
IMG, FREEZE = 640, 0
if not GPU:
    EPOCHS = int(os.environ.get('EPOCHS', 25))
if not GPU and CPUS < 16:
    # Plain Kaggle CPU session (4 cores): smaller images + frozen GELAN backbone keep an epoch under ~40 min
    IMG, FREEZE = 416, 10
    EPOCHS = int(os.environ.get('EPOCHS', 15))
    TRAIN_TIMEOUT = int(os.environ.get('TRAIN_TIMEOUT', 10 * 3600))
print(f'epochs={EPOCHS} img={IMG} freeze={FREEZE} timeout={TRAIN_TIMEOUT}s', flush=True)
sh('which ffmpeg || (apt-get -qq update && apt-get -qq install -y ffmpeg)', check=False)
sh(f'git clone -q https://github.com/SkalskiP/yolov9.git {Y9}')
sh('pip install -q easyocr thop seaborn')
sh(f'wget -q -P {W}/weights https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt')

# Patch yolov9 for current Pillow / NumPy (removed APIs)
for f in glob.glob(f'{Y9}/**/*.py', recursive=True):
    s = open(f).read()
    n = s.replace('self.font.getsize(label)', 'self.font.getbbox(label)[2:]')
    n = re.sub(r'\bnp\.int\b', 'int', n)
    n = re.sub(r'\bnp\.float\b', 'float', n)
    n = re.sub(r'\bnp\.bool\b', 'bool', n)
    if n != s:
        open(f, 'w').write(n)
        print('patched', f)

with open(f'{Y9}/anpr.py', 'wb') as f:
    f.write(base64.b64decode(MAIN_PY_B64))

# ---------------------------------------------------------------- data
ds_yaml = find_input('dataset.yaml')
root = os.path.dirname(ds_yaml)
print(open(ds_yaml).read())
for split in ('train', 'val'):
    print(split, len(os.listdir(f'{root}/images/{split}')), 'images,',
          len(os.listdir(f'{root}/labels/{split}')) if os.path.isdir(f'{root}/labels/{split}') else 0, 'labels')
if SUBSET:
    # YOLO finds labels by swapping /images/ for /labels/, so mirror both trees with symlinks
    for split in ('train', 'val'):
        os.makedirs(f'{W}/subset/images/{split}', exist_ok=True)
        os.makedirs(f'{W}/subset/labels/{split}', exist_ok=True)
        for img in sorted(os.listdir(f'{root}/images/{split}'))[:SUBSET if split == 'train' else SUBSET // 4]:
            os.symlink(f'{root}/images/{split}/{img}', f'{W}/subset/images/{split}/{img}')
            lbl = os.path.splitext(img)[0] + '.txt'
            if os.path.exists(f'{root}/labels/{split}/{lbl}'):
                os.symlink(f'{root}/labels/{split}/{lbl}', f'{W}/subset/labels/{split}/{lbl}')
    root_full, root = root, f'{W}/subset'
data_yaml = f'{W}/plates.yaml'
with open(data_yaml, 'w') as f:
    f.write(f"train: {root}/images/train\nval: {root}/images/val\nnc: 1\nnames: ['license_plate']\n")

# ---------------------------------------------------------------- train
rc = sh(f'timeout {TRAIN_TIMEOUT} python train.py --batch 16 --epochs {EPOCHS} --img {IMG} --device {DEVICE} '
   f'--freeze {FREEZE} '
   f'--min-items 0 --close-mosaic 5 --workers {min(8, CPUS)} --data {data_yaml} --weights {W}/weights/gelan-c.pt --cfg models/detect/gelan-c.yaml '
   f'--hyp data/hyps/hyp.scratch-high.yaml --name plates --exist-ok', cwd=Y9, check=False)
print('train exit code', rc, '(124 = stopped by timeout, best.pt so far is kept)')
exp = f'{Y9}/runs/train/plates'
best = f'{exp}/weights/best.pt'
# A run stopped by the timeout leaves the optimizer state in best.pt (~4x bigger), so always strip it
sh(f'python -c "from utils.general import strip_optimizer; strip_optimizer(\'{best}\')"', cwd=Y9)
shutil.copy(best, f'{OUT}/best.pt')
for name in ('results.png', 'results.csv', 'confusion_matrix.png', 'F1_curve.png', 'PR_curve.png',
             'val_batch0_pred.jpg', 'val_batch0_labels.jpg', 'labels.jpg'):
    if os.path.exists(f'{exp}/{name}'):
        shutil.copy(f'{exp}/{name}', f'{OUT}/{name}')

# ---------------------------------------------------------------- validate
sh(f'python val.py --img {IMG} --batch 32 --conf 0.001 --iou 0.7 --device {DEVICE} --data {data_yaml} '
   f'--weights {best} --name plates_val --exist-ok 2>&1 | tee {OUT}/val_log.txt', cwd=Y9)

# ---------------------------------------------------------------- ANPR on images
random.seed(0)
val_imgs = sorted(glob.glob(f'{root}/images/val/*'))
os.makedirs(f'{W}/samples', exist_ok=True)
for p in random.sample(val_imgs, min(12, len(val_imgs))):
    shutil.copy(p, f'{W}/samples/')
sh(f'python anpr.py --weights {best} --source {W}/samples --conf-thres 0.35 --device {DEVICE} --save-csv '
   f'--name anpr_images --exist-ok', cwd=Y9)
shutil.copytree(f'{Y9}/runs/detect/anpr_images', f'{OUT}/anpr_images', dirs_exist_ok=True)

# ---------------------------------------------------------------- ANPR on video
video = find_input('traffic_video_original.mp4')
sh(f'ffmpeg -y -loglevel error -ss 0 -t 20 -i {video} -an -c:v libx264 -preset fast {W}/traffic_20s.mp4')
sh(f'python anpr.py --weights {best} --source {W}/traffic_20s.mp4 --conf-thres 0.3 --device {DEVICE} --save-csv '
   f'--name anpr_video --exist-ok', cwd=Y9)
raw = glob.glob(f'{Y9}/runs/detect/anpr_video/*.mp4')[0]
sh(f'ffmpeg -y -loglevel error -i {raw} -c:v libx264 -pix_fmt yuv420p -crf 26 -preset fast -movflags +faststart '
   f'-vf "scale=1280:-2" {OUT}/anpr_demo.mp4')
sh(f'ffmpeg -y -loglevel error -ss 2 -t 8 -i {raw} -vf "fps=8,scale=720:-1:flags=lanczos,split[a][b];'
   f'[a]palettegen=max_colors=128[p];[b][p]paletteuse" {OUT}/anpr_demo.gif')
shutil.copy(f'{Y9}/runs/detect/anpr_video/plates.csv', f'{OUT}/video_plates.csv')

# ---------------------------------------------------------------- summary
import pandas as pd

res = pd.read_csv(f'{exp}/results.csv')
res.columns = [c.strip() for c in res.columns]
last = res.iloc[-1].to_dict()
vp = pd.read_csv(f'{OUT}/video_plates.csv')
summary = {
    'device': 'cuda' if GPU else f'cpu x{CPUS}',
    'epochs_planned': EPOCHS,
    'epochs_completed': int(len(res)),
    'img_size': IMG,
    'frozen_layers': FREEZE,
    'final_epoch_metrics': {k: float(v) for k, v in last.items() if k.startswith('metrics')},
    'best_map50_95': float(res[[c for c in res.columns if 'mAP_0.5:0.95' in c][0]].max()),
    'video_detections': int(len(vp)),
    'video_top_plates': vp[vp.plate_text.fillna('').str.len() >= 5].plate_text.value_counts().head(15).to_dict(),
}
json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=2)
print(json.dumps(summary, indent=2))
sh(f'ls -la {OUT} {OUT}/anpr_images; du -sh {OUT}')

# Kaggle saves everything under /kaggle/working as output; keep only the results
for d in (Y9, f'{W}/weights', f'{W}/samples', f'{W}/subset'):
    shutil.rmtree(d, ignore_errors=True)
for f in glob.glob(f'{W}/*.mp4') + glob.glob(f'{W}/*.yaml'):
    os.remove(f)
