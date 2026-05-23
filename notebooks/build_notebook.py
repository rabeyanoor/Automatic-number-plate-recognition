"""Build ANPR_YOLOv9_Kaggle_Colab.ipynb from main.py so the notebook always ships the current ANPR script.

    python notebooks/build_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN_PY = (HERE.parent / 'main.py').read_text()

cells = []


def md(src):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': src.strip('\n')})


def code(src):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': src.strip('\n')})


md('''
# Automatic Number Plate Recognition: YOLOv9 + EasyOCR

Trains a **YOLOv9 (GELAN-C)** license-plate detector, validates it, then reads plates with **EasyOCR** on test images and a traffic video.

Runs on **Kaggle** or **Google Colab** with any accelerator:

| Accelerator | What happens |
|---|---|
| GPU (T4 / P100) | Full training at 640 px (fastest) |
| TPU (Kaggle TPU v5e-8, Colab TPU) | YOLOv9 is plain PyTorch and can't use the TPU chips, so it trains on the TPU machine's large host CPU at 640 px |
| CPU only | 416 px with a frozen backbone so an epoch stays under ~40 min |

**Kaggle:** add the datasets `ronakgohil/license-plate-dataset` and `farzadnekouei/license-plate-recognition-for-red-light-violation`, turn Internet on, pick an accelerator, *Run All*.

**Colab:** add `KAGGLE_USERNAME` and `KAGGLE_KEY` in the 🔑 Secrets panel (the datasets are downloaded with the Kaggle API), pick a runtime, *Run all*.

Results (weights, plots, annotated images, demo video/GIF, CSVs) end up in `out/`.
''')

md('## 1. Settings')
code('''
EPOCHS = None          # None = automatic (40 on GPU, 25 on TPU host CPU, 15 on a small CPU)
SUBSET = 0             # >0: use only this many training images (quick pipeline check)
TRAIN_HOURS = None     # None = automatic (6 h, or 10 h on a small CPU); training stops early and keeps best.pt
VIDEO_SECONDS = 20     # length of the traffic clip used for the video demo
''')

md('## 2. Environment')
code('''
import glob, json, os, random, re, shutil, subprocess, sys

ON_KAGGLE = os.path.exists('/kaggle/input')
ON_COLAB = 'google.colab' in sys.modules
W = '/kaggle/working' if ON_KAGGLE else ('/content' if ON_COLAB else os.getcwd())
OUT, Y9 = f'{W}/out', f'{W}/yolov9'
os.makedirs(OUT, exist_ok=True)
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'  # yolov9 checkpoints pickle full models (torch>=2.6 default changed)


def sh(cmd, cwd=None, check=True):
    print(f'\\n$ {cmd}', flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if check and r.returncode:
        raise RuntimeError(f'command failed ({r.returncode}): {cmd}')
    return r.returncode


try:
    import torch
except ImportError:  # some TPU images ship without PyTorch
    sh('pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu')
    import torch

GPU = torch.cuda.is_available()
DEVICE = '0' if GPU else 'cpu'
CPUS = os.cpu_count()
torch.set_num_threads(CPUS)
IMG, FREEZE = 640, 0
if GPU:
    epochs, hours = 40, 6
elif CPUS >= 16:  # TPU VM host
    epochs, hours = 25, 6
else:             # plain CPU session
    IMG, FREEZE, epochs, hours = 416, 10, 15, 10
EPOCHS = EPOCHS or epochs
TRAIN_TIMEOUT = int((TRAIN_HOURS or hours) * 3600)
print(f'platform={"kaggle" if ON_KAGGLE else "colab" if ON_COLAB else "local"} torch={torch.__version__} '
      f'cuda={GPU} cpus={CPUS} | epochs={EPOCHS} img={IMG} freeze={FREEZE} timeout={TRAIN_TIMEOUT}s')
if GPU:
    sh('nvidia-smi')
''')

md('## 3. Datasets')
code('''
INPUT = '/kaggle/input'
if not ON_KAGGLE:
    # Colab / local: download the same Kaggle datasets with the Kaggle API
    if ON_COLAB:
        from google.colab import userdata
        for k in ('KAGGLE_USERNAME', 'KAGGLE_KEY'):
            os.environ[k] = userdata.get(k)
    INPUT = f'{W}/input'
    sh('pip install -q kaggle')
    for ref in ('ronakgohil/license-plate-dataset',
                'farzadnekouei/license-plate-recognition-for-red-light-violation'):
        sh(f'kaggle datasets download -q {ref} --unzip -p {INPUT}/{ref.split("/")[1]}')


def find_input(name):
    hits = glob.glob(f'{INPUT}/**/{name}', recursive=True)
    if not hits:
        raise FileNotFoundError(f'{name} not found under {INPUT}')
    return hits[0]


root = os.path.dirname(find_input('dataset.yaml'))
for split in ('train', 'val'):
    print(split, len(os.listdir(f'{root}/images/{split}')), 'images,', len(os.listdir(f'{root}/labels/{split}')), 'labels')

if SUBSET:
    # YOLO finds labels by swapping /images/ for /labels/, so mirror both trees with symlinks
    for split in ('train', 'val'):
        os.makedirs(f'{W}/subset/images/{split}', exist_ok=True)
        os.makedirs(f'{W}/subset/labels/{split}', exist_ok=True)
        for img in sorted(os.listdir(f'{root}/images/{split}'))[:SUBSET if split == 'train' else max(SUBSET // 4, 1)]:
            os.symlink(f'{root}/images/{split}/{img}', f'{W}/subset/images/{split}/{img}')
            lbl = os.path.splitext(img)[0] + '.txt'
            if os.path.exists(f'{root}/labels/{split}/{lbl}'):
                os.symlink(f'{root}/labels/{split}/{lbl}', f'{W}/subset/labels/{split}/{lbl}')
    root = f'{W}/subset'

data_yaml = f'{W}/plates.yaml'
with open(data_yaml, 'w') as f:
    f.write(f"train: {root}/images/train\\nval: {root}/images/val\\nnc: 1\\nnames: ['license_plate']\\n")
''')

md('## 4. Install YOLOv9 + EasyOCR')
code('''
sh('which ffmpeg || (apt-get -qq update && apt-get -qq install -y ffmpeg)', check=False)
if not os.path.exists(Y9):
    sh(f'git clone -q https://github.com/SkalskiP/yolov9.git {Y9}')
sh('pip install -q easyocr thop seaborn')
if not os.path.exists(f'{W}/weights/gelan-c.pt'):
    sh(f'wget -q -P {W}/weights https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt')

# Patch yolov9 for current Pillow / NumPy (removed APIs)
for f in glob.glob(f'{Y9}/**/*.py', recursive=True):
    s = open(f).read()
    n = s.replace('self.font.getsize(label)', 'self.font.getbbox(label)[2:]')
    for t in ('int', 'float', 'bool'):
        n = re.sub(rf'\\bnp\\.{t}\\b', t, n)
    if n != s:
        open(f, 'w').write(n)
        print('patched', f)
''')

md('''
## 5. ANPR script (`main.py`)

Same file as `main.py` in the repo: YOLOv9 detection, then EasyOCR on each padded, grayscale, upscaled plate crop. Saved into the YOLOv9 folder as `anpr.py`.
''')
code("%%writefile {Y9}/anpr.py\n" + MAIN_PY)

md('## 6. Train')
code('''
rc = sh(f'timeout {TRAIN_TIMEOUT} python train.py --batch 16 --epochs {EPOCHS} --img {IMG} --device {DEVICE} '
        f'--freeze {FREEZE} --min-items 0 --close-mosaic 5 --workers {min(8, CPUS)} --data {data_yaml} '
        f'--weights {W}/weights/gelan-c.pt --cfg models/detect/gelan-c.yaml --hyp data/hyps/hyp.scratch-high.yaml '
        f'--name plates --exist-ok', cwd=Y9, check=False)
print('train exit code', rc, '(124 = stopped by timeout, best.pt so far is kept)')

exp = f'{Y9}/runs/train/plates'
best = f'{exp}/weights/best.pt'
# A run stopped by the timeout leaves the optimizer state in best.pt (~4x bigger), so always strip it
sh(f'python -c "from utils.general import strip_optimizer; strip_optimizer(\\'{best}\\')"', cwd=Y9)
shutil.copy(best, f'{OUT}/best.pt')
for name in ('results.png', 'results.csv', 'confusion_matrix.png', 'F1_curve.png', 'PR_curve.png',
             'val_batch0_pred.jpg', 'val_batch0_labels.jpg', 'labels.jpg'):
    if os.path.exists(f'{exp}/{name}'):
        shutil.copy(f'{exp}/{name}', f'{OUT}/{name}')
''')
code('''
from IPython.display import Image, display
display(Image(f'{OUT}/results.png', width=1000))
''')

md('## 7. Validate')
code('''
sh(f'python val.py --img {IMG} --batch 32 --conf 0.001 --iou 0.7 --device {DEVICE} --data {data_yaml} '
   f'--weights {best} --name plates_val --exist-ok 2>&1 | tee {OUT}/val_log.txt', cwd=Y9)
display(Image(f'{OUT}/val_batch0_pred.jpg', width=900))
''')

md('## 8. ANPR on test images')
code('''
random.seed(0)
val_imgs = sorted(glob.glob(f'{root}/images/val/*'))
os.makedirs(f'{W}/samples', exist_ok=True)
for p in random.sample(val_imgs, min(12, len(val_imgs))):
    shutil.copy(p, f'{W}/samples/')
sh(f'python anpr.py --weights {best} --source {W}/samples --conf-thres 0.35 --device {DEVICE} --save-csv '
   f'--name anpr_images --exist-ok', cwd=Y9)
shutil.copytree(f'{Y9}/runs/detect/anpr_images', f'{OUT}/anpr_images', dirs_exist_ok=True)
for p in sorted(glob.glob(f'{OUT}/anpr_images/*.jpg'))[:4]:
    display(Image(p, width=600))
''')

md('## 9. ANPR on a traffic video')
code('''
video = find_input('traffic_video_original.mp4')
sh(f'ffmpeg -y -loglevel error -t {VIDEO_SECONDS} -i {video} -an -c:v libx264 -preset fast {W}/traffic_clip.mp4')
sh(f'python anpr.py --weights {best} --source {W}/traffic_clip.mp4 --conf-thres 0.3 --device {DEVICE} --save-csv '
   f'--name anpr_video --exist-ok', cwd=Y9)
raw = glob.glob(f'{Y9}/runs/detect/anpr_video/*.mp4')[0]
sh(f'ffmpeg -y -loglevel error -i {raw} -c:v libx264 -pix_fmt yuv420p -crf 26 -preset fast -movflags +faststart '
   f'-vf "scale=1280:-2" {OUT}/anpr_demo.mp4')
sh(f'ffmpeg -y -loglevel error -ss 2 -t 8 -i {raw} -vf "fps=8,scale=720:-1:flags=lanczos,split[a][b];'
   f'[a]palettegen=max_colors=128[p];[b][p]paletteuse" {OUT}/anpr_demo.gif')
shutil.copy(f'{Y9}/runs/detect/anpr_video/plates.csv', f'{OUT}/video_plates.csv')
display(Image(f'{OUT}/anpr_demo.gif'))
''')

md('## 10. Summary')
code('''
import pandas as pd

res = pd.read_csv(f'{exp}/results.csv')
res.columns = [c.strip() for c in res.columns]
vp = pd.read_csv(f'{OUT}/video_plates.csv')
summary = {
    'device': 'cuda' if GPU else f'cpu x{CPUS}',
    'epochs_planned': EPOCHS,
    'epochs_completed': int(len(res)),
    'img_size': IMG,
    'frozen_layers': FREEZE,
    'final_epoch_metrics': {k: float(v) for k, v in res.iloc[-1].items() if k.startswith('metrics')},
    'best_map50_95': float(res[[c for c in res.columns if 'mAP_0.5:0.95' in c][0]].max()),
    'video_detections': int(len(vp)),
    'video_top_plates': vp[vp.plate_text.fillna('').str.len() >= 5].plate_text.value_counts().head(15).to_dict(),
}
json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=2)
print(json.dumps(summary, indent=2))
''')
code('''
# Keep only the results (Kaggle saves everything under /kaggle/working as the notebook output)
for d in (Y9, f'{W}/weights', f'{W}/samples', f'{W}/subset', f'{W}/input'):
    shutil.rmtree(d, ignore_errors=True)
for f in glob.glob(f'{W}/*.mp4') + glob.glob(f'{W}/*.yaml'):
    os.remove(f)
if ON_COLAB:
    shutil.make_archive(f'{W}/anpr_results', 'zip', OUT)
    print('Download /content/anpr_results.zip from the Files panel')
''')

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'accelerator': 'TPU',
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}
for i, c in enumerate(nb['cells']):
    c['id'] = f'cell-{i:02d}'
    c['source'] = c['source'].splitlines(keepends=True)
out = HERE / 'ANPR_YOLOv9_Kaggle_Colab.ipynb'
out.write_text(json.dumps(nb, indent=1) + '\n')
print('wrote', out)
