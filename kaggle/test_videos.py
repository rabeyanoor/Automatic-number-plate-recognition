"""Kaggle check of the trained model on unseen videos: runs main.py (YOLOv9 + EasyOCR) on the first seconds of
every video in /kaggle/input and saves annotated clips, GIF previews and plate readings to /kaggle/working/out.

Inputs: the training kernel's output (best.pt) and nimishshandilya/car-number-plate-video
"""
import base64
import glob
import json
import os
import re
import shutil
import subprocess

SECONDS = 25        # length of each test clip
VID_STRIDE = 2      # process every 2nd frame to keep a CPU session fast
MAIN_PY_B64 = '__MAIN_PY_B64__'  # main.py from the repo, embedded by kaggle/push.sh

W = '/kaggle/working'
OUT = f'{W}/out'
Y9 = f'{W}/yolov9'
os.makedirs(OUT, exist_ok=True)
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'


def sh(cmd, cwd=None, check=True):
    print(f'\n$ {cmd}', flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if check and r.returncode:
        raise SystemExit(f'command failed ({r.returncode}): {cmd}')
    return r.returncode


import torch

DEVICE = '0' if torch.cuda.is_available() else 'cpu'
sh(f'git clone -q https://github.com/SkalskiP/yolov9.git {Y9}')
sh('pip install -q easyocr thop seaborn')
for f in glob.glob(f'{Y9}/**/*.py', recursive=True):
    s = open(f).read()
    n = s.replace('self.font.getsize(label)', 'self.font.getbbox(label)[2:]')
    for t in ('int', 'float', 'bool'):
        n = re.sub(rf'\bnp\.{t}\b', t, n)
    if n != s:
        open(f, 'w').write(n)
with open(f'{Y9}/anpr.py', 'wb') as f:
    f.write(base64.b64decode(MAIN_PY_B64))

best = glob.glob('/kaggle/input/**/best.pt', recursive=True)[0]
videos = sorted(v for v in glob.glob('/kaggle/input/**/*.mp4', recursive=True)
                if 'anpr_demo' not in v and '2160p' not in v)  # skip the training demo and 4K duplicates
print('weights:', best)
print('videos:', *videos, sep='\n  ')

import pandas as pd

report = {}
for i, video in enumerate(videos):
    name = f'test{i + 1}'
    clip = f'{W}/{name}.mp4'
    # 1280 px wide is plenty for plates and keeps decoding cheap
    sh(f'ffmpeg -y -loglevel error -t {SECONDS} -i "{video}" -an -vf "scale=1280:-2" -c:v libx264 -preset fast {clip}')
    sh(f'python anpr.py --weights {best} --source {clip} --conf-thres 0.3 --device {DEVICE} --vid-stride {VID_STRIDE} '
       f'--save-csv --name {name} --exist-ok', cwd=Y9)
    raw = f'{Y9}/runs/detect/{name}/{name}.mp4'
    # main.py writes kept frames at the source fps, so slow the clip back down to real time
    sh(f'ffmpeg -y -loglevel error -i {raw} -vf "setpts={VID_STRIDE}*PTS" -c:v libx264 -pix_fmt yuv420p -crf 27 -preset fast -movflags +faststart '
       f'{OUT}/{name}_anpr.mp4')
    shutil.copy(f'{Y9}/runs/detect/{name}/plates.csv', f'{OUT}/{name}_plates.csv')

    df = pd.read_csv(f'{OUT}/{name}_plates.csv')
    df['plate_text'] = df.plate_text.fillna('')
    frames = int(subprocess.run(f'ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames '
                                f'-of csv=p=0 {clip}', shell=True, capture_output=True, text=True).stdout.strip() or 0)
    # Preview GIF and still around the busiest second of the clip
    busy = int(df.frame.value_counts().idxmax()) if len(df) else 0  # index into the written (strided) video
    t0 = max(0.0, busy / (frames / SECONDS) - 2) if frames else 0
    sh(f'ffmpeg -y -loglevel error -ss {t0:.2f} -t 5 -i {raw} -vf "fps=6,scale=640:-1:flags=lanczos,split[a][b];'
       f'[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" {OUT}/{name}_preview.gif')
    sh(f'ffmpeg -y -loglevel error -ss {t0 + 2:.2f} -i {raw} -frames:v 1 -vf scale=960:-1 -q:v 3 {OUT}/{name}_frame.jpg')
    readable = df[df.plate_text.str.len() >= 4]
    report[name] = {
        'source_file': os.path.basename(video),
        'frames_in_clip': frames,
        'plate_detections': int(len(df)),
        'frames_with_plate': int(df.frame.nunique()),
        'mean_det_conf': round(float(df.det_conf.mean()), 3) if len(df) else None,
        'readings_4plus_chars': int(len(readable)),
        'top_readings': readable.plate_text.value_counts().head(10).to_dict(),
    }
    print(json.dumps(report[name], indent=2), flush=True)

json.dump(report, open(f'{OUT}/video_test_report.json', 'w'), indent=2)
print(json.dumps(report, indent=2))
for d in (Y9,):
    shutil.rmtree(d, ignore_errors=True)
for f in glob.glob(f'{W}/*.mp4'):
    os.remove(f)
