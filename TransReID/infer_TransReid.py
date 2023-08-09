import torch
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from config import cfg
from torchvision.transforms import functional as TF

from model import make_model



parser = argparse.ArgumentParser()
parser.add_argument('--videos', type=Path, default='videos/mot17')
parser.add_argument('--dets', type=Path, required=True)
parser.add_argument(
        "--c", default="", help="path to config file", type=str
)
args = parser.parse_args()

assert args.videos.exists()
assert args.dets.exists()

if args.c != "":
        cfg.merge_from_file(args.c)
cfg.freeze()


device = "cuda"
model = make_model(cfg, num_class=1041, camera_num=15, view_num =15)
model.load_param(cfg.TEST.WEIGHT)
model.to(device)
model.eval()


for video_paths in sorted(list(args.videos.glob('*.imgs'))):
    video_name = video_paths.stem
    paths = {ann.fid: ann.path for ann in pd.read_csv(video_paths).itertuples()}
    dets = pd.read_csv(args.dets / f'{video_name}.csv')
    embs = torch.zeros(len(dets), 3840) #3840
    check = torch.zeros(len(dets))
    print('embs shape: ', embs.shape)
    for fid, group in tqdm(dets.groupby('fid'), desc=video_name):
        image = Image.open('../' + paths[fid])

        patches = []
        for ann in group.itertuples():
            patch = image.crop((ann.x, ann.y, ann.x + ann.w, ann.y + ann.h))
            patch = patch.resize((128, 256))
            patch = TF.to_tensor(patch)
            patches.append(patch)
        patches = torch.stack(patches)
        patches = TF.normalize(patches, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        with torch.no_grad():
            
            embs[group.index.to_numpy()] = model(patches.to(device), cam_label=0, view_label=0).to('cpu')
        check[group.index.to_numpy()] += 1
    
    assert torch.all(check == 1)
    torch.save(embs, args.dets / f'{video_name}.emb')
