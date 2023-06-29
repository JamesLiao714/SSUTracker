# SSUTracker

## Install

1. pytorch & torchvision
2. [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
3. [torchreid](https://github.com/KaiyangZhou/deep-person-reid)
4. `pip install -r requirements.txt`

## Download data for pedestrian tracking

Download from [MOT Challenge](https://motchallenge.net/data/MOT17/) and [CrowdHuman](https://www.crowdhuman.org/)

## Pretrained Weights for pedestrian tracking

It should be available at [here](https://github.com/JamesLiao714/SSUTracker/releases).

1. `p300_ch.ckpt`: 300 proposal bounding boxes, trained on CrowdHuman
2. `p300_coco.ckpt`: 300 proposal bounding boxes, trained on COCO
3. `p300_ch17.ckpt`: 300 proposal bounding boxes, trained on CrowdHuman + MOT17
4. `p500_ch20.ckpt`: 500 proposal bounding boxes, trained on CrowdHuman + MOT20

Place them under `weights/`.

## Dataset

You can add other dataset such as car, vehicles, etc. by adding it into ```prepare.py```.

For example, 

```
python prepare.py --ch path/to/crowdhuman --mot17 /path/to/mot17 --mot20 /path/to/mot20
```

After exectuion, you will get

```
jsons/
    ch.json
    mot17.json
    mot20.json
    ch17.json
    ch20.json
videos/
    mot17/
        MOT17-01.imgs
        MOT17-02.gt
        MOT17-02.imgs
    mot20/
        MOT20-01.gt
        MOT20-01.imgs
        MOT20-02.gt
        MOT20-02.imgs
```

where `ch.json, ch17.json, ch20.json` are the COCO-format of annotation of CrowdHuman, MOT17, MOT20, CrowdHuman + MOT17, CrowdHuman + MOT20, respectively. `*.imgs` are csv files indicating the path of each frame. `*.gt` are csv files indicating the tracking ground truth of each video.


## Object Detector Training (Optional)


For example, train the Sparse R-CNN with SU on CrowdHuman and validate on MOT17:

```
python train_detector.py --train_json jsons/ch.json --valid_json jsons/mot17.json --num_proposals 300
```

By default, pretrained weight `weights/p300_coco.pth` is used. Refer to `train.py` for more detail.


## Object Detector Inference


Infer detection using model weight `weights/p300_ch17.ckpt` on all MOT17 videos:

```
python infer_dets.py --videos dataset/videos/mot17 --ckpt weights/p300_ch17.ckpt --dets dets/mot17 --score_thresh 0.4 --nms_thresh 0.7  --num_proposals 300
```

The detections are saved at `dets/mot17`.

Extract appearance embedding  on all MOT17 videos given detections `dets/mot17`:

```
python infer_reid_pedestrain.py --videos dataset/videos/mot17 --dets dets/mot17
```

For vehicles reid, please refer to [this](https://github.com/LCFractal/AIC21-MTMC).


Extracted embeddings are saved at save directory (i.e. `dets/mot17`).


## Tracking

```
python track17.py --videos dataset/videos/mot17 --dets dets/mot17 --outs outs/mot17
```

Tracking results is saved at `outs/mot17`.
