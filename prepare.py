'''
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
'''

import copy
import json
import argparse
import pandas as pd
from PIL import Image
from pathlib import Path


def load_odgt(odgt_path):
    img_dir = odgt_path.parent / 'Images'
    img_anns = []
    box_anns = []
    with open(odgt_path) as f:
        data = [json.loads(line) for line in f]
    for datum in data:
        img_id = datum['ID']
        img_path = str(img_dir / img_id) + '.jpg'
        imgW, imgH = Image.open(img_path).size

        fboxes = []
        vboxes = []
        for inst in datum['gtboxes']:
            fx, fy, fw, fh = inst['fbox']
            vx, vy, vw, vh = inst['vbox']
            if inst['tag'] != 'person':
                continue
            fboxes.append((fx, fy, fw, fh))
            vboxes.append((vx, vy, vw, vh))

        if len(fboxes) > 1:
            img_anns.append(
                {
                    'id': img_id,
                    'file_name': img_path,
                    'width': imgW,
                    'height': imgH,
                    'video': 'crowdhuman',
                }
            )
            for box in fboxes:
                box_anns.append(
                    {
                        'id': len(box_anns),
                        'image_id': img_id,
                        'category_id': 1,
                        'bbox': box,
                    }
                )

    return {
        'images': img_anns,
        'annotations': box_anns,
        'categories': [{'id': 1, 'name': 'person'}],
    }


def df2coco(df, video_dir):
    df = df[df['class'] == 1]
    df = df[df['vis'] > 0]

    images = []
    annotations = []
    imgW, imgH = Image.open(video_dir / 'img1' / '000001.jpg').size

    for (fid, group) in df.groupby('fid'):
        image_id = f'{video_dir.stem}-{fid}'
        images.append(
            {
                'id': image_id,
                'file_name': str(video_dir / 'img1' / f'{fid:06d}.jpg'),
                'width': imgW,
                'height': imgH,
                'video': video_dir.stem,
            }
        )

        data = group[['x', 'y', 'w', 'h', 'tag', 'vis']].values
        for (x, y, w, h, tag, vis) in data.tolist():
            annotations.append(
                {
                    'id': f'{video_dir.stem}-{len(annotations)}',
                    'image_id': image_id,
                    'category_id': 1,
                    'bbox': (x, y, w, h),
                    'vis': vis,
                    'area': w * h,
                    'iscrowd': 0,
                }
            )
    return images, annotations


def normalize_ids(data):
    data = copy.deepcopy(data)

    image_ids = [ann['id'] for ann in data['images']]
    mapping = {x: i for i, x in enumerate(image_ids)}
    for img_ann in data['images']:
        img_ann['id'] = mapping[img_ann['id']]
    for box_ann in data['annotations']:
        box_ann['image_id'] = mapping[box_ann['image_id']]

    box_ids = [ann['id'] for ann in data['annotations']]
    mapping = {x: i for i, x in enumerate(box_ids)}
    for box_ann in data['annotations']:
        box_ann['id'] = mapping[box_ann['id']]

    assert len(set([ann['id'] for ann in data['images']])) == len(data['images'])
    assert len(set([ann['id'] for ann in data['annotations']])) == len(data['annotations'])
    assert len(set([ann['image_id'] for ann in data['annotations']])) == len(data['images'])

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--ch', type=Path, default='/store_2/CrowdHuman/')
    # parser.add_argument('--mot17', type=Path, default='/store/MOT17/')
    # parser.add_argument('--mot20', type=Path, default='/store/MOT20/')
    parser.add_argument('--mot16', type=Path, default='/MOT16/')
    parser.add_argument('--jsons', type=Path, default='./dataset/jsons')
    parser.add_argument('--videos', type=Path, default='./dataset/videos/')
    args = parser.parse_args()

    # assert args.ch.exists()
    # assert args.mot17.exists()
    # assert args.mot20.exists()
    assert args.mot16.exists()


    # Prepare jsons
    #args.jsons.mkdir(parents=True)



    mot16_data = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'person'}],
    }
    for gt_path in args.mot16.glob('**/gt.txt'):
        df = pd.read_csv(gt_path, header=None)
        df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'use', 'class', 'vis']
        imgs, anns = df2coco(df, gt_path.parent.parent)
        mot16_data['images'].extend(imgs)
        mot16_data['annotations'].extend(anns)
    with open(args.jsons / 'mot16.json', 'w') as f:
        json.dump(normalize_ids(mot16_data), f)
    print('mot16: {} {}'.format(len(mot16_data['images']), len(mot16_data['annotations'])))



    #ch_data = load_odgt(args.ch / 'annotation_train.odgt')
    # with open(args.jsons / 'ch.json', 'w') as f:
    #     json.dump(normalize_ids(ch_data), f)
    # print('ch: {} {}'.format(len(ch_data['images']), len(ch_data['annotations'])))

    # mot17_data = {
    #     'images': [],
    #     'annotations': [],
    #     'categories': [{'id': 1, 'name': 'person'}],
    # }
    # for gt_path in args.mot17.glob('**/*-DPM/gt/gt.txt'):
    #     df = pd.read_csv(gt_path, header=None)
    #     df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'use', 'class', 'vis']
    #     imgs, anns = df2coco(df, gt_path.parent.parent)
    #     mot17_data['images'].extend(imgs)
    #     mot17_data['annotations'].extend(anns)
    # with open(args.jsons / 'mot17.json', 'w') as f:
    #     json.dump(normalize_ids(mot17_data), f)
    # print('mot17: {} {}'.format(len(mot17_data['images']), len(mot17_data['annotations'])))

    # mot20_data = {
    #     'images': [],
    #     'annotations': [],
    #     'categories': [{'id': 1, 'name': 'person'}],
    # }
    # for gt_path in args.mot20.glob('**/gt.txt'):
    #     df = pd.read_csv(gt_path, header=None)
    #     df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'use', 'class', 'vis']
    #     imgs, anns = df2coco(df, gt_path.parent.parent)
    #     mot20_data['images'].extend(imgs)
    #     mot20_data['annotations'].extend(anns)
    # with open(args.jsons / 'mot20.json', 'w') as f:
    #     json.dump(normalize_ids(mot20_data), f)
    # print('mot20: {} {}'.format(len(mot20_data['images']), len(mot20_data['annotations'])))

    # ch17 = {
    #     'images': [*ch_data['images'], *mot17_data['images']],
    #     'annotations': [*ch_data['annotations'], *mot17_data['annotations']],
    #     'categories': [{'id': 1, 'name': 'person'}],
    # }
    # with open(args.jsons / 'ch17.json', 'w') as f:
    #     json.dump(normalize_ids(ch17), f)
    # print('ch17: {} {}'.format(len(ch17['images']), len(ch17['annotations'])))

    # ch20 = {
    #     'images': [*ch_data['images'], *mot20_data['images']],
    #     'annotations': [*ch_data['annotations'], *mot20_data['annotations']],
    #     'categories': [{'id': 1, 'name': 'person'}],
    # }
    # with open(args.jsons / 'ch20.json', 'w') as f:
    #     json.dump(normalize_ids(ch20), f)
    # print('ch20: {} {}'.format(len(ch20['images']), len(ch20['annotations'])))

    # Prepare videos
    #args.videos.mkdir(parents=True)
    # (args.videos / 'mot17').mkdir()
    # (args.videos / 'mot20').mkdir()
    (args.videos / 'mot16').mkdir()

    # for video_dir in args.mot17.glob('*/*-DPM'):
    #     video_name = video_dir.stem[:-4]
    #     paths = sorted(list(video_dir.glob('img1/*.jpg')))
    #     fids = [int(p.stem) for p in paths]
    #     df_imgs = pd.DataFrame({'fid': fids, 'path': paths})
    #     df_imgs.to_csv(args.videos / 'mot17' / f'{video_name}.imgs', index=None)
    #     if (video_dir / 'gt' / 'gt.txt').exists():
    #         df_true = pd.read_csv(video_dir / 'gt' / 'gt.txt', header=None)
    #         df_true.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'use', 'class', 'vis']
    #         df_true.to_csv(args.videos / 'mot17' / f'{video_name}.gt', index=None)

    # for video_dir in args.mot20.glob('*/*'):
    #     video_name = video_dir.stem
    #     paths = sorted(list(video_dir.glob('img1/*.jpg')))
    #     fids = [int(p.stem) for p in paths]
    #     df_imgs = pd.DataFrame({'fid': fids, 'path': paths})
    #     df_imgs.to_csv(args.videos / 'mot20' / f'{video_name}.imgs', index=None)
    #     if (video_dir / 'gt' / 'gt.txt').exists():
    #         df_true = pd.read_csv(video_dir / 'gt' / 'gt.txt', header=None)
    #         df_true.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'use', 'class', 'vis']
    #         df_true.to_csv(args.videos / 'mot20' / f'{video_name}.gt', index=None)


    for video_dir in args.mot16.glob('*/*'):
        video_name = video_dir.stem
        paths = sorted(list(video_dir.glob('img1/*.jpg')))
        fids = [int(p.stem) for p in paths]
        df_imgs = pd.DataFrame({'fid': fids, 'path': paths})
        df_imgs.to_csv(args.videos / 'mot16' / f'{video_name}.imgs', index=None)
        if (video_dir / 'gt' / 'gt.txt').exists():
            df_true = pd.read_csv(video_dir / 'gt' / 'gt.txt', header=None)
            df_true.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'use', 'class', 'vis']
            df_true.to_csv(args.videos / 'mot16' / f'{video_name}.gt', index=None)
