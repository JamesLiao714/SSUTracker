import torch
import numpy as np
import pandas as pd
import argparse
import lapsolver
from collections import namedtuple
from tqdm import tqdm
from pathlib import Path
from torchvision.ops import box_convert, box_iou
from utils import MOTEvaluator, cosine_similarity, fix_invalid_bbox, camera_compensate
from filterpy import kalman


Detection = namedtuple('Detection', ['bbox', 'emb', 'score', 'unc'])

UNC = True

class Track:
    F = np.array(
        [
            [1, 0, 0, 0, 1, 0, 0, 0],  # x1
            [0, 1, 0, 0, 0, 1, 0, 0],  # y1
            [0, 0, 1, 0, 0, 0, 1, 0],  # x2
            [0, 0, 0, 1, 0, 0, 0, 1],  # y2
            [0, 0, 0, 0, 1, 0, 0, 0],  # v_x1
            [0, 0, 0, 0, 0, 1, 0, 0],  # v_y1
            [0, 0, 0, 0, 0, 0, 1, 0],  # v_x2
            [0, 0, 0, 0, 0, 0, 0, 1],  # v_y2
        ]
    )  # state transition

    Q = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],  # x1
            [0, 1, 0, 0, 0, 0, 0, 0],  # y1
            [0, 0, 1, 0, 0, 0, 0, 0],  # x2
            [0, 0, 0, 1, 0, 0, 0, 0],  # y2
            [0, 0, 0, 0, 1e-2, 0, 0, 0],  # v_x1
            [0, 0, 0, 0, 0, 1e-2, 0, 0],  # v_y1
            [0, 0, 0, 0, 0, 0, 1e-2, 0],  # v_x2
            [0, 0, 0, 0, 0, 0, 0, 1e-2],  # v_y2
        ]
    )  # process noise uncertainty

    H = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )  # convert x to z

    MSR = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)  # convert x to z

    def __init__(
        self, track_id: int, det: Detection, n_init: int, max_age: int, window: int
    ):
        #print(det)
        self.track_id = track_id
        self.n_init = n_init
        self.max_age = max_age
        self.window = window
        self.history = []
        self.history.append(det)
        # calculate first normalized uncertainty

        wh = det.bbox[2:] - det.bbox[:2]
        self.unc = (det.unc / torch.cat([wh, wh])).sum()
        self.avg_su = self.unc       
        
        self.t_emb = self.history[-1].emb
        # ##un
        self.ul = det.unc[0]
        self.ut = det.unc[1]
        self.ur = det.unc[2]
        self.ub = det.unc[3]


        self.state = 'tentative'
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.x = np.zeros((8, 1), dtype=np.float32)
        self.x[:4, 0] = det.bbox.numpy()
        self.P = np.array(
            [
                [10, 0, 0, 0, 0, 0, 0, 0],  # x1
                [0, 10, 0, 0, 0, 0, 0, 0],  # y1
                [0, 0, 10, 0, 0, 0, 0, 0],  # x2
                [0, 0, 0, 10, 0, 0, 0, 0],  # y2
                [0, 0, 0, 0, 1e4, 0, 0, 0],  # v_x1
                [0, 0, 0, 0, 0, 1e4, 0, 0],  # v_y1
                [0, 0, 0, 0, 0, 0, 1e4, 0],  # v_x2
                [0, 0, 0, 0, 0, 0, 0, 1e4],  # v_y2
            ]
        )  # state covariance matrix

    @property
    def bbox(self):
        bbox = torch.from_numpy(self.x[:4, 0]).float()
        return fix_invalid_bbox(bbox)

    @property
    def emb(self):
        uncs = []
        embs = []
        if UNC:

            # # for det in self.history[-self.window:]:
            # #     wh = det.bbox[2:] - det.bbox[:2]
            # #     unc = (det.unc / torch.cat([wh, wh])).sum()
            # #     uncs.append(unc)
            # #     embs.append(det.emb)
            
            # #uncs = torch.tensor(uncs)
            # #embs = torch.stack(embs)
            # # embedding fusion with last seen 
            # #emb = embs[uncs.argmin()] + self.history[-1].emb
            # emb = 0.9*self.emb + 0.1*self.history[-1].emb
            return self.t_emb
        else:
            return self.history[-1].emb
        
     

    @property
    def score(self):
        return self.history[-1].score

    def predict(self):
        self.x, self.P = kalman.predict(self.x, self.P, Track.F, Track.Q)
        self.age += 1
        self.time_since_update += 1

    def update(self, det):
        if UNC:
            self.x, self.P = kalman.update(
                self.x, self.P, det.bbox.numpy(), np.diag(det.unc), Track.H
            )
        else:
            self.x, self.P = kalman.update(
                self.x, self.P, det.bbox.numpy(), self.MSR, Track.H
            )
        self.history.append(det)
        
        #print("best: ", self.best_unc)
        wh = det.bbox[2:] - det.bbox[:2]
        wh = det.bbox[2:] - det.bbox[:2]
        su = (det.unc / torch.cat([wh, wh])).sum()
        if su <= self.avg_su:
            self.t_emb = 0.2* self.t_emb + 0.8* det.emb
            self.avg_su =(self.avg_su + su)/ len(self.history)
        
        self.hits += 1
        self.time_since_update = 0
        if self.state == 'tentative' and self.hits >= self.n_init:
            self.state = 'confirmed'

    def mark_missed(self):
        if self.state == 'tentative':
            self.state = 'deleted'
        elif self.time_since_update > self.max_age:
            self.state = 'deleted'


class Tracker:
    def __init__(self, max_age=30, n_init=3, window=60):
        self.max_age = max_age
        self.n_init = n_init
        self.window = window
        self.tracks = []
        self.track_id = 0

    def step(self, detections):
        for track in self.tracks:
            track.predict()

        confirmed_tracks = [
            track for track in self.tracks if track.state == 'confirmed'
        ]
        remaining_tracks = [
            track for track in self.tracks if track.state != 'confirmed'
        ] # tracks that are not matched in appearanace and tracks that are tentative

        if len(confirmed_tracks) > 0 and len(detections) > 0:
            confirmed_boxes = torch.stack([track.bbox for track in confirmed_tracks])
            confirmed_embs = torch.stack([track.emb for track in confirmed_tracks])
            detected_boxes = torch.stack([det.bbox for det in detections])
            detected_embs = torch.stack([det.emb for det in detections])
            iou_sim = box_iou(confirmed_boxes, detected_boxes)
            app_sim = cosine_similarity(confirmed_embs, detected_embs)
            iou_sim[iou_sim < 0.2] = np.nan

            #fusion 

            # con_app = app_sim < 0.25
            # #print(con_app)
            # #print(app_sim)
            # con_mot = iou_sim < 0.5
            # comb = con_app & con_mot
            # app_sim[comb] = 0.5
            # app_sim[~comb] = 1
            # #print(app_sim)

            #cost = -(0.25 * iou_sim + 0.75* app_sim)
            cost = -(0.25 * iou_sim + 0.75* app_sim)
            rr, cc = lapsolver.solve_dense(cost.numpy())
            for r, c in zip(rr, cc):
                confirmed_tracks[r].update(detections[c])
            unmatched_rr = np.setdiff1d(np.arange(len(confirmed_tracks)), rr)
            unmatched_cc = np.setdiff1d(np.arange(len(detections)), cc)
            unmatched_tracks = [confirmed_tracks[r] for r in unmatched_rr]
            remaining_tracks.extend(unmatched_tracks)
            detections = [detections[c] for c in unmatched_cc]

        if len(remaining_tracks) > 0 and len(detections) > 0:
            remaining_boxes = torch.stack([track.bbox for track in remaining_tracks])
            remaining_embs = torch.stack([track.emb for track in remaining_tracks])
            detected_boxes = torch.stack([det.bbox for det in detections])
            detected_embs = torch.stack([det.emb for det in detections])
            iou_sim = box_iou(remaining_boxes, detected_boxes)
            app_sim = cosine_similarity(remaining_embs, detected_embs)
            iou_sim[iou_sim < 0.4] = np.nan
            cost = -(1.0 * iou_sim + 0.0 * app_sim)
            #cost = (1 * iou_sim + 0.0* app_sim)
            rr, cc = lapsolver.solve_dense(cost.numpy())
            for r, c in zip(rr, cc):
                remaining_tracks[r].update(detections[c])
            unmatched_rr = np.setdiff1d(np.arange(len(remaining_tracks)), rr)
            unmatched_cc = np.setdiff1d(np.arange(len(detections)), cc)
            unmatched_tracks = [remaining_tracks[r] for r in unmatched_rr]
            detections = [detections[c] for c in unmatched_cc]

            for track in unmatched_tracks:
                track.mark_missed()

        # New tracks
        if len(detections) > 0:
            for det in detections:
                self.track_id += 1
                self.tracks.append(
                    Track(self.track_id, det, self.n_init, self.max_age, self.window)
                )

        # Remove tracks
        self.tracks = [track for track in self.tracks if track.state != 'deleted']

        # Extract result
        result = torch.tensor(
            [
                (track.track_id, *track.bbox.tolist(), track.score.item())
                for track in self.tracks
                if track.time_since_update == 0 and track.state == 'confirmed'
            ]
        )

        result_su = torch.tensor([
                (track.track_id, float(track.ul.item()), track.ut.item(), track.ur.item(), track.ub.item())
                for track in self.tracks
                if track.time_since_update == 0 and track.state == 'confirmed'
            ])

        return result, result_su


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', type=Path, default='videos/mot17')
    parser.add_argument('--dets', type=Path, required=True)
    parser.add_argument('--outs', type=Path, required=True)
    parser.add_argument('--su', default= False)
    args = parser.parse_args()

    assert args.videos.exists()
    assert args.dets.exists()
    if not args.outs.exists():
        args.outs.mkdir(parents=True)

    for video_paths in sorted(list(args.videos.glob('*.imgs'))):
        video_name = video_paths.stem
        fids = pd.read_csv(video_paths)['fid'].values
        dets = pd.read_csv(args.dets / f'{video_name}.csv')
        embs = torch.load(args.dets / f'{video_name}.emb')

        tracker = Tracker()

        preds = []
        preds_su = []
        for fid in tqdm(fids, desc=video_name):
            mask = dets['fid'] == fid

            curr_group = dets[mask]
            curr_embs = embs[mask.values]
            curr_boxes = torch.tensor(curr_group[['x', 'y', 'w', 'h']].values)
            curr_boxes = box_convert(curr_boxes, 'xywh', 'xyxy')
            curr_scores = torch.tensor(curr_group['s'].values)
            curr_sigmas = torch.tensor(curr_group[['ul', 'ut', 'ur', 'ub']].values)

            curr_dets = [
                Detection(b, e, s, u)
                for b, e, s, u in zip(curr_boxes, curr_embs, curr_scores, curr_sigmas)
            ]

            result, result_su = tracker.step(curr_dets)
            if len(result) > 0:

                result[:, 1:5] = box_convert(result[:, 1:5], 'xyxy', 'xywh')
                result = result.numpy()
                result = pd.DataFrame(result, columns=['tag', 'x', 'y', 'w', 'h', 's'])
                result['fid'] = fid
                result['tag'] = result['tag'].astype(int)
    
                # if args.su:
                #     result_su = result_su.numpy()
                #     result_su = pd.DataFrame(result_su, columns=['tag', 'ul', 'ut', 'ur', 'ub'])
                #     result['ul'] = result_su['ul']
                #     result['ut'] = result_su['ut']
                #     result['ur'] = result_su['ur']
                #     result['ub'] = result_su['ub']
           
                preds.append(result)

        df_pred = pd.concat(preds, ignore_index=True)
        df_pred['1'] = -1
        df_pred['2'] = -1
        df_pred['3'] = -1

        # if args.su:
        #     df_pred.to_csv(
        #     args.outs / f'{video_name}.txt',
        #     index=None,
        #     header=None,
        #     columns=['fid', 'tag', 'x', 'y', 'w', 'h', 's', '1', '2', '3',  'ul', 'ut', 'ur', 'ub'],
        # )
        # else:
        #df_pred.drop(['ur', 'ut', 'ur', 'ub'], axis=1, inplace = True)
        df_pred.to_csv(
            args.outs / f'{video_name}.txt',
            index=None,
            header=None,
            columns=['fid', 'tag', 'x', 'y', 'w', 'h', 's', '1', '2', '3'],
        )

evaluator = MOTEvaluator(remove_distractor=True)
for gt_path in sorted(list(args.videos.glob('*.gt'))):
    video_name = gt_path.stem
    df_pred = pd.read_csv(args.outs / f'{video_name}.txt')
    # if args.su:
    #     df_pred.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 's', '1', '2', '3', 'ur', 'ut', 'ur', 'ub']
    #     df_pred.drop(['ur', 'ut', 'ur', 'ub'], axis=1, inplace = True)
    # else:
    df_pred.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 's', '1', '2', '3']
    df_true = pd.read_csv(gt_path)
    evaluator.add(video_name, df_pred, df_true)
summary, text = evaluator.evaluate(False)
print(text)