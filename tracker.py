# tracker.py
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou(bb_test, bb_gt):
    """
    IoU between two bboxes in [x1,y1,x2,y2] format
    """
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area1 + area2 - inter
    return 0 if union <= 0 else inter / union

class Track:
    _count = 0
    def __init__(self, bbox, cls=None, conf=1.0):
        # bbox: [x1,y1,x2,y2]
        self.kf = self._init_kf(bbox)
        self.time_since_update = 0
        self.id = Track._count
        Track._count += 1
        self.hits = 1
        self.age = 1
        self.cls = cls
        self.conf = conf
        self.history = []
        self.last_bbox = bbox

    def _init_kf(self, bbox):
        # state [cx, cy, s, r, vx, vy, vs]
        kf = KalmanFilter(dim_x=7, dim_z=4)
        dt = 1.0
        # state transition
        kf.F = np.eye(7)
        for i in range(3):
            kf.F[i, i+4] = dt
        # measurement function maps state to [cx, cy, s, r]
        kf.H = np.zeros((4,7))
        kf.H[0,0] = 1.0
        kf.H[1,1] = 1.0
        kf.H[2,2] = 1.0
        kf.H[3,3] = 1.0
        kf.R *= 10.0
        kf.P *= 10.0
        kf.Q *= 1.0
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        s = w * h
        r = w / float(h + 1e-6)
        kf.x = np.array([cx, cy, s, r, 0, 0, 0]).reshape((7,1))
        return kf

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        cx, cy, s, r = self.kf.x[0,0], self.kf.x[1,0], self.kf.x[2,0], self.kf.x[3,0]
        w = np.sqrt(max(s,1e-6) * r)
        h = s / (w + 1e-6)
        x1 = cx - w/2.0
        y1 = cy - h/2.0
        x2 = cx + w/2.0
        y2 = cy + h/2.0
        self.last_bbox = [x1, y1, x2, y2]
        return self.last_bbox

    def update(self, bbox, cls=None, conf=1.0):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        s = w * h
        r = w / float(h + 1e-6)
        z = np.array([cx, cy, s, r]).reshape((4,1))
        self.kf.update(z)
        self.time_since_update = 0
        self.hits += 1
        self.conf = conf
        if cls is not None:
            self.cls = cls
        self.history.append(self.last_bbox)

class SortTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []

    def update(self, detections):
        """
        detections: list of dicts with keys 'bbox' (x1,y1,x2,y2), 'cls', 'conf'
        returns: list of active tracks dicts: {'id','bbox','cls','conf'}
        """
        # predict all tracks
        for tr in self.tracks:
            tr.predict()

        N = len(self.tracks)
        M = len(detections)
        iou_matrix = np.zeros((N, M), dtype=np.float32)

        for t, tr in enumerate(self.tracks):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = iou(tr.last_bbox, det['bbox'])

        matched_indices = []
        if N > 0 and M > 0:
            # Hungarian maximize, so minimize negative iou
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched_indices.append((r, c))

        unmatched_tracks = set(range(N))
        unmatched_dets = set(range(M))
        matches = []

        for r, c in matched_indices:
            matches.append((r, c))
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)

        # update matched tracks
        for r, c in matches:
            self.tracks[r].update(detections[c]['bbox'], cls=detections[c].get('cls'), conf=detections[c].get('conf',1.0))

        # create new tracks for unmatched detections
        for idx in unmatched_dets:
            det = detections[idx]
            self.tracks.append(Track(det['bbox'], cls=det.get('cls'), conf=det.get('conf',1.0)))

        # remove dead tracks
        alive_tracks = []
        outputs = []
        for tr in self.tracks:
            if tr.time_since_update < 1:
                # included in this frame
                outputs.append({'id': tr.id, 'bbox': tr.last_bbox, 'cls': tr.cls, 'conf': tr.conf, 'hits': tr.hits})
            if tr.time_since_update <= self.max_age:
                alive_tracks.append(tr)
        self.tracks = alive_tracks

        # optionally filter out tracks with few hits (unconfirmed)
        final_outputs = []
        for out in outputs:
            if out['hits'] >= self.min_hits:
                final_outputs.append(out)
        return final_outputs
