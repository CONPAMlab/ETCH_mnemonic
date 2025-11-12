# detector.py
import torch
import numpy as np
from yolov10 import YOLOv10

class YOLOv10Detector:
    def __init__(self, weights="yolov10n.pt", device="0", conf=0.25, iou=0.45, classes=None):
        """
        YOLOv10 detector wrapper
        """
        self.weights = weights
        self.device = device
        self.conf = conf
        self.iou = iou
        self.classes = classes

        # load model
        self.model = YOLOv10(weights)
        self.model.to(device if device != "cpu" else "cpu")
        self.model.eval()

    def predict(self, frame_bgr):
        """
        frame_bgr: BGR np.ndarray
        returns: list of dicts: {bbox, conf, cls, label}
        """
        frame_rgb = frame_bgr[:, :, ::-1]
        results = self.model.predict(
            source=frame_rgb,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )[0]

        dets = []
        if hasattr(results, "boxes") and results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(boxes)):
                xyxy = boxes[i].tolist()
                conf = float(confs[i])
                cls = int(clss[i])
                label = results.names[cls] if hasattr(results, "names") else str(cls)
                dets.append({"bbox": xyxy, "conf": conf, "cls": cls, "label": label})
        return dets
