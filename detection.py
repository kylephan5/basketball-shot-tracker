import cv2
import numpy as np
import torch


class ShotDetector:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = 'best.pt'
        self.model = torch.hub.load("WongKinYiu/yolov7", "custom", f"{self.path}", trust_repo=True)

    def draw_boxes(self, frame, results):
        for *box, conf, cls in results:
            x1, y1, x2, y2 = map(int, box)
            label = f"{self.model.names[int(cls)]}: {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def start(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)
            detections = results.xyxy[0].cpu().numpy()
            frame = self.draw_boxes(frame, detections)
            cv2.imshow("YOLOv7 Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
