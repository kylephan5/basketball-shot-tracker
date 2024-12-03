import cv2
import torch
import time
from helper import close_to_hoop, get_position_hoop

class ShotDetector:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = 'best2.pt'
        self.model = torch.hub.load("WongKinYiu/yolov7", "custom", f"{self.path}", trust_repo=True)
        self.makes = 0
        self.total_shots = 0
        self.start_time_draw_make = 0
        self.draw_make = False
        self.hoop_detected = False

    def detect_make(self, rim_height, rim_width, center_x, center_y, last_frame_x, last_frame_y):
        # close to hoop, falling, and passed the rim
        if center_y > rim_height > last_frame_y[-1]:
            m = (center_y - last_frame_y[-1]) / (center_x - last_frame_x[-1])
            x_target = ((rim_height - last_frame_y[-1]) / m) + last_frame_x[-1]

            if rim_width[0] < x_target < rim_width[1]:
                self.makes += 1
                self.draw_make = True
                self.start_time_draw_make = time.time()

            self.total_shots += 1

    def draw_hoop(self, frame, detections):
        found = False
        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            label = f"{self.model.names[int(cls)]}: {conf:.2f}"
            if conf > .9 and self.model.names[int(cls)] == 'Basketball Hoop':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                found = True

        return found

    def start(self):
        cap = cv2.VideoCapture(1)

        tracker = None
        tracking = False
        frame_count = 0
        rim_height, rim_width = 0, (0, 0)
        last_frame_x, last_frame_y = [], []
        detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # object detection, reset object detection after some number of frames
            self.hoop_detected = False
            if self.draw_hoop(frame, detections):
                self.hoop_detected = True

            if not tracking or frame_count > 30:
                results = self.model(frame)
                detections = results.xyxy[0].cpu().numpy()
                # detect basketball
                basketball_detections = [d for d in detections if self.model.names[int(d[5])] == 'Basketball' and d[4] > .8]
                if basketball_detections:
                    basketball_detections.sort(key=lambda d: d[4], reverse=True)
                    x1,y1,x2,y2, conf, cls = basketball_detections[0]
                    track_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

                    center_x = x1 + (x2 - x1) // 2
                    center_y = y1 + (y2 - y1) // 2
                    last_frame_x.append(center_x)
                    last_frame_y.append(center_y)

                    # create tracker
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, track_bbox)
                    tracking = True
                    frame_count = 0

                # calculate rim_height, rim_width values
                rim_height, rim_width = get_position_hoop(detections, self.model)

            if tracking:
                success, track_bbox = tracker.update(frame)
                if success:
                    x, y, w, h = map(int, track_bbox)
                    # if basketball goes off frame, stop tracking
                    # if x < 0 or y < 0 or x + w > frame_width or y + h > frame_height:
                    #     tracking = False
                    #     continue

                    center_x = x + w // 2
                    center_y = y + h // 2
                    if center_y > last_frame_y[-1]:
                        if close_to_hoop(rim_height, rim_width, center_x, center_y):
                            self.detect_make(rim_height, rim_width, center_x, center_y, last_frame_x, last_frame_y)

                    cv2.circle(frame, (int(center_x), int(center_y)), 15, (255, 0, 0), -1)
                    last_frame_x.append(center_x)
                    last_frame_y.append(center_y)
                else:
                    tracking = False

            if tracking and last_frame_x and last_frame_y:
                cv2.circle(frame, (last_frame_x[-1], last_frame_y[-1]), 15, (255, 0, 0), -1)

            if self.draw_make:
                cv2.putText(frame, "$", ((rim_width[0] + rim_width[1]) // 2, rim_height - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                if time.time() - self.start_time_draw_make > 2:
                    self.draw_make = False

            frame_count += 1
            cv2.putText(frame, f'{self.makes} / {self.total_shots}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
            cv2.imshow("YOLOv7 Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()








