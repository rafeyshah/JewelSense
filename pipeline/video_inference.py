import cv2
import os
from detectors.jewelry_detector import JewelryDetector
from trackers.jewelry_tracker import JewelryTracker


def run_video_inference(video_path, output_path,
                        model_path="runs/detect/train/weights/best.pt"):
    CLASS_NAMES = ['ring', 'earring', 'dress']

    cap = cv2.VideoCapture(video_path)
    detector = JewelryDetector(model_path=model_path, class_names=class_names)
    tracker = JewelryTracker()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.resize(frame, (640, 480))

        detections = detector.detect(frame)
        tracked = tracker.update(detections, frame)

        for obj in tracked:
            x1, y1, x2, y2 = obj['bbox']
            cls_name = obj['class_name']
            track_id = obj['track_id']

            label = f"{cls_name} ID {track_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            # cv2.rectangle(frame, (100, 100), (101, 101), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))

        out.write(frame)
        frame_idx += 1
        print(f"[✓] Frame {frame_idx} processed")

    cap.release()
    out.release()
    print(f"\n🎬 Saved video to: {output_path}")
