import cv2
import os
from models.hand_tracker import HandTracker
from detectors.ring_detector import RingDetector
from trackers.deepsort_tracker import RingTracker


def run_video_inference(video_path, output_path, model_path="runs/detect/train/weights/best.pt"):
    cap = cv2.VideoCapture(video_path)
    tracker = HandTracker()
    detector = RingDetector(model_path=model_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    frame_count = 0

    ring_tracker = RingTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize (optional)
        frame = cv2.resize(frame, (640, 480))

        # Hand landmarks
        hand_landmarks = tracker.process_image(frame)
        frame = tracker.draw_landmarks(frame, hand_landmarks)

        # Ring detection
        detections = detector.detect_rings(frame)
        tracked_rings = ring_tracker.update(detections, frame)

        for ring in tracked_rings:
            x1, y1, x2, y2 = ring['bbox']
            track_id = ring['track_id']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, f"Ring ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Init video writer
        if out is None:
            height, width = frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

        out.write(frame)
        frame_count += 1
        print(f"[✓] Frame {frame_count} processed")

    cap.release()
    out.release()
    print(f"\n🎉 Video saved to: {output_path}")
