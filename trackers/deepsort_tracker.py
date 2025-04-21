from deep_sort_realtime.deepsort_tracker import DeepSort

class RingTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30)

    def update(self, detections, image_bgr):
        # Detections = list of dicts: {bbox: [x1,y1,x2,y2], confidence: float}
        input_dets = [
            (det["bbox"], det["confidence"], "ring") for det in detections
        ]
        tracks = self.tracker.update_tracks(input_dets, frame=image_bgr)

        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            results.append({
                "track_id": track_id,
                "bbox": [l, t, r, b]
            })

        return results
