from deep_sort_realtime.deepsort_tracker import DeepSort

class JewelryTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30)

    def update(self, detections, image_bgr):
        input_dets = [
            (det["bbox"], det["confidence"], det["class_name"]) for det in detections
        ]
        tracks = self.tracker.update_tracks(input_dets, frame=image_bgr)

        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            results.append({
                "track_id": track.track_id,
                "bbox": [x1, y1, x2, y2],
                "class_name": track.get_det_class(),
            })

        return results
