import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_num_hands=2, detection_confidence=0.7, tracking_confidence=0.6):
        self.max_num_hands = max_num_hands
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

    def process_image(self, image_bgr):
        results = self.hands.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Store (x, y) landmarks
                hand_coords = [
                    (int(lm.x * image_bgr.shape[1]), int(lm.y * image_bgr.shape[0]))
                    for lm in hand_landmarks.landmark
                ]
                landmarks.append(hand_coords)
            print(f"[✓] {len(results.multi_hand_landmarks)} hand(s) detected")

        return landmarks

    def draw_landmarks(self, image_bgr, landmarks):
        for hand in landmarks:
            for point in hand:
                cv2.circle(image_bgr, point, 4, (0, 255, 0), -1)
        return image_bgr
