import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands


def extract_hand_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Image not found. Check path.")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hands = mp_hands.Hands(
        static_image_mode=True,       # ✅ MUST be True for single images
        max_num_hands=2,              # Try detecting both hands
        min_detection_confidence=0.5  # You can reduce this to 0.3 if needed
    )

    results = hands.process(rgb_image)

    if not results.multi_hand_landmarks:
        print("❌ No hands detected.")
        return None, image

    hand_landmarks = results.multi_hand_landmarks[0]

    h, w, _ = image.shape
    landmarks = []

    for lm in hand_landmarks.landmark:
        x, y, z = lm.x * w, lm.y * h, lm.z * w  # scale z using image width
        landmarks.append((x, y, z))

    return landmarks, image
