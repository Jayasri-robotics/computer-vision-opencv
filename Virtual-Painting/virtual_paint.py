import cv2
import mediapipe as mp
import numpy as np
import time

# ---------------- MODEL PATH ----------------

MODEL_PATH = r"C:\Users\DELL\computer vision and open cv\hand_landmarker.task"

# ---------------- CAMERA SETTINGS ----------------

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# ---------------- MEDIAPIPE SETTINGS ----------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

# ---------------- PAINT SETTINGS ----------------

canvas = None

previous_point = None
smoothed_point = None

# Lower value = smoother, but more delay
SMOOTHING = 0.20

BRUSH_THICKNESS = 10

# BGR colors
colors = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (0, 255, 255)     # Yellow
]

color_names = ["BLUE", "GREEN", "RED", "YELLOW"]

current_color = colors[0]

# ---------------- HAND CONNECTIONS ----------------

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# ---------------- WINDOW ----------------

cv2.namedWindow("Virtual Paint", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Virtual Paint", 1280, 720)

# ---------------- MAIN PROGRAM ----------------

with HandLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("Camera not detected!")
            break

        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        if canvas is None:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp = time.monotonic_ns() // 1_000_000

        results = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        # ---------------- HAND DETECTION ----------------

        if results.hand_landmarks:

            hand_landmarks = results.hand_landmarks[0]

            points = []

            for landmark in hand_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

            # Draw hand landmarks
            for x, y in points:

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    cv2.FILLED
                )

            # Draw hand connections
            for start, end in connections:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 0, 255),
                    2
                )

            # ---------------- FINGER DETECTION ----------------

            index_up = (
                points[8][1] < points[6][1]
            )

            middle_up = (
                points[12][1] < points[10][1]
            )

            index_x, index_y = points[8]

            # ---------------- COLOR SELECTION ----------------

            if index_up and middle_up:

                previous_point = None
                smoothed_point = None

                if index_y < 100:

                    for i in range(len(colors)):

                        x1 = 20 + i * 140
                        x2 = x1 + 120

                        if x1 < index_x < x2:

                            current_color = colors[i]

                cv2.putText(
                    frame,
                    "SELECT COLOR",
                    (20, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

            # ---------------- DRAWING ----------------

            elif index_up and not middle_up:

                current_point = np.array(
                    [index_x, index_y],
                    dtype=np.float32
                )

                if smoothed_point is None:

                    smoothed_point = current_point

                else:

                    smoothed_point = (
                        SMOOTHING * current_point
                        + (1 - SMOOTHING) * smoothed_point
                    )

                smooth_x = int(smoothed_point[0])
                smooth_y = int(smoothed_point[1])

                current_smooth_point = (smooth_x, smooth_y)

                if previous_point is not None:

                    cv2.line(
                        canvas,
                        previous_point,
                        current_smooth_point,
                        current_color,
                        BRUSH_THICKNESS,
                        cv2.LINE_AA
                    )

                previous_point = current_smooth_point

                cv2.circle(
                    frame,
                    current_smooth_point,
                    8,
                    current_color,
                    cv2.FILLED
                )

            else:

                previous_point = None
                smoothed_point = None

        else:

            previous_point = None
            smoothed_point = None

        # ---------------- COLOR MENU ----------------

        for i, color in enumerate(colors):

            x1 = 20 + i * 140
            x2 = x1 + 120

            cv2.rectangle(
                frame,
                (x1, 20),
                (x2, 80),
                color,
                cv2.FILLED
            )

            cv2.putText(
                frame,
                color_names[i],
                (x1, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # ---------------- COMBINE CANVAS ----------------

        output = cv2.addWeighted(
            frame, 1.0,
            canvas, 1.0,
            0
        )

        cv2.imshow("Virtual Paint", output)

        # ---------------- KEYBOARD CONTROLS ----------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):

            canvas = np.zeros(
                (height, width, 3),
                dtype=np.uint8
            )

            previous_point = None
            smoothed_point = None

        elif key == ord('q'):
            break

# ---------------- CLEANUP ----------------

cap.release()
cv2.destroyAllWindows()

print("Virtual Paint Closed")
