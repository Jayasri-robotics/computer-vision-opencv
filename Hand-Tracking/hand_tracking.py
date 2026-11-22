
import cv2
import mediapipe as mp
import time

MODEL_PATH = r"C:\Users\DELL\computer vision and open cv\hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Webcam could not be opened.")
    raise SystemExit

print("Hand Tracking Started. Press Q to exit.")

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = camera.read()

            if not success:
                print("Cannot read webcam.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = time.monotonic_ns() // 1_000_000

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            height, width, _ = frame.shape

            for hand_index, hand in enumerate(result.hand_landmarks):

                points = []

                for landmark in hand:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    points.append((x, y))

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # Draw hand skeleton
                for start, end in connections:
                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (255, 0, 255),
                        2
                    )

                # Get handedness
                handedness = result.handedness[hand_index][0].category_name

                # Correct label for mirrored webcam view
                if handedness == "Left":
                    handedness = "Right"
                else:
                    handedness = "Left"

                # Display hand label
                wrist_x, wrist_y = points[0]

                cv2.putText(
                    frame,
                    handedness + " Hand",
                    (wrist_x - 50, wrist_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

            # Display hand count
            hand_count = len(result.hand_landmarks)

            cv2.putText(
                frame,
                f"Hands Detected: {hand_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "Real-Time Hand Tracking",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Hand Tracking Stopped.")
