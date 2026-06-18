import cv2
import mediapipe as mp
import numpy as np
from math import hypot

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

speakers = AudioUtilities.GetSpeakers()

volume = speakers.EndpointVolume.QueryInterface(
    IAudioEndpointVolume
)

vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# ==================================
# MediaPipe Hands
# ==================================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ==================================
# Webcam
# ==================================

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            h, w, c = frame.shape

            # Thumb tip (4)
            x1 = int(hand_landmarks.landmark[4].x * w)
            y1 = int(hand_landmarks.landmark[4].y * h)

            # Index tip (8)
            x2 = int(hand_landmarks.landmark[8].x * w)
            y2 = int(hand_landmarks.landmark[8].y * h)

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Draw points
            cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 10, (255, 0, 255), cv2.FILLED)

            cv2.line(frame, (x1, y1), (x2, y2),
                     (255, 0, 255), 3)

            # Distance
            length = hypot(x2 - x1, y2 - y1)

            # Map distance to volume
            vol = np.interp(
                length,
                [20, 200],
                [min_vol, max_vol]
            )

            volume.SetMasterVolumeLevel(vol, None)

            # Volume Bar
            vol_bar = np.interp(
                length,
                [20, 200],
                [400, 150]
            )

            vol_percent = np.interp(
                length,
                [20, 200],
                [0, 100]
            )

            cv2.rectangle(frame,
                          (50, 150),
                          (85, 400),
                          (0, 255, 0),
                          3)

            cv2.rectangle(frame,
                          (50, int(vol_bar)),
                          (85, 400),
                          (0, 255, 0),
                          cv2.FILLED)

            cv2.putText(frame,
                        f'{int(vol_percent)}%',
                        (30, 450),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        3)

            # If fingers are very close
            if length < 30:
                cv2.circle(
                    frame,
                    ((x1 + x2) // 2, (y1 + y2) // 2),
                    10,
                    (0, 255, 0),
                    cv2.FILLED
                )

    cv2.imshow("Hand Gesture Volume Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()
