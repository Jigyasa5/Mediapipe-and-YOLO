import cv2
import mediapipe as mp
import numpy as np
from math import hypot

cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def EAR(points):

    p1,p2,p3,p4,p5,p6 = points

    vertical1 = hypot(
        p2[0]-p6[0],
        p2[1]-p6[1]
    )

    vertical2 = hypot(
        p3[0]-p5[0],
        p3[1]-p5[1]
    )

    horizontal = hypot(
        p1[0]-p4[0],
        p1[1]-p4[1]
    )

    ear = (vertical1 + vertical2) / (2.0 * horizontal)

    return ear

EAR_THRESHOLD = 0.20
CLOSED_FRAMES = 20

counter = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face in results.multi_face_landmarks:

            h, w, _ = frame.shape

            left_points = []
            right_points = []

            for idx in LEFT_EYE:

                lm = face.landmark[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)

                left_points.append((x, y))

            for idx in RIGHT_EYE:

                lm = face.landmark[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)

                right_points.append((x, y))

            for point in left_points:
                cv2.circle(frame, point, 2, (0,255,0), -1)

            for point in right_points:
                cv2.circle(frame, point, 2, (0,255,0), -1)

            left_ear = EAR(left_points)

            right_ear = EAR(right_points)

            avg_ear = (left_ear + right_ear) / 2

            cv2.putText(
                frame,
                f"EAR: {avg_ear:.2f}",
                (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,0,0),
                2
            )
            if avg_ear < EAR_THRESHOLD:

                counter += 1

            else:

                counter = 0

            if counter > CLOSED_FRAMES:

                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (50,120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3
                )

                cv2.imshow("Driver Drowsiness Detection", frame)

    cv2.imshow("Face Mesh", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()