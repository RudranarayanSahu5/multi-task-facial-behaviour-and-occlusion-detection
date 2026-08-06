# ================================================================
# debug_occlusion.py
# Run this to see ACTUAL visibility scores from your webcam
# This helps us tune the thresholds correctly for YOUR face
#
# RUN: python debug_occlusion.py
# Press Q to quit
# ================================================================

import cv2
import numpy as np
import mediapipe as mp
from collections import deque

try:
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh

# Same landmark indices as occlusion_detector.py
LEFT_EYE_IDX  = [33, 133, 160, 159, 158, 157, 173, 144,
                  163, 7, 246, 161, 160, 159, 158, 157]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 373,
                  390, 249, 466, 388, 387, 386, 385, 384]
MOUTH_IDX     = [61, 291, 13, 14, 78, 308, 95, 191,
                  80, 81, 82, 87, 178, 88, 317, 402]
NOSE_IDX      = [1, 2, 98, 327, 4, 19, 94, 168]

def region_visibility(landmarks, indices):
    scores = [landmarks[i].visibility
              for i in indices if i < len(landmarks)]
    return float(np.mean(scores)) if scores else 0.0

def texture_score(img, landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
           for i in indices if i < len(landmarks)]
    if len(pts) < 2:
        return 100.0
    pts  = np.array(pts)
    x, y, bw, bh = cv2.boundingRect(pts)
    pad  = 4
    x, y = max(0, x-pad), max(0, y-pad)
    bw   = min(img.shape[1]-x, bw+pad*2)
    bh   = min(img.shape[0]-y, bh+pad*2)
    roi  = img[y:y+bh, x:x+bw]
    if roi.size == 0:
        return 100.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
print("\n=== OCCLUSION DEBUG MODE ===")
print("Watch the scores printed below.")
print("CLEAR FACE   → all scores should be HIGH (>0.5)")
print("COVER MOUTH  → mouth_vis should DROP (<0.3)")
print("SUNGLASSES   → eye_vis should DROP (<0.3)")
print("Press Q to quit\n")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res  = face_mesh.process(rgb)

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        left_vis  = region_visibility(lm, LEFT_EYE_IDX)
        right_vis = region_visibility(lm, RIGHT_EYE_IDX)
        mouth_vis = region_visibility(lm, MOUTH_IDX)
        nose_vis  = region_visibility(lm, NOSE_IDX)
        eye_vis   = (left_vis + right_vis) / 2.0

        eye_tex   = texture_score(frame, lm,
                                  LEFT_EYE_IDX + RIGHT_EYE_IDX, w, h)
        mouth_tex = texture_score(frame, lm, MOUTH_IDX, w, h)

        # Print every 15 frames to avoid spam
        frame_count += 1
        if frame_count % 15 == 0:
            print(f"eye_vis={eye_vis:.3f} | mouth_vis={mouth_vis:.3f} | "
                  f"nose_vis={nose_vis:.3f} | "
                  f"eye_tex={eye_tex:.1f} | mouth_tex={mouth_tex:.1f}")

        # Draw scores on frame
        lines = [
            f"eye_vis   : {eye_vis:.3f}",
            f"mouth_vis : {mouth_vis:.3f}",
            f"nose_vis  : {nose_vis:.3f}",
            f"eye_tex   : {eye_tex:.1f}",
            f"mouth_tex : {mouth_tex:.1f}",
        ]
        cv2.rectangle(frame, (5, 40), (280, 175), (20,20,20), -1)
        for i, line in enumerate(lines):
            # Color red if low, green if high
            val = float(line.split(":")[-1].strip())
            color = (0,200,0) if val > 0.4 else (0,0,220)
            cv2.putText(frame, line,
                        (10, 65 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

        # Show what current thresholds would decide
        eye_occ   = eye_vis   < 0.20 or (eye_vis   < 0.38 and eye_tex   < 15)
        mouth_occ = mouth_vis < 0.20 or (mouth_vis < 0.38 and mouth_tex < 15)
        nose_occ  = nose_vis  < 0.22

        n = sum([eye_occ, mouth_occ, nose_occ])
        if n == 0:   label = "No Occlusion"
        elif n == 1: label = "Partial Occlusion"
        else:        label = "Severe Occlusion"

        color = (0,220,0) if n==0 else (0,165,255) if n==1 else (0,0,220)
        cv2.putText(frame, f"Decision: {label}",
                    (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

    else:
        cv2.putText(frame, "No face detected",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,0,220), 2)

    cv2.putText(frame, "DEBUG MODE — Q to quit",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200,200,0), 1)

    cv2.imshow("Occlusion Debug", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("\nDebug session ended.")
print("Share the printed scores so thresholds can be tuned!")
