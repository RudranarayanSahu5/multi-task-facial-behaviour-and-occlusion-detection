
# occlusion_detector.py 

import cv2
import numpy as np
from collections import deque

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh

LEFT_EYE_IDX  = [33, 133, 160, 159, 158, 157, 173, 144]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 373]
MOUTH_IDX     = [61, 291, 13, 14, 78, 308, 95, 191]
NOSE_IDX      = [1, 2, 98, 327, 4, 19, 94, 168]


class OcclusionDetector:


    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        #  Fix 1: Larger history buffer for stable majority vote
        self.history = deque(maxlen=20)

        # Fix 2: Rolling texture buffers (smooth noise)
        self.eye_tex_buf   = deque(maxlen=5)
        self.mouth_tex_buf = deque(maxlen=5)
        self.nose_tex_buf  = deque(maxlen=5)

        # Fix 3: Hysteresis — sticky state
        self.current_label    = "No Occlusion"
        self.no_occ_counter   = 0   # consecutive No-Occlusion frames
        self.STICKY_THRESHOLD = 15  # frames needed to switch back

        print("[OcclusionDetector] Stable Anti-Flicker Mode")

    def _get_roi(self, img, landmarks, indices, w, h, pad=6):
        pts = []
        for idx in indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                pts.append((int(lm.x * w), int(lm.y * h)))
        if len(pts) < 2:
            return None
        pts = np.array(pts)
        x, y, bw, bh = cv2.boundingRect(pts)
        x  = max(0, x-pad);  y = max(0, y-pad)
        bw = min(img.shape[1]-x, bw+pad*2)
        bh = min(img.shape[0]-y, bh+pad*2)
        roi = img[y:y+bh, x:x+bw]
        return roi if roi.size > 0 else None

    def _texture(self, roi):
        if roi is None or roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _darkness(self, roi):
        if roi is None or roi.size == 0:
            return 128.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def detect(self, face_bgr):
        if face_bgr is None or face_bgr.size == 0:
            return self._make("Severe Occlusion", 2, "empty")

        h, w = face_bgr.shape[:2]
        rgb  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        res  = self.face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            return self._make("Severe Occlusion", 2, "no face")

        lm = res.multi_face_landmarks[0].landmark

        # ── Extract ROIs ──────────────────────────────────────────
        left_roi  = self._get_roi(face_bgr, lm, LEFT_EYE_IDX,  w, h)
        right_roi = self._get_roi(face_bgr, lm, RIGHT_EYE_IDX, w, h)
        mouth_roi = self._get_roi(face_bgr, lm, MOUTH_IDX,     w, h)
        nose_roi  = self._get_roi(face_bgr, lm, NOSE_IDX,      w, h)

        # ── Raw scores ────────────────────────────────────────────
        raw_eye_tex   = (self._texture(left_roi) +
                         self._texture(right_roi)) / 2.0
        raw_mouth_tex = self._texture(mouth_roi)
        raw_nose_tex  = self._texture(nose_roi)
        eye_dark      = (self._darkness(left_roi) +
                         self._darkness(right_roi)) / 2.0

        #  Fix 2: Add to rolling buffers
        self.eye_tex_buf.append(raw_eye_tex)
        self.mouth_tex_buf.append(raw_mouth_tex)
        self.nose_tex_buf.append(raw_nose_tex)

        # Use rolling average instead of raw value
        eye_tex   = float(np.mean(self.eye_tex_buf))
        mouth_tex = float(np.mean(self.mouth_tex_buf))
        nose_tex  = float(np.mean(self.nose_tex_buf))

        # ── Occlusion decision ────────────────────────────────────
        #  Fix 4: Raised mouth threshold 25 → 35 for mask stability
        eye_occluded   = (
            eye_tex < 20
            or (eye_dark < 45 and eye_tex < 40)
        )
        mouth_occluded = mouth_tex < 35    # ✅ raised from 25
        nose_occluded  = nose_tex  < 18

        parts = []
        if eye_occluded:   parts.append("eyes")
        if mouth_occluded: parts.append("mouth")
        if nose_occluded and not parts: parts.append("nose")
        details = ", ".join(parts) if parts else "clear"

        n = sum([eye_occluded, mouth_occluded, nose_occluded])
        if n == 0:   raw_label, sev = "No Occlusion",      0
        elif n == 1: raw_label, sev = "Partial Occlusion",  1
        else:        raw_label, sev = "Severe Occlusion",   2

        #  Fix 1: Majority vote over 20 frames
        self.history.append(raw_label)
        voted = max(set(self.history), key=self.history.count)

        #  Fix 3: Hysteresis — sticky state logic
        # Once occlusion detected → need STICKY_THRESHOLD consecutive
        # No-Occlusion frames to switch back
        if voted == "No Occlusion":
            if self.current_label != "No Occlusion":
                self.no_occ_counter += 1
                if self.no_occ_counter >= self.STICKY_THRESHOLD:
                    self.current_label  = "No Occlusion"
                    self.no_occ_counter = 0
                # Keep showing previous occlusion state
                smooth = self.current_label
            else:
                self.no_occ_counter = 0
                self.current_label  = "No Occlusion"
                smooth = "No Occlusion"
        else:
            self.no_occ_counter = 0
            self.current_label  = voted
            smooth = voted

        COLOR = {
            "No Occlusion":      (0, 220, 0),
            "Partial Occlusion": (0, 165, 255),
            "Severe Occlusion":  (0, 0, 220),
        }
        return {
            "label":      smooth,
            "severity":   sev,
            "color":      COLOR[smooth],
            "details":    details,
            "eye_tex":    round(eye_tex,   1),
            "mouth_tex":  round(mouth_tex, 1),
            "eye_dark":   round(eye_dark,  1),
        }

    @staticmethod
    def _make(label, severity, details):
        COLOR = {
            "No Occlusion":      (0, 220, 0),
            "Partial Occlusion": (0, 165, 255),
            "Severe Occlusion":  (0, 0, 220),
        }
        return {"label": label, "severity": severity,
                "color": COLOR.get(label, (0,0,220)),
                "details": details}

    def close(self):
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


if __name__ == "__main__":
    det = OcclusionDetector()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        r = det.detect(frame)
        cv2.putText(frame, r["label"],
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, r["color"], 2)
        cv2.putText(frame,
                    f"eye_tex={r.get('eye_tex',0):.0f} "
                    f"mouth_tex={r.get('mouth_tex',0):.0f}",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,255,255), 1)
        cv2.imshow("Occlusion Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    det.close()