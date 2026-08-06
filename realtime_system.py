# ================================================================
# realtime_system.py — FINAL DEMO VERSION (MTCNN + RTX GPU)
# Multi-Task Facial Behaviour and Occlusion Detection
# MCA Final Year Project — ITER, SOA University, Bhubaneswar
#
# PIPELINE:
#   Face Detection  : MTCNN (facenet-pytorch) — accurate alignment
#   Emotion Model   : ResNet-18 + CBAM (FER2013, 71%+ acc)
#   Occlusion Det.  : MediaPipe FaceMesh visibility scores
#   Device          : CUDA (RTX GPU recommended)
#
# INSTALL (on GPU PC):
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
#   pip install facenet-pytorch opencv-python mediapipe Pillow numpy
#
# RUN:
#   python realtime_system.py
#   python realtime_system.py --model fer_resnet18_cbam.pth
#   python realtime_system.py --source video.mp4
#
# CONTROLS:
#   Q — Quit
#   S — Screenshot
#   B — Toggle confidence bars
#   H — Toggle help
# ================================================================

import argparse
import time
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from facenet_pytorch import MTCNN

from model import load_model
from occlusion_detector import OcclusionDetector

# CONFIG

CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

EMOTION_COLORS = {
    'Angry':    (30,  30,  220),
    'Disgust':  (30,  140, 30),
    'Fear':     (160, 30,  160),
    'Happy':    (30,  200, 30),
    'Neutral':  (180, 180, 180),
    'Sad':      (220, 100, 30),
    'Surprise': (30,  200, 220),
}

EMOTION_ICONS = {
    'Angry':    '>:(',
    'Disgust':  '>_<',
    'Fear':     ':O',
    'Happy':    ':D',
    'Neutral':  ':-|',
    'Sad':      ':(',
    'Surprise': ':O!',
}

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# HUD DRAWING FUNCTIONS

def draw_face_box(frame, x1, y1, x2, y2, color, thickness=2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    corner = min((x2-x1), (y2-y1)) // 6
    for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),
                            (x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx,cy), (cx+dx*corner, cy), color, thickness+1)
        cv2.line(frame, (cx,cy), (cx, cy+dy*corner), color, thickness+1)


def draw_emotion_label(frame, x1, y1, emotion, confidence, color):
    icon = EMOTION_ICONS.get(emotion, '')
    text = f"{icon} {emotion}: {confidence:.1f}%"
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    bg_y1 = max(y1 - th - bl - 8, 0)
    cv2.rectangle(frame, (x1, bg_y1), (x1+tw+8, y1), (20,20,20), -1)
    cv2.rectangle(frame, (x1, bg_y1), (x1+tw+8, y1), color, 1)
    cv2.putText(frame, text, (x1+4, y1-bl-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_occlusion_badge(frame, x1, y2, occ_result):
    label = occ_result["label"]
    color = occ_result["color"]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y2+2), (x1+tw+10, y2+th+10), (20,20,20), -1)
    cv2.rectangle(frame, (x1, y2+2), (x1+tw+10, y2+th+10), color, 1)
    cv2.putText(frame, label, (x1+5, y2+th+4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_confidence_bars(frame, x1, y2, probs, emo_i, frame_h):
    start_y = y2 + 55
    bar_w   = 110
    bar_h   = 10
    gap     = 16
    text_x  = x1 + bar_w + 5
    for i, (cls, prob) in enumerate(zip(CLASSES, probs)):
        by = start_y + i * gap
        if by + bar_h > frame_h - 5:
            break
        filled    = int(prob * bar_w)
        is_top    = (i == emo_i)
        bar_color = EMOTION_COLORS[cls] if is_top else (60,60,60)
        cv2.rectangle(frame, (x1,by), (x1+bar_w, by+bar_h), (30,30,30), -1)
        if filled > 0:
            cv2.rectangle(frame, (x1,by), (x1+filled, by+bar_h), bar_color, -1)
        cv2.rectangle(frame, (x1,by), (x1+bar_w, by+bar_h), (80,80,80), 1)
        lc = (255,255,255) if is_top else (150,150,150)
        cv2.putText(frame, f"{cls[:3]:3s} {prob*100:5.1f}%",
                    (text_x, by+bar_h-1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, lc, 1, cv2.LINE_AA)


def draw_top_bar(frame, fps, face_count):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0,0), (w,35), (15,15,15), -1)
    cv2.putText(frame,
                "Multi-Task Facial Behaviour & Occlusion Detection",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(frame,
                f"FPS: {fps:5.1f}  |  Faces: {face_count}",
                (w-210, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (100,220,100), 1, cv2.LINE_AA)


def draw_help(frame):
    h = frame.shape[0]
    lines = ["Q=Quit  S=Screenshot  B=Bars  H=Help",
             "Green=No Occlusion  Orange=Partial  Red=Severe"]
    for i, line in enumerate(lines):
        cv2.putText(frame, line,
                    (8, h - 10 - (len(lines)-1-i)*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (160,160,160), 1, cv2.LINE_AA)


def draw_no_face(frame):
    h, w = frame.shape[:2]
    cv2.putText(frame, "No face detected",
                (w//2-100, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (80,80,80), 2, cv2.LINE_AA)


# MAIN SYSTEM

def run(model_path="fer_resnet18_cbam.pth",
        source=0,
        skip_n=2,
        mtcnn_conf=0.85):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*55}")
    print(f" Multi-Task Facial Behaviour & Occlusion Detection")
    print(f" FINAL DEMO — MTCNN + {'CUDA' if device.type=='cuda' else 'CPU'}")
    print(f"{'='*55}")
    print(f" Device     : {device}")
    if torch.cuda.is_available():
        print(f" GPU        : {torch.cuda.get_device_name(0)}")
    print(f" Model      : {model_path}")
    print(f" Frame skip : every {skip_n} frames")
    print(f" MTCNN conf : {mtcnn_conf}")
    print(f"{'='*55}\n")

    # ── Load FER model ───────────────────────────────────────────
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        print("Place fer_resnet18_cbam.pth in the same folder.")
        return

    model   = load_model(model_path, device)
    occ_det = OcclusionDetector()

    # ── MTCNN ────────────────────────────────────────────────────
    mtcnn = MTCNN(
        keep_all=True,
        device=device,
        thresholds=[0.6, 0.7, 0.7],
        min_face_size=60,
        post_process=False
    )
    print("[FaceDetector] MTCNN initialized\n")

    # ── Camera ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {source}")
        return

    print("Controls: Q=Quit  S=Screenshot  B=Bars  H=Help\n")

    # ── State ────────────────────────────────────────────────────
    show_bars      = True
    show_help      = True
    fps_buffer     = []
    prev_tick      = cv2.getTickCount()
    frame_idx      = 0
    cached_faces   = []
    screenshot_dir = "results/screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)

    # ── Main loop ────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended.")
            break

        # FPS
        curr_tick  = cv2.getTickCount()
        fps_now    = cv2.getTickFrequency() / (curr_tick - prev_tick + 1e-9)
        prev_tick  = curr_tick
        fps_buffer.append(fps_now)
        if len(fps_buffer) > 20:
            fps_buffer.pop(0)
        fps_smooth = np.mean(fps_buffer)

        frame_h, frame_w = frame.shape[:2]
        run_inference     = (frame_idx % skip_n == 0)
        frame_idx        += 1

        # ── Inference on every skip_n frame ──────────────────────
        if run_inference:
            cached_faces = []
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs = mtcnn.detect(rgb)

            if boxes is not None:
                for box, score in zip(boxes, probs):
                    if score is None or score < mtcnn_conf:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    x1 = max(0, x1);  y1 = max(0, y1)
                    x2 = min(frame_w-1, x2)
                    y2 = min(frame_h-1, y2)

                    if (x2-x1) < 40 or (y2-y1) < 40:
                        continue

                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue

                    # Task 1 — Occlusion Detection
                    occ_result = occ_det.detect(face)

                    # Task 2 — Emotion Recognition
                    img_t = TRANSFORM(face).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(img_t)
                    emotion_probs = torch.softmax(
                        logits, dim=1)[0].cpu().numpy()
                    emo_i      = int(np.argmax(emotion_probs))
                    emotion    = CLASSES[emo_i]
                    confidence = float(emotion_probs[emo_i]) * 100

                    cached_faces.append({
                        "box":        (x1, y1, x2, y2),
                        "emotion":    emotion,
                        "confidence": confidence,
                        "probs":      emotion_probs,
                        "emo_i":      emo_i,
                        "occ_result": occ_result,
                    })

        # ── Draw cached results every frame ──────────────────────
        for item in cached_faces:
            x1, y1, x2, y2 = item["box"]
            draw_face_box(frame, x1, y1, x2, y2,
                          item["occ_result"]["color"])
            draw_emotion_label(frame, x1, y1,
                               item["emotion"], item["confidence"],
                               EMOTION_COLORS[item["emotion"]])
            draw_occlusion_badge(frame, x1, y2, item["occ_result"])
            if show_bars:
                draw_confidence_bars(frame, x1, y2,
                                     item["probs"], item["emo_i"], frame_h)

        if not cached_faces:
            draw_no_face(frame)

        draw_top_bar(frame, fps_smooth, len(cached_faces))
        if show_help:
            draw_help(frame)

        cv2.imshow("Multi-Task Facial Behaviour & Occlusion Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = os.path.join(screenshot_dir,
                                 f"shot_{int(time.time())}.jpg")
            cv2.imwrite(fname, frame)
            print(f"  Screenshot: {fname}")
        elif key == ord('b'):
            show_bars = not show_bars
        elif key == ord('h'):
            show_help = not show_help

    cap.release()
    cv2.destroyAllWindows()
    occ_det.close()
    print("System stopped.")


# ENTRY POINT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="fer_resnet18_cbam.pth")
    parser.add_argument("--source", default="0")
    parser.add_argument("--skip",   type=int,   default=2)
    parser.add_argument("--conf",   type=float, default=0.85)
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    run(model_path=args.model, source=src,
        skip_n=args.skip, mtcnn_conf=args.conf)
