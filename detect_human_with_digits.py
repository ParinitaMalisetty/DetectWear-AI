import os
import cv2
from ultralytics import YOLO

# =====================================================
# PATHS
# =====================================================

VIDEO_PATH = r"uploads/test_no.mp4"

HUMAN_MODEL_PATH = r"best1/best.pt"
DIGIT_MODEL_PATH = r"bestf1/best.pt"

import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_PATH = rf"outputs/output_{timestamp}.mp4"

# =====================================================
# SETTINGS
# =====================================================

PERSON_CLASS_ID = 0

HUMAN_CONF = 0.35
DIGIT_CONF = 0.45

MIN_DIGIT_HEIGHT_RATIO = 0.18
MIN_DIGIT_WIDTH_RATIO = 0.04

os.makedirs("outputs", exist_ok=True)

# =====================================================
# LOAD MODELS
# =====================================================

human_model = YOLO(HUMAN_MODEL_PATH)
digit_model = YOLO(DIGIT_MODEL_PATH)

print("Human model loaded:", HUMAN_MODEL_PATH)
print("Digit model loaded:", DIGIT_MODEL_PATH)
print("Digit model classes:", digit_model.names)

# =====================================================
# HELPERS
# =====================================================

def remove_duplicate_digits(digits, x_threshold=18):
    """
    Removes duplicate detections that are very close horizontally.
    Keeps the higher-confidence box.
    """
    if not digits:
        return []

    digits = sorted(digits, key=lambda d: d["x"])

    cleaned = []

    for d in digits:
        if not cleaned:
            cleaned.append(d)
            continue

        last = cleaned[-1]

        if abs(d["x"] - last["x"]) < x_threshold:
            if d["conf"] > last["conf"]:
                cleaned[-1] = d
        else:
            cleaned.append(d)

    return cleaned

# =====================================================
# VIDEO SETUP
# =====================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise Exception("Could not open video. Check VIDEO_PATH.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 80

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# =====================================================
# PROCESS VIDEO
# =====================================================

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    human_results = human_model.predict(
        source=frame,
        conf=HUMAN_CONF,
        imgsz=640,
        verbose=False
    )

    for result in human_results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0])

            if cls_id != PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue
            
            ph, pw = person_crop.shape[:2]

            # Larger jersey crop
            jy1 = int(ph * 0.08)
            jy2 = int(ph * 0.78)

            jx1 = int(pw * 0.05)
            jx2 = int(pw * 0.95)

            jersey_crop = person_crop[jy1:jy2, jx1:jx2]

            if jersey_crop.size == 0:
                continue
            
            crop_h, crop_w = jersey_crop.shape[:2]
            digit_results = digit_model.predict(
                source=jersey_crop,
                conf=DIGIT_CONF,
                imgsz=640,
                iou=0.45,
                verbose=False
            )

            detected_digits = []

            for dr in digit_results:
                if dr.boxes is None:
                    continue

                for dbox in dr.boxes:
                    digit_cls = int(dbox.cls[0])
                    digit_name = digit_model.names[digit_cls]
                    digit_conf = float(dbox.conf[0])

                    dx1, dy1, dx2, dy2 = map(int, dbox.xyxy[0])

                    digit_w = dx2 - dx1
                    digit_h = dy2 - dy1

                    # =====================================================
                    # FILTER: KEEP ONLY BIG DIGITS
                    # =====================================================

                    if digit_h < crop_h * MIN_DIGIT_HEIGHT_RATIO:
                        continue

                    if digit_w < crop_w * MIN_DIGIT_WIDTH_RATIO:
                        continue

                    full_dx1 = x1 + jx1 + dx1
                    full_dy1 = y1 + jy1 + dy1
                    full_dx2 = x1 + jx1 + dx2
                    full_dy2 = y1 + jy1 + dy2

                    detected_digits.append({
                        "digit": str(digit_name),
                        "x": dx1,
                        "conf": digit_conf,
                        "box": (full_dx1, full_dy1, full_dx2, full_dy2)
                    })
            detected_digits = remove_duplicate_digits(detected_digits)

            if len(detected_digits) > 0:
                detected_digits = sorted(detected_digits, key=lambda d: d["x"])
                jersey_number = "".join([d["digit"] for d in detected_digits])
            else:
                jersey_number = "No Number"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for d in detected_digits:
                dx1, dy1, dx2, dy2 = d["box"]

                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (255, 0, 0), 2)

                cv2.putText(
                    frame,
                    f"{d['digit']} {d['conf']:.2f}",
                    (dx1, max(0, dy1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

            label = f"Jersey: {jersey_number}"

            cv2.rectangle(
                frame,
                (x1, max(0, y1 - 35)),
                (x1 + 280, y1),
                (0, 255, 0),
                -1
            )

            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 8 if y1 - 8 > 0 else y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2
            )

    out.write(frame)
    print(f"Processed frame: {frame_count}", end="\r")

cap.release()
out.release()

print("\nDone.")
print("Output saved at:", OUTPUT_PATH)