import cv2
from ultralytics import YOLO

# ===== LOAD MODEL =====
model = YOLO("yolov8n.pt")

# ===== LOAD IMAGE =====
img_path = r"C:\Users\admin\Downloads\2011_09_26\2011_09_26_drive_0001_sync\image_02\data\0000000000.png"
img = cv2.imread(img_path)

# ===== CREATE CONDITIONS =====

# Bright
bright = cv2.convertScaleAbs(img, alpha=1.5, beta=50)

# Dark
dark = cv2.convertScaleAbs(img, alpha=0.5, beta=-50)

# Blur (fog-like)
blur = cv2.GaussianBlur(img, (15, 15), 0)

# Zoom (close-up)
h, w, _ = img.shape
zoom = img[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
zoom = cv2.resize(zoom, (w, h))

# ===== STORE ALL CONDITIONS =====
conditions = {
    "original": img,
    "bright": bright,
    "dark": dark,
    "blur": blur,
    "zoom": zoom
}

# ===== TEST LOOP =====
for name, test_img in conditions.items():

    results = model(test_img)
    output = test_img.copy()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = model.names[int(box.cls[0])]

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, f"{label} {conf:.2f}",
                    (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

    cv2.imshow(name, output)
    cv2.waitKey(0)

cv2.destroyAllWindows()