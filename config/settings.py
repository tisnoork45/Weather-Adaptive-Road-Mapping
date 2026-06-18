YOLO_MODEL = "weights/yolov8n.pt"

CONF_THRESHOLD = 0.25

FOCAL_LENGTH_PX = 718.0

OUTPUT_VIDEO = "outputs/processed_video.mp4"

COLORS = {
    "safe": (0, 255, 0),
    "caution": (0, 165, 255),
    "danger": (0, 0, 255),
    "text": (255, 255, 255)
}