import cv2
import numpy as np

from ui.hud_renderer import draw_hud

# Create blank test frame
frame = np.zeros((720, 1280, 3), dtype=np.uint8)

# Draw HUD
frame = draw_hud(
    frame,
    condition="CLEAR",
    object_count=5,
    fusion_confidence=0.87,
    decision="SAFE"
)

cv2.imshow("Weather Adaptive Road Mapping", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()