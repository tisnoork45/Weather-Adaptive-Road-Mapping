import cv2

def draw_hud(frame,
             condition,
             object_count,
             fusion_confidence,
             decision):

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60),
                  (20, 20, 20), -1)

    cv2.putText(frame,
                f"Condition: {condition}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2)

    cv2.putText(frame,
                f"Objects: {object_count}",
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2)

    cv2.putText(frame,
                f"Fusion: {fusion_confidence:.2f}",
                (220, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2)

    cv2.putText(frame,
                f"Decision: {decision}",
                (450, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2)

    return frame