import cv2
from src.vision.detector import EyeDetector

detector = EyeDetector()
cap = cv2.VideoCapture(0)

print("Starting webcam test. Press 'q' on the video window to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    # Flip horizontally for a natural mirror view
    frame = cv2.flip(frame, 1)

    result = detector.process_frame(frame)

    # Render simple latency and count overlay
    cv2.putText(
        result.annotated_frame,
        f"Eyes: {result.eye_count} | Latency: {result.latency_ms:.1f}ms",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Eye Detection Smoke Test", result.annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()