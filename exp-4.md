import cv2
from ultralytics import YOLO

# ===============================
# Load YOLO Model
# ===============================
model = YOLO("yolov8n.pt")   # Pretrained lightweight model


# ===============================
# IMAGE OBJECT DETECTION
# ===============================
def detect_image(image_path):
    results = model(image_path)

    # Plot results
    annotated_frame = results[0].plot()

    cv2.imshow("Object Detection - Image", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ===============================
# VIDEO OBJECT DETECTION
# ===============================
def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        cv2.imshow("Object Detection - Video", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    detect_image("image.jpg")   # Replace with your image
    detect_video("video.mp4")   # Replace with your video
    output
    <img width="450" height="579" alt="image" src="https://github.com/user-attachments/assets/ab89b042-00b5-47af-8866-1f92438dfc81" />
