import cv2
from ultralytics import YOLO
from utilities.utils import point_in_polygons, draw_roi
import os
import time


# Function to check if the RTMP stream is available
def is_rtmp_stream_available(stream_url):
    cap = cv2.VideoCapture(stream_url)
    available = cap.isOpened()
    cap.release()
    return available


# Create the face detector
current_dir = os.path.dirname(os.path.abspath(__file__))
yolo_model_path = os.path.join(current_dir, 'models', 'yolov8n-face.pt')
model_face = YOLO(yolo_model_path)

# Define the RTSP stream to be read
stream = "rtmp://rtmp01...."
points_polygon = [[[701, 522], [2429, 1220], [2496, 989], [936, 360], [701, 520]]]

# Define the base name for the saved images
base_name = 'faces'
frame_count = 0
name_count = 0

# Start the loop to read the RTSP stream
while True:
    # Check if the RTMP stream is available
    while not is_rtmp_stream_available(stream):
        print("Stream offline. Trying to reconnect...")
        time.sleep(5)  # Wait for 5 seconds before trying again

    # Stream is online, so proceed with reading the frames
    for result in model_face(source=stream, verbose=False, max_det=100, stream=True, show=False,
                             conf=0.2, agnostic_nms=True):
        frame = result.orig_img
        boxes = result.boxes
        for r in boxes:
            # Extract the bounding box coordinates, confidence, and class of each object
            x1, x2, x3, x4 = map(int, r.xyxy[0])
            class_id = int(r.cls[0])
            score = float(r.conf[0])

            # Check if the centroid of each object is inside the polygon
            cX = (x1 + x3) / 2
            cY = (x2 + x4) / 2
            if not point_in_polygons((cX, cY), points_polygon):
                continue

            if len(r):
                face = frame[x2:x4, x1:x3]
                if frame_count % 20 == 0:
                    # Salve a face em um arquivo .jpg com o nome base + o número da face detectada
                    cv2.imwrite("data/" + base_name + '_' + str(name_count) + '.jpg', face)
                    name_count += 1

                cv2.rectangle(frame, (x1, x2), (x3, x4), (0, 255, 0), 6)

        # Incrementa o contador de frames e reseta quando for igual a 60
        frame_count += 1
        if frame_count == 20:
            frame_count = 0

        # Mostre o frame com as faces detectadas
        resized = cv2.resize(frame, (1200, int(frame.shape[0] * 1200 / frame.shape[1])))
        cv2.imshow('Faces detectadas', resized)

        # Espere por uma tecla para encerrar o loop
        if cv2.waitKey(1) == ord('q'):
            break
