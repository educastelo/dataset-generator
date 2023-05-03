import cv2
import imutils
from ultralytics import YOLO
from utilities.utils import point_in_polygons, draw_roi
import time


# Cria o detector de faces
model_face = YOLO('models/yolov8n-face.pt')

# Inicie a captura da stream RTSP
cap = cv2.VideoCapture("toronto_02.mkv")

detection_threshold = 0.7

# Defina o nome base para as imagens que serão salvas
nome_base = 'toronto_02'
nome_count = 0
frame_count = 0

# Inicie o loop de leitura da stream RTSP
while True:
    # Leia um frame da stream
    ret, frame = cap.read()
    # print(frame.shape)

    results = model_face(source=frame, verbose=False)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, x2, x3, x4, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            x3 = int(x3)
            x4 = int(x4)

            # filter out weak detections
            if score < detection_threshold:
                continue

            # # Check if the centroid of each object is inside the polygon
            # cX = (x1 + x3) / 2
            # cY = (x2 + x4) / 2
            # test_polygon = point_in_polygons((cX, cY), points_polygon)
            # if not test_polygon:
            #     continue

            if len(r):
                face = frame[x2:x4, x1:x3]
                if frame_count % 240 == 0:
                    # Salve a face em um arquivo .jpg com o nome base + o número da face detectada
                    cv2.imwrite("data/" + nome_base + '_' + str(nome_count) + '.jpg', face)
                    nome_count += 1

                cv2.rectangle(frame, (x1, x2), (x3, x4), (0, 255, 0), 6)

    # Incrementa o contador de frames e reseta quando for igual a 60
    frame_count += 1
    if frame_count == 240:
        frame_count = 0

    # Mostre o frame com as faces detectadas
    resized = imutils.resize(frame, width=1200)
    cv2.imshow('Faces detectadas', resized)

    # Espere por uma tecla para encerrar o loop
    if cv2.waitKey(1) == ord('q'):
        break

# Libere os recursos utilizados
cap.release()
cv2.destroyAllWindows()
