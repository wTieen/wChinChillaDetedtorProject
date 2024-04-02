import os
from ultralytics import YOLO
import cv2

# DUONG DAN INPUT VIDEO
VIDEOS_DIR = os.path.join('.', 'data', 'videos', 'test')
# DUONG DAN OUTPUT VIDEO
VIDEOS_DIR_OUT = os.path.join('.', 'data', 'videos', 'output', 'new')

# DUONG DAN DEN VIDEO INPUT
video_path_in = os.path.join(VIDEOS_DIR, 'chinchilla-video (8).mp4')
# # DUONG DAN DEN VIDEO OUTPUT
video_path_out = os.path.join(VIDEOS_DIR_OUT, 'chinchilla-video (8).mp4')

# LUU VIDEO O DUONG DAN DEN VIDEO OUTPUT O TREN
video_path_out = '{}_out-mS-200.mp4'.format(video_path_out)

cap = cv2.VideoCapture(video_path_in)
ret, frame = cap.read()
if ret:
    assert not isinstance(frame, type(None)), 'frame not found'
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# SU DUNG FILE DA TRAIN DE DU DOAN DOI TUONG TRONG VIDEO
model_path = os.path.join('.', 'runs', 'detect', 'train200s', 'weights', 'best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while ret:
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            label = f'{results.names[int(class_id)].upper()}: {score:.2f}'
            cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
