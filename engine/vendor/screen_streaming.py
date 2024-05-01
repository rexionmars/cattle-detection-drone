from ultralytics import YOLO
import cv2
import numpy as np
import time
import mss
import screeninfo


# Função para capturar a tela usando mss
def capture_screen():
    screen = screeninfo.get_monitors()[0]
    monitor = {"top": screen.y, "left": screen.x, "width": screen.width, "height": screen.height}
    with mss.mss() as sct:
        img = sct.grab(monitor)
        return np.array(img)


# Configurações do modelo YOLO
model = YOLO("yolov8n.pt")

# Variáveis de rastreamento e configuração
track_history = {}
seguir = True
deixar_rastro = True

while True:
    img = capture_screen()

    if seguir:
        results = model.track(img, persist=True)
    else:
        results = model(img)

    # Processar resultados
    for result in results:
        img = result.plot()

        if seguir and deixar_rastro:
            try:
                # Obter as caixas e IDs de rastreamento
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    if track_id not in track_history:
                        track_history[track_id] = []
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)

                    # Desenhar as linhas de rastreamento
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [points], isClosed=False, color=(230, 0, 0), thickness=5)
            except:
                pass

    cv2.imshow("Tela", img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
print("Desligando")
