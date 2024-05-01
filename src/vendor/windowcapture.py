from ultralytics import YOLO
import cv2
import numpy as np
import time
import mss
import screeninfo


# Função para capturar a tela usando mss e redimensionar para 1280x720
def capture_screen():
    screen = screeninfo.get_monitors()[1]
    monitor = {"top": screen.y, "left": screen.x, "width": screen.width, "height": screen.height}
    with mss.mss() as sct:
        img = sct.grab(monitor)
        # Convertendo a imagem para formato RGB
        img_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2RGB)
        # Redimensionando a imagem para 1280x720
        img_resized = cv2.resize(img_rgb, (1280, 720))
        return img_resized


# Configurações do modelo YOLO
model = YOLO("../models/best.pt")

# Variáveis de rastreamento e configuração
track_history = {}
seguir = False
deixar_rastro = False

while True:
    img = capture_screen()

    if seguir:
        results = model.track(img, persist=True)
    else:
        results = model(img)

    # Processar resultados
    # Processar resultados
    for result in results:
        # Desenhar novas bounding boxes
        for box in result.boxes.xywh:
            x, y, w, h = box
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = (0, 25, 255)  # cor verde
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

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
