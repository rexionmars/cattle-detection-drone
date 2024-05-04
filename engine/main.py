import cv2
import numpy as np
from ultralytics import YOLO
from common.geometrics import BasicGeometrics
from common.window_capture import WindowCapture

# Configurações do modelo YOLO
model = YOLO("/home/rexionmars/space/OpenSource/cattle-detection-drone/engine/models/best.pt")

# Variáveis de rastreamento e configuração
track_history = {}
seguir = False
deixar_rastro = False
geometric = BasicGeometrics()
stream = WindowCapture(1)

while True:
    img = stream.screen()

    results = model(img)

    # Processar resultados
    for result in results:
        # Desenhar novas bounding boxes
        for box in result.boxes.xywh:
            x, y, w, h = box
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = (25, 220, 255)  # cor verde
            geometric.rounded_rectangle(img, (x1, y1, x2 - x1, y2 - y1),
                                        lenght_of_corner=5,
                                        thickness_of_line=2,
                                        radius_corner=3)
            # Rectângulo de confiança
            cv2.rectangle(img, (x1, y1 - 20), (x2, y1), color, -1)
            cv2.putText(img, f"{result.boxes.conf[0].item():.3f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1)

    cv2.imshow("Tela", img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
print("Desligando")
