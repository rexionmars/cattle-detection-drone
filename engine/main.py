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
                                        thickness_of_line=1,
                                        radius_corner=3)
            # Rectângulo de confiança
            # Cordenadas        x1, y1, x2, y2
            cv2.rectangle(img, (x1 - 20, y1 - 40), (x1 + 35, y1 - 20), color, -1)

            # Calcula o centro do retângulo
            retangulo_centro_x = x1 + 20
            retangulo_centro_y = y1 - 20

            # Desenha uma linha do centro do objeto para o centro do retângulo
            objeto_centro_x = int((x1 + x2) / 2)
            objeto_centro_y = int((y1 + y2) / 2)
            cv2.line(img, (objeto_centro_x, objeto_centro_y), (retangulo_centro_x, retangulo_centro_y), color, 1)

            cv2.putText(img, f"{result.boxes.conf[0].item():.3f}", (x1 - 15, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1)

    cv2.imshow("Tela", img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
print("Desligando")
