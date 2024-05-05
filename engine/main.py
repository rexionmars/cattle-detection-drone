import cv2
import numpy as np
from ultralytics import YOLO

from common import BasicGeometrics
from common import WindowCapture

# Configurações do modelo YOLO
model = YOLO("/home/rexionmars/space/OpenSource/cattle-detection-drone/engine/models/best.pt")

# Variáveis de rastreamento e configuração
track_history = {}
seguir = False
deixar_rastro = False
geometric = BasicGeometrics()
stream = WindowCapture(1)

# Caminho para o arquivo de fonte TrueType
font_path = "../assets/fonts/UbuntuNerdFont-Regular.ttf"

# Carregar a fonte
font = cv2.FONT_HERSHEY_SIMPLEX

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
            # geometric.rounded_rectangle(img, (x1, y1, x2 - x1, y2 - y1),
            #                             lenght_of_corner=5,
            #                             thickness_of_line=1,
            #                             radius_corner=3)
            # Rectângulo de confiança
            # Cordenadas        x1, y1, x2, y2
            cv2.rectangle(img, (x1 - 20, y1 - 40), (x1 + 85, y1 - 20), color, -1)

            # Calcula o centro do retângulo
            rect_center_x = x1 + 20
            rect_center_y = y1 - 20

            # Desenha uma linha do centro do objeto para o centro do retângulo
            objeto_centro_x = int((x1 + x2) / 2)
            objeto_centro_y = int((y1 + y2) / 2)
            cv2.line(img, (objeto_centro_x, objeto_centro_y), (rect_center_x, rect_center_y), color, 1)

            # Desenhar o texto com a fonte personalizada
            cv2.putText(img, f"{result.boxes.conf[0].item():.3f} {result.names[0]}", (x1 - 15, y1 - 25), font, 0.5,
                        (0, 0, 0), 1)

            # Desenha um circulo no centro do objeto
            #cv2.circle(img, (objeto_centro_x, objeto_centro_y), 5, color, 2)

            # Obtem o width e height do objeto detectado
            width = x2 - x1
            height = y2 - y1
            # Desenha uma elipse no centro do objeto
            cv2.ellipse(img,
                        center=(objeto_centro_x, objeto_centro_y),
                        axes=(int(width), int(0.15 * width)),
                        angle=0,
                        startAngle=0,
                        endAngle=360,
                        color=color,
                        thickness=1)

    cv2.imshow("Tela", img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
print("Desligando")
