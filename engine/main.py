import cv2
import numpy as np
from ultralytics import YOLO
from common import BasicGeometrics
from common import WindowCapture
from PIL import Image, ImageDraw, ImageFont

# Configurações do modelo YOLO
model = YOLO("file_system/models/best.pt")

# Variáveis de rastreamento e configuração
track_history = {}
seguir = False
deixar_rastro = False
geometric = BasicGeometrics()
stream = WindowCapture(1)

# Caminho para o arquivo de fonte TrueType
font_path = "/home/milkzinha/space/sources/cattle-detection-drone/assets/fonts/CommitMonoNerdFont-Regular.otf"

# Carregar a fonte personalizada
font_size = 14
custom_font = ImageFont.truetype(font_path, font_size)

while True:
    frame = stream.screen()

    results = model(frame)

    # Desenhar formas geométricas usando OpenCV
    for result in results:
        for box in result.boxes.xywh:
            x, y, w, h = box
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            color = (25, 220, 255)

            geometric.rounded_rectangle(frame, (x1, y1, x2 - x1, y2 - y1),
                                        lenght_of_corner=3,
                                        thickness_of_line=1,
                                        radius_corner=3)

            # Retângulo de confiança
            cv2.rectangle(frame, (x1 - 20, y1 - 40), (x1 + 60, y1 - 20), color, -1)

            # Calcula o centro do retângulo
            rect_center_x = x1 + 20
            rect_center_y = y1 - 20

            # Desenha uma linha do centro do objeto para o centro do retângulo
            objeto_centro_x = int((x1 + x2) / 2)
            objeto_centro_y = int((y1 + y2) / 2)
            cv2.line(frame, (objeto_centro_x, objeto_centro_y), (rect_center_x, rect_center_y), color, 1)

            # Desenha um circulo no centro do objeto
            cv2.circle(frame, (objeto_centro_x, objeto_centro_y), 1, color, 2)

    # Convert the frame to RGB (Pillow uses RGB format)
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    # Desenha o retangulo do texto
    draw.rectangle((0, 0, 100, 20), fill=(74, 100, 24))
    # Draw PotySpy Logo with custom font
    text = "PotySpy"
    draw.text((0, 0), text, font=custom_font, fill=(255, 255, 255))

    # Desenhar o texto com a fonte personalizada usando Pillow
    for result in results:
        for box in result.boxes.xywh:
            x, y, w, h = box
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Desenhar o texto
            text = f"{result.boxes.conf[0].item():.2f}% "
            draw.text((x1 - 15, y1 - 37), text, font=custom_font, fill=(0, 0, 0))

    # Convert the frame back to BGR format for OpenCV
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("Tela", frame)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
print("Desligando")
