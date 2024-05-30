import cv2
import numpy as np
from common import BasicGeometrics, WindowCapture
from icecream import ic
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Configurações do modelo YOLO
model = YOLO("file_system/models/best.pt")

# Variáveis de rastreamento e configuração
track_history = {}
deixar_rastro = False
geometric = BasicGeometrics()
# stream = WindowCapture(1)

# Caminho para o arquivo de fonte TrueType
font_path = "/home/milkzinha/space/sources/cattle-detection-drone/assets/fonts/CommitMonoNerdFont-Regular.otf"

# Carregar a fonte personalizada
font_size = 14
custom_font = ImageFont.truetype(font_path, font_size)
POTYSPY_LOGO_FONT = ImageFont.truetype(font_path, font_size)
# frame_capture = cv2.VideoCapture("http://192.168.101.35:81/stream")
frame_capture = cv2.VideoCapture(0)

while True:
    ret, frame = frame_capture.read()
    if not ret:
        break

    # Convert the frame to RGB (Pillow uses RGB format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    results = model(frame_pil)

    # Desenhar formas geométricas usando OpenCV
    for result in results:
        ic.configureOutput(prefix="[INFO] result: ")
        ic(result)

        for box in result.boxes.xywh:
            x, y, w, h = box
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            color = (25, 220, 255)

            geometric.rounded_rectangle(
                frame,
                (x1, y1, x2 - x1, y2 - y1),
                lenght_of_corner=20,
                thickness_of_line=2,
                radius_corner=20,
            )

            # Retângulo de confiança
            cv2.rectangle(frame, (x1 - 20, y1 - 40), (x1 + 60, y1 - 20), color, -1)

            # Calcula o centro do retângulo
            rect_center_x = x1 + 20
            rect_center_y = y1 - 20

            # Desenha uma linha do centro do objeto para o centro do retângulo
            objeto_centro_x = int((x1 + x2) / 2)
            objeto_centro_y = int((y1 + y2) / 2)
            cv2.line(
                frame,
                (objeto_centro_x, objeto_centro_y),
                (rect_center_x, rect_center_y),
                color,
                1,
            )

            # Desenha um circulo no centro do objeto
            cv2.circle(frame, (objeto_centro_x, objeto_centro_y), 1, color, 2)

    # Convert the frame to RGB (Pillow uses RGB format)
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    # Desenha o retângulo para o texto do tempo de predição
    draw.rectangle((100, 0, 280, 16), fill=(249, 184, 12))

    # Desenha o retângulo para o texto do logo
    draw.rectangle((0, 0, 100, 18), fill=(0, 0, 0))
    # Draw PotySpy Logo with custom font
    draw.text((7, 2), "A R K T U S", font=POTYSPY_LOGO_FONT, fill=(255, 255, 255))

    # Desenhar o texto com a fonte personalizada usando Pillow
    for result in results:
        for box in result.boxes.xywh:
            x, y, w, h = box
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Obtém o tempo de predição diretamente do dicionário de velocidades
            if "speed" in result and "inference" in result.speed:
                predict_time = f"Tempo de predição: {result.speed['inference']:.0f} ms"
            else:
                predict_time = "Tempo de predição: N/A"
            ic(predict_time)
            draw.text((105, 2), predict_time, font=custom_font, fill=(0, 0, 0))

            # Desenhar o texto
            if len(result.boxes.conf) > 0:
                conf_text = f"{result.boxes.conf[0].item():.2f}% Vaca"
                draw.text(
                    (x1 - 15, y1 - 37), conf_text, font=custom_font, fill=(0, 0, 0)
                )

    # Convert the frame back to BGR format for OpenCV
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("Tela", frame)

    k = cv2.waitKey(1)
    if k == ord("q"):
        break

frame_capture.release()
cv2.destroyAllWindows()
print("Desligando")
