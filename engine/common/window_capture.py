import cv2
import numpy as np
import mss
import screeninfo

from ultralytics import YOLO


class WindowCapture:

    def __init__(self,
                 monitor: int = 0,
                 window_name: str = "",
                 output_resolution: tuple = (1280, 720),
                 origin: tuple = (0, 0)):

        monitor_info = screeninfo.get_monitors()[monitor]
        self.output_resolution = output_resolution
        self.origin = origin
        self.window_name = window_name
        self.monitor = {
            "top": monitor_info.y,
            "left": monitor_info.x,
            "width": monitor_info.width,
            "height": monitor_info.height
        }

    def screen(self):

        with mss.mss() as sct:
            frame = sct.grab(self.monitor)
            # Convertendo a imagem para formato RGB
            frame_as_rgb = cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2RGB)
            frame_resized = cv2.resize(frame_as_rgb, (1280, 720))

            return frame_resized
