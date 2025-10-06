import cv2
import numpy as np
import os
from primesense import openni2

# Углы поворота
angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
current_idx = 0

# Инициализация
openni2.initialize()
dev = openni2.Device.open_any()

depth_stream = dev.create_depth_stream()
color_stream = dev.create_color_stream()
depth_stream.start()
color_stream.start()

# Создание папки, если нет
os.makedirs("photos", exist_ok=True)

print("[INFO] Нажми 'b' для сохранения фона, 's' — для сохранения с текущим углом, 'n' — следующий угол, 'Esc' — выход.")

try:
    while True:
        depth_frame = depth_stream.read_frame()
        color_frame = color_stream.read_frame()

        depth_data = depth_frame.get_buffer_as_uint16()
        color_data = color_frame.get_buffer_as_uint8()

        depth = np.frombuffer(depth_data, dtype=np.uint16).reshape((480, 640))
        color = np.frombuffer(color_data, dtype=np.uint8).reshape((480, 640, 3))
        color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        # Визуализация глубины
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        # Показываем
        display_text = f"Угол: {angles[current_idx]}° (нажми 'n' чтобы сменить)"
        cv2.putText(color_bgr, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Color", color_bgr)
        cv2.imshow("Depth", depth_colored)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # Esc
            break
        elif key == ord('b'):
            # Сохраняем фоновое изображение
            cv2.imwrite("photos/rgb_background.png", color_bgr)
            cv2.imwrite("photos/depth_background.png", depth)
            print("[INFO] Фоновое изображение сохранено.")
        elif key == ord('s'):
            # Сохраняем изображение текущего угла
            angle = angles[current_idx]
            cv2.imwrite(f"photos/rgb_{angle}.png", color_bgr)
            cv2.imwrite(f"photos/depth_{angle}.png", depth)
            print(f"[INFO] Сохранено: rgb_{angle}.png и depth_{angle}.png")
        elif key == ord('n'):
            # Переход к следующему углу
            current_idx += 1
            if current_idx >= len(angles):
                print("[INFO] Все углы пройдены.")
                break

finally:
    depth_stream.stop()
    color_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()
