import cv2
import numpy as np
from primesense import openni2
import time

def main():
    # Инициализация OpenNI2
    try:
        openni2.initialize()
    except Exception as e:
        print(f"Failed to initialize OpenNI2: {e}")
        return

    dev = None
    depth_stream = None
    color_stream = None

    try:
        # Открываем устройство
        dev = openni2.Device.open_any()

        # Создаём потоки
        depth_stream = dev.create_depth_stream()
        color_stream = dev.create_color_stream()

        depth_stream.start()
        color_stream.start()

        print("Streaming started. Press 'q' to exit...")

        while True:
            # Получаем кадры с проверкой
            depth_frame = depth_stream.read_frame()
            color_frame = color_stream.read_frame()

            if depth_frame is None or color_frame is None:
                print("Warning: Got empty frame")
                continue

            try:
                # Преобразуем данные
                depth_data = depth_frame.get_buffer_as_uint16()
                color_data = color_frame.get_buffer_as_uint8()

                # Формируем изображения
                depth_image = np.frombuffer(depth_data, dtype=np.uint16).reshape((480, 640))
                color_image = np.frombuffer(color_data, dtype=np.uint8).reshape((480, 640, 3))

                # Конвертируем цвет и обрабатываем глубину
                color_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

                # Отображаем
                cv2.imshow("Color", color_bgr)
                cv2.imshow("Depth", depth_colored)

                # Выход по 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"Processing error: {e}")
                continue

    except Exception as e:
        print(f"Error during streaming: {e}")

    finally:
        print("Shutting down...")
        
        # Закрываем окна
        cv2.destroyAllWindows()
        
        # Останавливаем потоки с проверкой
        if depth_stream is not None:
            depth_stream.stop()
        if color_stream is not None:
            color_stream.stop()
        
        # Даём время на завершение операций
        time.sleep(0.1)
        
        # Закрываем устройство
        if dev is not None:
            dev.close()
        
        # Выгружаем OpenNI
        openni2.unload()
        print("Resources released")

if __name__ == "__main__":
    main()