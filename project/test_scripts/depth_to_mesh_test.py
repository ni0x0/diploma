import numpy as np
import cv2
from primesense import openni2
import open3d as o3d

# 1. Инициализация OpenNI2
openni2.initialize()  # Или: openni2.initialize('/path/to/OpenNI2/Redist')
dev = openni2.Device.open_any()

# 2. Запуск потоков
depth_stream = dev.create_depth_stream()
color_stream = dev.create_color_stream()
depth_stream.start()
color_stream.start()

# 3. Параметры камеры (Orbbec Astra — 640x480)
width, height = 640, 480
fx = 525.0  # примерное фокусное расстояние
fy = 525.0
cx = width / 2
cy = height / 2

try:
    while True:
        # 4. Получаем данные
        depth_frame = depth_stream.read_frame()
        color_frame = color_stream.read_frame()

        depth_data = depth_frame.get_buffer_as_uint16()
        color_data = color_frame.get_buffer_as_uint8()

        depth = np.frombuffer(depth_data, dtype=np.uint16).reshape((height, width))
        color = np.frombuffer(color_data, dtype=np.uint8).reshape((height, width, 3))

        # 5. Создаем Open3D изображения
        depth_o3d = o3d.geometry.Image(depth)
        color_o3d = o3d.geometry.Image(color)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            convert_rgb_to_intensity=False,
            depth_scale=1000.0,  # если глубина в миллиметрах
            depth_trunc=20.0       # отсечка по глубине в метрах
        )

        # 6. Создаем камеру (pinhole model)
        intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        # 7. Получаем облако точек
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsics
        )

        # 8. Отображаем облако точек (однократно — нажми ESC для выхода)
        o3d.visualization.draw_geometries([pcd])

        # Пауза или обработка одного кадра — break или continue
        break

finally:
    depth_stream.stop()
    color_stream.stop()
    openni2.unload()
