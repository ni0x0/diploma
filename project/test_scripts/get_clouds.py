import os
import cv2
import numpy as np
import open3d as o3d

# === Параметры камеры ===
fx, fy = 594.21, 591.04
cx, cy = 320, 240

# === Пути ===
photo_dir = "./photos/"
cloud_dir = "./clouds/"
os.makedirs(cloud_dir, exist_ok=True)

# === Углы съёмки ===
angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

# === Выбор метода фильтрации: 'background' или 'depth_range' ===
filter_mode = "depth_range"  # или 'background'

# === Диапазон глубины (в метрах) для метода 'depth_range' ===
depth_min = 0.5
depth_max = 1.5

def depth_to_point_cloud(depth, rgb, fx, fy, cx, cy, mask=None, z_offset=0.88):
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32) / 1000.0  # в метры
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy
    z = z - z_offset  # смещаем глубину так, чтобы ось вращения оказалась в (0, 0, 0)

    if mask is not None:
        mask = mask & (z > -z_offset)  # отбрасываем те, что позади камеры
        x, y, z = x[mask], y[mask], z[mask]
        colors = rgb[mask]
    else:
        x, y, z = x.flatten(), y.flatten(), z.flatten()
        colors = rgb.reshape(-1, 3)

    points = np.stack((x, y, z), axis=-1)
    return points, colors / 255.0


def main():
    rgb_bg = None
    if filter_mode == "background":
        rgb_bg = cv2.imread(os.path.join(photo_dir, "rgb_background.png"))

    for angle in angles:
        print(f"[ ] Обработка угла {angle}°...")

        rgb_path = os.path.join(photo_dir, f"rgb_{angle}.png")
        depth_path = os.path.join(photo_dir, f"depth_{angle}.png")

        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        # === Маска объекта ===
        if filter_mode == "background":
            diff = cv2.absdiff(rgb, rgb_bg)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, fg_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            mask = fg_mask.astype(bool)

        elif filter_mode == "depth_range":
            z = depth.astype(np.float32) / 1000.0  # мм → м
            mask = (z >= depth_min) & (z <= depth_max)

        else:
            raise ValueError(f"Неподдерживаемый режим фильтрации: {filter_mode}")

        # === Построение облака точек ===
        points, colors = depth_to_point_cloud(depth, rgb, fx, fy, cx, cy, mask=mask)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        cloud_path = os.path.join(cloud_dir, f"cloud_{angle}.ply")
        o3d.io.write_point_cloud(cloud_path, pcd)

        print(f"[✓] Сохранено облако: {cloud_path}")
        o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
