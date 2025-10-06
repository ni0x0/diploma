import open3d as o3d
import numpy as np
import os

# === Пути ===
cloud0_path = "D:\\diploma\\teapot_clouds\\teapot_cloud_0.ply"
cloud180_path = "D:\\diploma\\teapot_clouds\\teapot_cloud_180.ply"

if not os.path.exists(cloud0_path) or not os.path.exists(cloud180_path):
    print("[Error] Один из файлов не найден!")
    exit()

# === Загрузка и фильтрация ===
def load_filtered_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd.select_by_index(ind)

cloud0 = load_filtered_cloud(cloud0_path)
cloud180 = load_filtered_cloud(cloud180_path)


# === Поворот обоих облаков на 35° вокруг X ===
initial_translation = np.eye(4)
angle_x_rad = np.deg2rad(-30)
rotation_35_x = np.array([
    [1, 0,                0,               0],
    [0, np.cos(angle_x_rad), -np.sin(angle_x_rad), 0],
    [0, np.sin(angle_x_rad),  np.cos(angle_x_rad), 0],
    [0, 0,                0,               1]
])
cloud0.transform(rotation_35_x)
cloud180.transform(rotation_35_x)

# # === Смещение к общей системе координат ===

initial_translation = np.eye(4)
initial_translation[0, 3] = -0.02   # Сдвиг по X (если нужно — сюда число)
initial_translation[1, 3] = -0.2  # Сдвиг по Y
initial_translation[2, 3] = -0.60   # Сдвиг по Z (если нужно — сюда число)

cloud0.transform(initial_translation)
cloud180.transform(initial_translation)

# === Поворот второго облака на 180° вокруг Y ===

initial_translation = np.eye(4)

angle_rad = np.deg2rad(180)
rotation_180_y = np.array([
    [ np.cos(angle_rad), 0, np.sin(angle_rad), 0],
    [ 0,                 1, 0,                 0],
    [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
    [ 0,                 0, 0,                 1]
])
cloud180.transform(rotation_180_y)

# === Координатная система ===
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

# === Интерфейс визуализации ===
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="Cloud Alignment Tool")

vis.add_geometry(cloud0)
vis.add_geometry(cloud180)
vis.add_geometry(axis)

# === Отслеживание смещений ===
offsets0 = {"x": 0.0, "y": 0.0, "z": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}
offsets180 = {"x": 0.0, "y": 0.0, "z": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}

def print_offsets():
    def fmt(v): return f"{v:+.3f}" if isinstance(v, float) else f"{v:+}"
    print("-" * 80)
    print("[cloud_0]    Pos: x={} y={} z={} | Rot: rx={}° ry={}° rz={}°".format(
        fmt(offsets0["x"]), fmt(offsets0["y"]), fmt(offsets0["z"]),
        fmt(offsets0["rx"]), fmt(offsets0["ry"]), fmt(offsets0["rz"])
    ))
    print("[cloud_180]  Pos: x={} y={} z={} | Rot: rx={}° ry={}° rz={}°".format(
        fmt(offsets180["x"]), fmt(offsets180["y"]), fmt(offsets180["z"]),
        fmt(offsets180["rx"]), fmt(offsets180["ry"]), fmt(offsets180["rz"])
    ))

# === Трансформации ===
def translate_cloud(cloud, axis_idx, delta):
    T = np.eye(4)
    T[axis_idx, 3] = delta
    cloud.transform(T)

def rotate_cloud(cloud, axis, degrees):
    radians = np.deg2rad(degrees)
    R = np.eye(4)
    if axis == 'x':
        R[1:3, 1:3] = [[np.cos(radians), -np.sin(radians)],
                       [np.sin(radians),  np.cos(radians)]]
    elif axis == 'y':
        R[[0, 0, 2, 2], [0, 2, 0, 2]] = [np.cos(radians), np.sin(radians),
                                        -np.sin(radians), np.cos(radians)]
    elif axis == 'z':
        R[0:2, 0:2] = [[np.cos(radians), -np.sin(radians)],
                       [np.sin(radians),  np.cos(radians)]]
    cloud.transform(R)

def make_translate_fn(cloud, axis_idx, delta, offsets, key):
    def fn(vis):
        translate_cloud(cloud, axis_idx, delta)
        offsets[key] += delta
        print_offsets()
        vis.update_geometry(cloud)
    return fn

def make_rotate_fn(cloud, axis, delta_deg, offsets, key):
    def fn(vis):
        rotate_cloud(cloud, axis, delta_deg)
        offsets[key] += delta_deg
        print_offsets()
        vis.update_geometry(cloud)
    return fn

# === Назначение клавиш ===

# cloud180 (движение: QWASZX, вращение: ERDFCV)
bindings = [
    ("Q", make_translate_fn(cloud180, 0, +0.01, offsets180, "x")),
    ("W", make_translate_fn(cloud180, 0, -0.01, offsets180, "x")),
    ("A", make_translate_fn(cloud180, 1, +0.01, offsets180, "y")),
    ("S", make_translate_fn(cloud180, 1, -0.01, offsets180, "y")),
    ("Z", make_translate_fn(cloud180, 2, +0.01, offsets180, "z")),
    ("X", make_translate_fn(cloud180, 2, -0.01, offsets180, "z")),
    
    ("E", make_rotate_fn(cloud180, 'x', +1, offsets180, "rx")),
    ("R", make_rotate_fn(cloud180, 'x', -1, offsets180, "rx")),
    ("D", make_rotate_fn(cloud180, 'y', +1, offsets180, "ry")),
    ("F", make_rotate_fn(cloud180, 'y', -1, offsets180, "ry")),
    ("C", make_rotate_fn(cloud180, 'z', +1, offsets180, "rz")),
    ("V", make_rotate_fn(cloud180, 'z', -1, offsets180, "rz")),
    
    # cloud0 (движение: UIPOJK, вращение: OL,.;BN)
    ("U", make_translate_fn(cloud0, 0, +0.01, offsets0, "x")),
    ("I", make_translate_fn(cloud0, 0, -0.01, offsets0, "x")),
    ("H", make_translate_fn(cloud0, 1, +0.01, offsets0, "y")),
    ("J", make_translate_fn(cloud0, 1, -0.01, offsets0, "y")),
    ("B", make_translate_fn(cloud0, 2, +0.01, offsets0, "z")),
    ("N", make_translate_fn(cloud0, 2, -0.01, offsets0, "z")),
    
    ("O", make_rotate_fn(cloud0, 'x', +1, offsets0, "rx")),
    ("P", make_rotate_fn(cloud0, 'x', -1, offsets0, "rx")),
    ("K", make_rotate_fn(cloud0, 'y', +1, offsets0, "ry")),
    ("L", make_rotate_fn(cloud0, 'y', -1, offsets0, "ry")),
    ("M", make_rotate_fn(cloud0, 'z', +1, offsets0, "rz")),
    (",", make_rotate_fn(cloud0, 'z', -1, offsets0, "rz")),
]

for key, func in bindings:
    vis.register_key_callback(ord(key), func)

# === Запуск ===
vis.run()
vis.destroy_window()



# import open3d as o3d
# import numpy as np
# import os

# # === Пути ===
# cloud0_path = "D:\\diploma\\teapot_clouds\\teapot_cloud_0.ply" # os.path.join("pointclouds", "cloud_0.ply")
# cloud180_path = "D:\\diploma\\teapot_clouds\\teapot_cloud_180.ply"  # os.path.join("pointclouds", "cloud_30.ply")

# if not os.path.exists(cloud0_path) or not os.path.exists(cloud180_path):
#     print("[Error] Один из файлов не найден!")
#     exit()

# # === Загрузка и фильтрация ===
# def load_filtered_cloud(path):
#     pcd = o3d.io.read_point_cloud(path)
#     cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
#     return pcd.select_by_index(ind)

# cloud0 = load_filtered_cloud(cloud0_path)
# cloud180 = load_filtered_cloud(cloud180_path)

# # === Смещение обоих облаков к общей системе координат ===
# initial_translation = np.array([
#     [1, 0, 0, -0.060],
#     [0, 1, 0,    0.0],
#     [0, 0, 1, -1.420],
#     [0, 0, 0,    1.0]
# ])
# cloud0.transform(initial_translation)
# cloud180.transform(initial_translation)

# # === Поворот cloud_30 на 30° вокруг Y ===
# angle_rad = np.deg2rad(180)
# rotation_30_y = np.array([
#     [ np.cos(angle_rad), 0, np.sin(angle_rad), 0],
#     [ 0,                 1, 0,                 0],
#     [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
#     [ 0,                 0, 0,                 1]
# ])
# cloud180.transform(rotation_30_y)

# # === Координатная система ===
# axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

# # === Интерфейс визуализации ===
# vis = o3d.visualization.VisualizerWithKeyCallback()
# vis.create_window(window_name="Control cloud_30")

# vis.add_geometry(cloud0)
# vis.add_geometry(cloud180)
# vis.add_geometry(axis)

# # === Управление состоянием cloud_30 ===
# offsets = {
#     "x": 0.0, "y": 0.0, "z": 0.0,
#     "rx": 0.0, "ry": 0.0, "rz": 0.0
# }

# def print_offsets():
#     print(f"[cloud_30] Position: x={offsets['x']:.3f}, y={offsets['y']:.3f}, z={offsets['z']:.3f} | "
#           f"Rotation: rx={offsets['rx']:.1f}°, ry={offsets['ry']:.1f}°, rz={offsets['rz']:.1f}°")

# # === Преобразования ===
# def translate_cloud(cloud, axis_idx, delta):
#     T = np.eye(4)
#     T[axis_idx, 3] = delta
#     cloud.transform(T)

# def rotate_cloud(cloud, axis, degrees):
#     radians = np.deg2rad(degrees)
#     R = np.eye(4)
#     if axis == 'x':
#         R[1:3, 1:3] = [[np.cos(radians), -np.sin(radians)],
#                        [np.sin(radians),  np.cos(radians)]]
#     elif axis == 'y':
#         R[[0, 0, 2, 2], [0, 2, 0, 2]] = [np.cos(radians), np.sin(radians),
#                                         -np.sin(radians), np.cos(radians)]
#     elif axis == 'z':
#         R[0:2, 0:2] = [[np.cos(radians), -np.sin(radians)],
#                        [np.sin(radians),  np.cos(radians)]]
#     cloud.transform(R)

# # === Трансляции cloud_30 ===
# def move_dx(vis):
#     translate_cloud(cloud180, 0, 0.01)
#     offsets["x"] += 0.01
#     print_offsets()
#     vis.update_geometry(cloud180)

# def move_sx(vis):
#     translate_cloud(cloud180, 0, -0.01)
#     offsets["x"] -= 0.01
#     print_offsets()
#     vis.update_geometry(cloud180)

# def move_up(vis):
#     translate_cloud(cloud180, 1, 0.01)
#     offsets["y"] += 0.01
#     print_offsets()
#     vis.update_geometry(cloud180)

# def move_down(vis):
#     translate_cloud(cloud180, 1, -0.01)
#     offsets["y"] -= 0.01
#     print_offsets()
#     vis.update_geometry(cloud180)

# def move_fw(vis):
#     translate_cloud(cloud180, 2, 0.01)
#     offsets["z"] += 0.01
#     print_offsets()
#     vis.update_geometry(cloud180)

# def move_bw(vis):
#     translate_cloud(cloud180, 2, -0.01)
#     offsets["z"] -= 0.01
#     print_offsets()
#     vis.update_geometry(cloud180)

# # === Вращения cloud_30 ===
# def rot_x_p(vis):
#     rotate_cloud(cloud180, 'x', 1)
#     offsets["rx"] += 1.0
#     print_offsets()
#     vis.update_geometry(cloud180)

# def rot_x_n(vis):
#     rotate_cloud(cloud180, 'x', -1)
#     offsets["rx"] -= 1.0
#     print_offsets()
#     vis.update_geometry(cloud180)

# def rot_y_p(vis):
#     rotate_cloud(cloud180, 'y', 1)
#     offsets["ry"] += 1.0
#     print_offsets()
#     vis.update_geometry(cloud180)

# def rot_y_n(vis):
#     rotate_cloud(cloud180, 'y', -1)
#     offsets["ry"] -= 1.0
#     print_offsets()
#     vis.update_geometry(cloud180)

# def rot_z_p(vis):
#     rotate_cloud(cloud180, 'z', 1)
#     offsets["rz"] += 1.0
#     print_offsets()
#     vis.update_geometry(cloud180)

# def rot_z_n(vis):
#     rotate_cloud(cloud180, 'z', -1)
#     offsets["rz"] -= 1.0
#     print_offsets()
#     vis.update_geometry(cloud180)

# # === Назначение клавиш ===
# vis.register_key_callback(ord("D"), move_dx)
# vis.register_key_callback(ord("A"), move_sx)
# vis.register_key_callback(ord("W"), move_up)
# vis.register_key_callback(ord("S"), move_down)
# vis.register_key_callback(ord("E"), move_fw)
# vis.register_key_callback(ord("Q"), move_bw)

# vis.register_key_callback(ord("J"), rot_x_p)
# vis.register_key_callback(ord("U"), rot_x_n)
# vis.register_key_callback(ord("K"), rot_y_p)
# vis.register_key_callback(ord("I"), rot_y_n)
# vis.register_key_callback(ord("L"), rot_z_p)
# vis.register_key_callback(ord("O"), rot_z_n)

# # === Запуск ===
# vis.run()
# vis.destroy_window()




# #######################################################   смещение   #############################################################

# # import open3d as o3d
# # import numpy as np
# # import os

# # # === Настройки ===
# # angles = [0, 30, ] #60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
# # input_folder = "pointclouds"
# # output_folder = "aligned_clouds"
# # os.makedirs(output_folder, exist_ok=True)

# # # Смещения (чтобы привести облака к общему центру координат)
# # shift_x = -0.060
# # shift_z = -1.420

# # # Список для хранения всех облаков для финальной визуализации
# # all_pcds = []

# # for angle in angles:
# #     filename = f"cloud_{angle}.ply"
# #     input_path = os.path.join(input_folder, filename)
    
# #     if not os.path.exists(input_path):
# #         print(f"[!] Пропущено: {input_path} не найдено")
# #         continue
    
# #     # Загрузка облака
# #     pcd = o3d.io.read_point_cloud(input_path)

# #     # Фильтрация: удаляем выбросы (шум) из облака
# #     cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
# #     pcd_filtered = pcd.select_by_index(ind)

# #     # Даунсемплинг для ускорения обработки (если нужно)
# #     pcd_filtered = pcd_filtered.voxel_down_sample(voxel_size=0.005)

# #     # Смещение
# #     pcd_filtered.translate((shift_x, 0.0, shift_z))

# #     # Поворот на угол вокруг оси Y (в радианах)
# #     theta = np.radians(angle)
# #     R = pcd_filtered.get_rotation_matrix_from_axis_angle([0, theta, 0])
# #     pcd_filtered.rotate(R, center=(0, 0, 0))

# #     # Добавляем в список облаков для финальной визуализации
# #     all_pcds.append(pcd_filtered)

# #     # Сохранение результата
# #     output_path = os.path.join(output_folder, f"cloud_aligned_{angle}.ply")
# #     o3d.io.write_point_cloud(output_path, pcd_filtered)
# #     print(f"[✓] Обработано: {output_path}")

# # # Визуализация всех облаков
# # o3d.visualization.draw_geometries(all_pcds, window_name="3D Сканирование объекта", width=800, height=600)



# ##################################################   интерактив выравнивание  ###########################################################
# # import open3d as o3d
# # import numpy as np
# # import os

# # # === Пути к файлам ===
# # cloud0_path = os.path.join("pointclouds", "cloud_0.ply")
# # cloud180_path = os.path.join("pointclouds", "cloud_180.ply")

# # if not os.path.exists(cloud0_path) or not os.path.exists(cloud180_path):
# #     print("[Error] Один из файлов не найден!")
# #     exit()

# # # === Загрузка и фильтрация ===
# # def load_filtered_cloud(path):
# #     pcd = o3d.io.read_point_cloud(path)
# #     cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
# #     return pcd.select_by_index(ind)

# # cloud0 = load_filtered_cloud(cloud0_path)
# # cloud180 = load_filtered_cloud(cloud180_path)

# # # === Поворот cloud_180 на 180° вокруг оси Y ===
# # rotation_180_y = np.array([
# #     [-1,  0,  0, 0],
# #     [ 0,  1,  0, 0],
# #     [ 0,  0, -1, 0],
# #     [ 0,  0,  0, 1]
# # ])
# # cloud180.transform(rotation_180_y)

# # # === Координатная система ===
# # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

# # # === Интерактивный визуализатор ===
# # vis = o3d.visualization.VisualizerWithKeyCallback()
# # vis.create_window(window_name="Move Clouds Along X and Z")

# # # Добавляем в сцену
# # vis.add_geometry(cloud0)
# # vis.add_geometry(cloud180)
# # vis.add_geometry(axis)

# # # === Смещения ===
# # offsets = {
# #     "cloud0_z": 0.0,
# #     "cloud180_z": 0.0,
# #     "cloud0_x": 0.0,
# #     "cloud180_x": 0.0
# # }

# # # === Вспомогательная функция ===
# # def translate_cloud(cloud, axis_index, delta):
# #     T = np.eye(4)
# #     T[axis_index, 3] = delta
# #     cloud.transform(T)

# # # === Колбэки для cloud_0 ===
# # def move_c0_forward(vis):
# #     offsets["cloud0_z"] += 0.01
# #     translate_cloud(cloud0, 2, 0.01)
# #     vis.update_geometry(cloud0)
# #     print(f"[cloud_0] Z: {offsets['cloud0_z']:.3f}, X: {offsets['cloud0_x']:.3f}")
# #     return False

# # def move_c0_backward(vis):
# #     offsets["cloud0_z"] -= 0.01
# #     translate_cloud(cloud0, 2, -0.01)
# #     vis.update_geometry(cloud0)
# #     print(f"[cloud_0] Z: {offsets['cloud0_z']:.3f}, X: {offsets['cloud0_x']:.3f}")
# #     return False

# # def move_c0_left(vis):
# #     offsets["cloud0_x"] -= 0.01
# #     translate_cloud(cloud0, 0, -0.01)
# #     vis.update_geometry(cloud0)
# #     print(f"[cloud_0] Z: {offsets['cloud0_z']:.3f}, X: {offsets['cloud0_x']:.3f}")
# #     return False

# # def move_c0_right(vis):
# #     offsets["cloud0_x"] += 0.01
# #     translate_cloud(cloud0, 0, 0.01)
# #     vis.update_geometry(cloud0)
# #     print(f"[cloud_0] Z: {offsets['cloud0_z']:.3f}, X: {offsets['cloud0_x']:.3f}")
# #     return False

# # # === Колбэки для cloud_180 ===
# # def move_c180_forward(vis):
# #     offsets["cloud180_z"] += 0.01
# #     translate_cloud(cloud180, 2, 0.01)
# #     vis.update_geometry(cloud180)
# #     print(f"[cloud_180] Z: {offsets['cloud180_z']:.3f}, X: {offsets['cloud180_x']:.3f}")
# #     return False

# # def move_c180_backward(vis):
# #     offsets["cloud180_z"] -= 0.01
# #     translate_cloud(cloud180, 2, -0.01)
# #     vis.update_geometry(cloud180)
# #     print(f"[cloud_180] Z: {offsets['cloud180_z']:.3f}, X: {offsets['cloud180_x']:.3f}")
# #     return False

# # def move_c180_left(vis):
# #     offsets["cloud180_x"] -= 0.01
# #     translate_cloud(cloud180, 0, -0.01)
# #     vis.update_geometry(cloud180)
# #     print(f"[cloud_180] Z: {offsets['cloud180_z']:.3f}, X: {offsets['cloud180_x']:.3f}")
# #     return False

# # def move_c180_right(vis):
# #     offsets["cloud180_x"] += 0.01
# #     translate_cloud(cloud180, 0, 0.01)
# #     vis.update_geometry(cloud180)
# #     print(f"[cloud_180] Z: {offsets['cloud180_z']:.3f}, X: {offsets['cloud180_x']:.3f}")
# #     return False

# # # === Назначение клавиш ===
# # vis.register_key_callback(ord("W"), move_c0_forward)
# # vis.register_key_callback(ord("S"), move_c0_backward)
# # vis.register_key_callback(ord("Z"), move_c0_left)
# # vis.register_key_callback(ord("X"), move_c0_right)

# # vis.register_key_callback(ord("R"), move_c180_forward)
# # vis.register_key_callback(ord("F"), move_c180_backward)
# # vis.register_key_callback(ord("C"), move_c180_left)
# # vis.register_key_callback(ord("V"), move_c180_right)

# # # === Запуск ===
# # vis.run()
# # vis.destroy_window()
