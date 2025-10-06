import open3d as o3d
import numpy as np
import os
import copy

# ===== НАСТРОЙКИ =====
# 1. Режимы регистрации
REGISTRATION_MODE = "advanced"  # "basic" или "advanced"
ICP_METHOD = "point_to_plane"   # "point_to_point" или "point_to_plane"

# 2. Параметры для basic-режима
THRESHOLD_METHOD = "iterative"  # "auto", "manual" или "iterative"
MANUAL_THRESHOLD = 0.02         # Используется если THRESHOLD_METHOD = "manual"

# 3. Параметры для advanced-режима
VOXEL_SIZE = 0.01               # Размер вокселя для даунсемплинга (меньше = точнее, но медленнее)
GLOBAL_DISTANCE_MULTIPLIER = 1.5 # Коэффициент для глобальной регистрации (1.5-2.0)
ICP_DISTANCE_MULTIPLIER = 0.4    # Коэффициент для ICP (0.3-0.5)

# 4. Визуализация
BACKGROUND_COLOR = [1.0, 1.0, 1.0]  # Белый фон [R,G,B]
POINT_SIZE = 2                      # Размер точек

# === Загрузка и фильтрация ===
def load_filtered_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd.select_by_index(ind)

# === Функции регистрации ===
def basic_icp_registration(source, target):
    def compute_threshold(pcd1, pcd2, method):
        if method == "manual":
            return MANUAL_THRESHOLD
        
        def avg_distance(pcd):
            return np.mean(pcd.compute_nearest_neighbor_distance())
        
        avg_dist = max(avg_distance(pcd1), avg_distance(pcd2))
        
        if method == "auto":
            return avg_dist * 2.5
        elif method == "iterative":
            print("\nИтеративный подбор threshold:")
            thresholds = [avg_dist * x for x in [1.5, 2.0, 2.5, 3.0]]
            best = None
            for th in thresholds:
                reg = o3d.pipelines.registration.registration_icp(
                    source, target, th, np.identity(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
                print(f"Threshold: {th:.4f} -> Fitness: {reg.fitness:.3f}, RMSE: {reg.inlier_rmse:.5f}")
                if best is None or reg.fitness > best[1]:
                    best = (th, reg.fitness)
            print(f"\nВыбран threshold: {best[0]:.4f} (fitness: {best[1]:.3f})")
            return best[0]

    threshold = compute_threshold(source, target, THRESHOLD_METHOD)
    
    if ICP_METHOD == "point_to_plane":
        radius_normal = threshold * 2
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    estimation_method = (o3d.pipelines.registration.TransformationEstimationPointToPlane() 
                        if ICP_METHOD == "point_to_plane" else 
                        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    return o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.identity(4), estimation_method, criteria)

def advanced_icp_registration(source, target):
    # 1. Даунсемплинг
    source_down = source.voxel_down_sample(VOXEL_SIZE)
    target_down = target.voxel_down_sample(VOXEL_SIZE)
    
    # 2. Оценка нормалей
    radius_normal = VOXEL_SIZE * 2
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # 3. FPFH признаки
    radius_feature = VOXEL_SIZE * 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    # 4. Глобальная регистрация (RANSAC)
    distance_threshold = VOXEL_SIZE * GLOBAL_DISTANCE_MULTIPLIER
    result_global = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    # 5. Применяем глобальное преобразование
    source_temp = copy.deepcopy(source)
    source_temp.transform(result_global.transformation)
    
    # 6. Оценка нормалей для полных облаков
    source_temp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal*2, max_nn=50))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal*2, max_nn=50))
    
    # 7. Точечная ICP-регистрация
    distance_threshold_icp = VOXEL_SIZE * ICP_DISTANCE_MULTIPLIER
    result_icp = o3d.pipelines.registration.registration_icp(
        source_temp, target, distance_threshold_icp, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    
    return result_icp.transformation @ result_global.transformation

# === Основной код ===
def main():
    # Загрузка данных
    cloud0 = load_filtered_cloud(os.path.join("pointclouds", "cloud_0.ply"))
    cloud30 = load_filtered_cloud(os.path.join("pointclouds", "cloud_30.ply"))

    # Начальные преобразования
    initial_translation = np.array([
        [1, 0, 0, -0.060],
        [0, 1, 0,    0.0],
        [0, 0, 1, -1.420],
        [0, 0, 0,    1.0]
    ])
    cloud0.transform(initial_translation)
    cloud30.transform(initial_translation)

    angle_rad = np.deg2rad(30)
    rotation_30_y = np.array([
        [ np.cos(angle_rad), 0, np.sin(angle_rad), 0],
        [ 0,                 1, 0,                 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
        [ 0,                 0, 0,                 1]
    ])
    cloud30.transform(rotation_30_y)

    # Выбор метода регистрации
    if REGISTRATION_MODE == "advanced":
        print("=== Режим: Advanced ICP с RANSAC ===")
        transformation = advanced_icp_registration(cloud30, cloud0)
    else:
        print("=== Режим: Basic ICP ===")
        result = basic_icp_registration(cloud30, cloud0)
        transformation = result.transformation

    # Применение преобразования
    cloud30.transform(transformation)

    # Визуализация
    def custom_visualize(*geometries):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for geom in geometries:
            vis.add_geometry(geom)
        opt = vis.get_render_option()
        opt.background_color = np.asarray(BACKGROUND_COLOR)
        opt.light_on = True
        opt.point_size = POINT_SIZE
        vis.run()
        vis.destroy_window()

    # Первая визуализация: цветное сопоставление
    temp_cloud0 = o3d.geometry.PointCloud(cloud0)
    temp_cloud30 = o3d.geometry.PointCloud(cloud30)
    temp_cloud0.paint_uniform_color([0.9, 0.1, 0.1])  # Красный
    temp_cloud30.paint_uniform_color([0.1, 0.1, 0.9]) # Синий
    print("\nВизуализация совмещения (красный-синий)...")
    custom_visualize(temp_cloud0, temp_cloud30)

    # Вторая визуализация: оригинальные цвета
    print("\nВизуализация объединенного облака...")
    merged_cloud = cloud0 + cloud30
    custom_visualize(merged_cloud)

    # Сохранение результата
    o3d.io.write_point_cloud("merged_cloud.ply", merged_cloud)
    print("\nРезультаты сохранены в 'merged_cloud.ply'")
    print("Матрица преобразования:\n", transformation)

if __name__ == "__main__":
    main()


# import open3d as o3d
# import numpy as np
# import os

# # === Настройки ===
# THRESHOLD_METHOD = "iterative"  # Выберите: "auto", "manual" или "iterative"
# MANUAL_THRESHOLD = 0.02    # Используется если THRESHOLD_METHOD = "manual"
# ICP_METHOD = "point_to_plane"  # "point_to_point" или "point_to_plane"

# # === Пути ===
# cloud0_path = os.path.join("pointclouds", "cloud_0.ply")
# cloud30_path = os.path.join("pointclouds", "cloud_30.ply")

# if not os.path.exists(cloud0_path) or not os.path.exists(cloud30_path):
#     print("[Error] Один из файлов не найден!")
#     exit()

# # === Загрузка и фильтрация ===
# def load_filtered_cloud(path):
#     pcd = o3d.io.read_point_cloud(path)
#     cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
#     return pcd.select_by_index(ind)

# cloud0 = load_filtered_cloud(cloud0_path)
# cloud30 = load_filtered_cloud(cloud30_path)

# # === Преобразования ===
# initial_translation = np.array([
#     [1, 0, 0, -0.060],
#     [0, 1, 0,    0.0],
#     [0, 0, 1, -1.420],
#     [0, 0, 0,    1.0]
# ])
# cloud0.transform(initial_translation)
# cloud30.transform(initial_translation)

# angle_rad = np.deg2rad(30)
# rotation_30_y = np.array([
#     [ np.cos(angle_rad), 0, np.sin(angle_rad), 0],
#     [ 0,                 1, 0,                 0],
#     [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
#     [ 0,                 0, 0,                 1]
# ])
# cloud30.transform(rotation_30_y)

# # === Функция визуализации ===
# def draw_registration_result(source, target, transformation=None):
#     source_temp = o3d.geometry.PointCloud(source)  # Создаем копию
#     target_temp = o3d.geometry.PointCloud(target)  # Создаем копию
    
#     if transformation is not None:
#         source_temp.transform(transformation)
    
#     source_temp.paint_uniform_color([1, 0, 0])  # Красный
#     target_temp.paint_uniform_color([0, 1, 0])  # Зеленый
    
#     o3d.visualization.draw_geometries([source_temp, target_temp],
#                                      zoom=0.5,
#                                      front=[0, -1, 0],
#                                      lookat=[0, 0, 0],
#                                      up=[0, 0, 1])

# # === Подбор threshold ===
# def compute_threshold(pcd1, pcd2, method):
#     if method == "manual":
#         return MANUAL_THRESHOLD
    
#     # Вычисляем средние расстояния между точками
#     def avg_distance(pcd):
#         distances = pcd.compute_nearest_neighbor_distance()
#         return np.mean(distances)
    
#     avg_dist = max(avg_distance(pcd1), avg_distance(pcd2))
    
#     if method == "auto":
#         return avg_dist * 2.5  # Эмпирический коэффициент
    
#     elif method == "iterative":
#         print("\nИтеративный подбор threshold:")
#         thresholds = [avg_dist * x for x in [1.5, 2.0, 2.5, 3.0]]
#         best = None
        
#         for th in thresholds:
#             reg = o3d.pipelines.registration.registration_icp(
#                 cloud30, cloud0, th, np.identity(4),
#                 o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#                 o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
            
#             print(f"Threshold: {th:.4f} -> Fitness: {reg.fitness:.3f}, RMSE: {reg.inlier_rmse:.5f}")
            
#             if best is None or reg.fitness > best[1]:
#                 best = (th, reg.fitness)
        
#         print(f"\nВыбран threshold: {best[0]:.4f} (fitness: {best[1]:.3f})")
#         return best[0]

# # Вычисляем threshold
# threshold = compute_threshold(cloud0, cloud30, THRESHOLD_METHOD)
# print(f"\nИспользуемый threshold: {threshold:.4f}")

# # === Вычисление нормалей (для point-to-plane) ===
# if ICP_METHOD == "point_to_plane":
#     print("Вычисление нормалей...")
#     radius_normal = threshold * 2
#     cloud0.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
#         radius=radius_normal, max_nn=30))
#     cloud30.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
#         radius=radius_normal, max_nn=30))

# # === ICP регистрация ===
# print("\nВыполнение ICP...")
# initial_transform = np.identity(4)

# if ICP_METHOD == "point_to_point":
#     reg_result = o3d.pipelines.registration.registration_icp(
#         cloud30, cloud0, threshold, initial_transform,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
# else:
#     reg_result = o3d.pipelines.registration.registration_icp(
#         cloud30, cloud0, threshold, initial_transform,
#         o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

# # === Вывод результатов ===
# print("\nРезультаты регистрации:")
# print(f"Метод: {ICP_METHOD}, Threshold: {threshold:.4f}")
# print(f"Fitness: {reg_result.fitness:.3f} (1 - идеально)")
# print(f"RMSE: {reg_result.inlier_rmse:.5f}")
# print("\nМатрица преобразования:")
# print(reg_result.transformation)

# # === Визуализация финального результата ===
# print("\nВизуализация результата...")
# draw_registration_result(cloud30, cloud0, reg_result.transformation)

# # === Объединение облаков ===
# cloud30.transform(reg_result.transformation)
# merged_cloud = cloud0 + cloud30

# print("\nВизуализация объединенного облака...")
# o3d.visualization.draw_geometries([merged_cloud],
#                                  zoom=0.5,
#                                  front=[0, -1, 0],
#                                  lookat=[0, 0, 0],
#                                  up=[0, 0, 1])

# # import open3d as o3d
# # import numpy as np
# # import os

# # # === Пути ===
# # cloud0_path = os.path.join("pointclouds", "cloud_0.ply")
# # cloud30_path = os.path.join("pointclouds", "cloud_30.ply")

# # if not os.path.exists(cloud0_path) or not os.path.exists(cloud30_path):
# #     print("[Error] Один из файлов не найден!")
# #     exit()

# # # === Загрузка и фильтрация ===
# # def load_filtered_cloud(path):
# #     pcd = o3d.io.read_point_cloud(path)
# #     cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
# #     return pcd.select_by_index(ind)

# # cloud0 = load_filtered_cloud(cloud0_path)
# # cloud30 = load_filtered_cloud(cloud30_path)

# # # === Смещение в общую систему координат ===
# # initial_translation = np.array([
# #     [1, 0, 0, -0.060],
# #     [0, 1, 0,    0.0],
# #     [0, 0, 1, -1.420],
# #     [0, 0, 0,    1.0]
# # ])
# # cloud0.transform(initial_translation)
# # cloud30.transform(initial_translation)

# # # === Поворот cloud_30 на 30° вокруг Y (начальное приближение) ===
# # angle_rad = np.deg2rad(30)
# # rotation_30_y = np.array([
# #     [ np.cos(angle_rad), 0, np.sin(angle_rad), 0],
# #     [ 0,                 1, 0,                 0],
# #     [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
# #     [ 0,                 0, 0,                 1]
# # ])
# # cloud30.transform(rotation_30_y)

# # # === Порог для ICP (в метрах) ===
# # threshold = 0.02

# # # === Начальная трансформация: можно использовать просто identity, т.к. облако уже повёрнуто вручную ===
# # trans_init = np.identity(4)

# # # === Выполнение ICP ===
# # print("=== Старт ICP ===")
# # reg_p2p = o3d.pipelines.registration.registration_icp(
# #     cloud30, cloud0, threshold, trans_init,
# #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
# #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
# # )
# # print("ICP завершен. Полученная трансформация:")
# # print(reg_p2p.transformation)

# # # === Применим полученную трансформацию к cloud30 ===
# # cloud30.transform(reg_p2p.transformation)

# # # === Визуализация результата ===
# # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

# # o3d.visualization.draw_geometries([cloud0, cloud30, axis],
# #                                   window_name="ICP Result",
# #                                   width=800, height=600)
