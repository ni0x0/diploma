# криво косо но собирает фулл

import open3d as o3d
import numpy as np
import os
import copy
import time
import matplotlib.pyplot as plt

# ===== НАСТРОЙКИ =====
VOXEL_SIZE = 0.002  # Размер вокселя для даунсэмплинга
ICP_DISTANCE_COARSE = 0.008  # Порог для грубой ICP
ICP_DISTANCE_FINE = 0.003   # Порог для точной ICP
VISUALIZE_STEPS = True      # Визуализация промежуточных этапов
USE_COLORED_ICP = True      # Использовать цветную ICP
SAVE_INTERMEDIATE = True    # Сохранять промежуточные результаты
OPTIMIZE_SPEED = False      # Оптимизация скорости (менее точная)
GLOBAL_LOOP_CLOSURE = True  # Глобальная оптимизация замыкания

# Углы сканирования
ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
BASE_TRANSFORM = np.array([
    [1, 0, 0, -0.060],
    [0, 1, 0,    0.0],
    [0, 0, 1, -1.420],
    [0, 0, 0,    1.0]
])

# === Вспомогательные функции ===
def print_header(title):
    print("\n" + "="*50)
    print(f"=== {title.upper()}")
    print("="*50)

def timeit(func):
    """Декоратор для измерения времени выполнения"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Время выполнения {func.__name__}: {end-start:.2f} сек")
        return result
    return wrapper

def visualize_registration_step(source, target, transform=None, title="Регистрация"):
    """Визуализация процесса регистрации"""
    if not VISUALIZE_STEPS:
        return
    
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    # Применяем преобразование если есть
    if transform is not None:
        source_temp.transform(transform)
    
    # Раскрашиваем облака
    source_temp.paint_uniform_color([1, 0, 0])  # Красный - источник
    target_temp.paint_uniform_color([0, 0, 1])  # Синий - цель
    
    # Координатные оси
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    o3d.visualization.draw_geometries([source_temp, target_temp, axis], 
                                     window_name=title,
                                     width=800, height=600)

def visualize_merged(clouds, title="Объединенное облако"):
    """Визуализация нескольких объединенных облаков"""
    if not VISUALIZE_STEPS:
        return
    
    # Создаем копии для визуализации
    vis_objects = []
    colors = plt.cm.tab20.colors
    
    for i, cloud in enumerate(clouds):
        temp = copy.deepcopy(cloud)
        if not temp.has_colors():
            temp.paint_uniform_color(colors[i % len(colors)][:3])
        vis_objects.append(temp)
    
    # Добавляем оси координат
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis_objects.append(axis)
    
    o3d.visualization.draw_geometries(vis_objects, 
                                     window_name=title,
                                     width=800, height=600)

# === Основные функции обработки ===
@timeit
def load_and_preprocess(path, angle):
    """Загрузка и подготовка облака точек"""
    print(f"\nЗагрузка {os.path.basename(path)}...")
    cloud = o3d.io.read_point_cloud(path)
    
    if not cloud.has_points():
        raise ValueError(f"Облако {path} пустое!")
    
    # Фильтрация выбросов
    cl, ind = cloud.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.0)
    cloud = cloud.select_by_index(ind)
    
    # Применение начальных преобразований
    cloud.transform(BASE_TRANSFORM)
    if angle != 0:
        rotate_cloud(cloud, angle)
    
    # Расчет нормалей
    radius_normal = VOXEL_SIZE * 2.0
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_normal, max_nn=80))
    
    return cloud

def rotate_cloud(cloud, angle_deg):
    """Точный поворот облака вокруг оси Y"""
    angle_rad = np.deg2rad(angle_deg)
    rotation = np.array([
        [ np.cos(angle_rad), 0, np.sin(angle_rad), 0],
        [ 0, 1, 0, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
        [ 0, 0, 0, 1]
    ])
    cloud.transform(rotation)

@timeit
def enhanced_global_registration(source, target):
    """Улучшенная глобальная регистрация"""
    print("\nГлобальная регистрация (RANSAC):")
    voxel_sizes = [VOXEL_SIZE * 3, VOXEL_SIZE * 2, VOXEL_SIZE]
    best_result = None
    
    for i, voxel_size in enumerate(voxel_sizes):
        print(f"Попытка {i+1} с вокселем {voxel_size*1000:.1f} мм")
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        
        # Оценка нормалей
        radius_normal = voxel_size * 3
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=100))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=100))
        
        # Вычисление FPFH признаков
        radius_feature = voxel_size * 6
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down, o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_feature, max_nn=200))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down, o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_feature, max_nn=200))
        
        # Глобальная регистрация
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            voxel_size * 3,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            5, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 3),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(np.deg2rad(20))
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(2000000, 0.999))
        
        print(f"  Fitness: {result.fitness:.3f}, Inliers: {result.fitness*100:.1f}%")
        if best_result is None or result.fitness > best_result.fitness:
            best_result = result
    
    if best_result is None or best_result.fitness < 0.3:
        print("Глобальная регистрация не удалась, используем грубую инициализацию")
        source_center = source.get_center()
        target_center = target.get_center()
        translation = target_center - source_center
        rough_transform = np.identity(4)
        rough_transform[:3, 3] = translation
        return rough_transform
    
    print(f"Лучший результат: fitness={best_result.fitness:.3f}")
    return best_result.transformation

@timeit
def multi_scale_icp(source, target, initial_transform):
    """Многоуровневая ICP-регистрация"""
    thresholds = [ICP_DISTANCE_COARSE, ICP_DISTANCE_FINE]
    current_transform = initial_transform
    
    for level, threshold in enumerate(thresholds):
        print(f"\nICP Уровень {level+1} (порог: {threshold*1000:.1f} мм)")
        
        # Визуализация перед ICP
        visualize_registration_step(source, target, current_transform, 
                                  f"Перед ICP Уровень {level+1}")
        
        # Выбор метода ICP
        if USE_COLORED_ICP and source.has_colors() and target.has_colors():
            print("Используется Colored ICP")
            estimation = o3d.pipelines.registration.TransformationEstimationForColoredICP()
        else:
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        
        # Выполнение ICP
        result = o3d.pipelines.registration.registration_icp(
            source, target, threshold, current_transform,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=100))
        
        current_transform = result.transformation
        print(f"  Fitness: {result.fitness:.3f}, RMSE: {result.inlier_rmse:.5f} м")
        
        # Визуализация после ICP
        visualize_registration_step(source, target, current_transform,
                                  f"После ICP Уровень {level+1}")
    
    return current_transform

@timeit
def pairwise_registration(source, target, use_global=True):
    """Попарная регистрация"""
    print_header(f"Регистрация облаков")
    
    # Визуализация перед регистрацией
    visualize_registration_step(source, target, None, "Перед регистрацией")
    
    # Глобальная регистрация
    if use_global:
        transformation = enhanced_global_registration(source, target)
        visualize_registration_step(source, target, transformation, 
                                  "После глобальной регистрации")
    else:
        transformation = np.identity(4)
    
    # Многоуровневая ICP
    final_transform = multi_scale_icp(source, target, transformation)
    
    # Визуализация финального результата
    visualize_registration_step(source, target, final_transform, 
                              "Финальный результат регистрации")
    
    return final_transform

@timeit
def incremental_registration(clouds):
    """Последовательная регистрация с визуализацией"""
    print_header("Последовательная регистрация")
    
    merged_cloud = clouds[0]
    transformations = [np.identity(4)]
    
    for i in range(1, len(clouds)):
        print(f"\nРегистрация облака {i} ({ANGLES[i]}°)")
        
        # Визуализация перед объединением
        visualize_merged([merged_cloud, clouds[i]], f"Перед регистрацией {i}")
        
        # Регистрация
        transform = pairwise_registration(clouds[i], merged_cloud)
        transformations.append(transform)
        clouds[i].transform(transform)
        
        # Объединение
        merged_cloud += clouds[i]
        
        # Визуализация после объединения
        visualize_merged([merged_cloud], f"После объединения {i}")
        
        if SAVE_INTERMEDIATE:
            o3d.io.write_point_cloud(f"intermediate_{i}.ply", merged_cloud)
    
    return merged_cloud

@timeit
def postprocess_merged_cloud(cloud):
    """Постобработка объединенного облака"""
    print_header("Постобработка")
    
    # Визуализация до обработки
    visualize_merged([cloud], "До постобработки")
    
    # 1. Удаление дубликатов
    cloud = cloud.voxel_down_sample(VOXEL_SIZE)
    visualize_merged([cloud], "После удаления дубликатов")
    
    # 2. Удаление выбросов
    cl, ind = cloud.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.8)
    cloud = cl.select_by_index(ind)
    visualize_merged([cloud], "После удаления выбросов")
    
    # 3. Улучшение нормалей
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=VOXEL_SIZE*3, max_nn=50))
    
    return cloud

@timeit
def main():
    print_header("Начало обработки")
    
    # 1. Загрузка облаков
    clouds = []
    for angle in ANGLES:
        path = os.path.join("pointclouds", f"cloud_{angle}.ply")
        try:
            cloud = load_and_preprocess(path, angle)
            clouds.append(cloud)
        except Exception as e:
            print(f"Ошибка: {str(e)}")
            return
    
    # 2. Регистрация
    final_cloud = incremental_registration(clouds)
    
    # 3. Постобработка
    final_cloud = postprocess_merged_cloud(final_cloud)
    
    # 4. Финальный результат
    print_header("Финальный результат")
    
    # Раскрашиваем финальное облако
    final_cloud.paint_uniform_color([0.7, 0.7, 0.7])
    
    # Создаем координатные оси
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Визуализация с осями
    o3d.visualization.draw_geometries([final_cloud, axis], 
                                    window_name="Финальная реконструкция",
                                    width=1200, height=800)
    
    # Сохранение
    output_path = "final_reconstruction.ply"
    o3d.io.write_point_cloud(output_path, final_cloud)
    print(f"\nФинальное облако сохранено как '{output_path}'")
    
    # Явное сообщение о завершении
    print("\nОбработка завершена успешно!")
    print("Финальное облако сохранено и отображено.")

if __name__ == "__main__":
    main()


# import open3d as o3d
# import numpy as np
# import os
# import copy
# from collections import deque
# import time

# # ===== НАСТРОЙКИ =====
# VOXEL_SIZE = 0.002  # Уменьшенный размер вокселя для большей точности
# ICP_DISTANCE_COARSE = 0.008  # Порог для грубой ICP
# ICP_DISTANCE_FINE = 0.003   # Порог для точной ICP
# VISUALIZE_STEPS = True      # Визуализация промежуточных этапов
# USE_COLORED_ICP = False     # Использовать цветную информацию
# SAVE_INTERMEDIATE = True    # Сохранять промежуточные результаты

# # Углы сканирования
# ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
# BASE_TRANSFORM = np.array([
#     [1, 0, 0, -0.060],
#     [0, 1, 0,    0.0],
#     [0, 0, 1, -1.420],
#     [0, 0, 0,    1.0]
# ])

# # === Вспомогательные функции ===
# def print_header(title):
#     """Красивый вывод заголовков"""
#     print("\n" + "="*50)
#     print(f"=== {title.upper()}")
#     print("="*50)

# def timeit(func):
#     """Декоратор для измерения времени выполнения"""
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         result = func(*args, **kwargs)
#         end = time.time()
#         print(f"Время выполнения {func.__name__}: {end-start:.2f} сек")
#         return result
#     return wrapper

# # === Основные функции обработки ===
# @timeit
# def load_and_preprocess(path, angle):
#     """Загрузка и подготовка облака точек"""
#     print(f"\nЗагрузка {os.path.basename(path)}...")
#     cloud = o3d.io.read_point_cloud(path)
    
#     if not cloud.has_points():
#         raise ValueError(f"Облако {path} пустое!")
    
#     # Фильтрация выбросов (более агрессивная)
#     cl, ind = cloud.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.0)
#     inliers = len(ind)/len(cloud.points)
#     print(f"Фильтрация выбросов: сохранено {inliers:.1%} точек")
#     cloud = cloud.select_by_index(ind)
    
#     # Применение начальных преобразований
#     cloud.transform(BASE_TRANSFORM)
#     if angle != 0:
#         rotate_cloud(cloud, angle)
    
#     # Расчет нормалей с увеличенными параметрами
#     radius_normal = VOXEL_SIZE * 2.0
#     cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
#         radius=radius_normal, max_nn=80))
    
#     return cloud

# def rotate_cloud(cloud, angle_deg):
#     """Точный поворот облака вокруг оси Y"""
#     angle_rad = np.deg2rad(angle_deg)
#     rotation = np.array([
#         [ np.cos(angle_rad), 0, np.sin(angle_rad), 0],
#         [ 0,                 1, 0,                 0],
#         [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
#         [ 0,                 0, 0,                 1]
#     ])
#     cloud.transform(rotation)

# @timeit
# def enhanced_global_registration(source, target):
#     """Улучшенная глобальная регистрация с увеличенными параметрами"""
#     print("\nГлобальная регистрация (RANSAC):")
#     voxel_sizes = [VOXEL_SIZE * 3, VOXEL_SIZE * 2, VOXEL_SIZE]
#     best_result = None
    
#     for i, voxel_size in enumerate(voxel_sizes):
#         print(f"Попытка {i+1} с вокселем {voxel_size*1000:.1f} мм")
#         source_down = source.voxel_down_sample(voxel_size)
#         target_down = target.voxel_down_sample(voxel_size)
        
#         # Оценка нормалей с увеличенными параметрами
#         radius_normal = voxel_size * 3
#         source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
#             radius=radius_normal, max_nn=100))
#         target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
#             radius=radius_normal, max_nn=100))
        
#         # Вычисление FPFH признаков
#         radius_feature = voxel_size * 6
#         source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#             source_down, o3d.geometry.KDTreeSearchParamHybrid(
#                 radius=radius_feature, max_nn=200))
#         target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#             target_down, o3d.geometry.KDTreeSearchParamHybrid(
#                 radius=radius_feature, max_nn=200))
        
#         # Глобальная регистрация с увеличенными параметрами
#         result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#             source_down, target_down, source_fpfh, target_fpfh, True,
#             voxel_size * 2,
#             o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#             5, [
#                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
#                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 3),
#                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(np.deg2rad(20))
#             ],
#             o3d.pipelines.registration.RANSACConvergenceCriteria(2000000, 0.999))
        
#         print(f"  Fitness: {result.fitness:.3f}, Inliers: {result.fitness*100:.1f}%")
#         if best_result is None or result.fitness > best_result.fitness:
#             best_result = result
    
#     print(f"Лучший результат: fitness={best_result.fitness:.3f}")
#     return best_result.transformation if best_result else np.identity(4)

# @timeit
# def multi_scale_icp(source, target, initial_transform, max_iterations=[150, 100]):
#     """Многоуровневая ICP-регистрация с увеличенными параметрами"""
#     thresholds = [ICP_DISTANCE_COARSE, ICP_DISTANCE_FINE]
#     current_transform = initial_transform
    
#     for level, (threshold, max_iter) in enumerate(zip(thresholds, max_iterations)):
#         print(f"\nICP Уровень {level+1} (порог: {threshold*1000:.1f} мм):")
        
#         # Обновление нормалей с увеличенными параметрами
#         radius_normal = threshold * 3
#         source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
#             radius=radius_normal, max_nn=80))
#         target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
#             radius=radius_normal, max_nn=80))
        
#         # Выбор метода ICP
#         if USE_COLORED_ICP and source.has_colors() and target.has_colors():
#             print("Используется Colored ICP")
#             estimation = o3d.pipelines.registration.TransformationEstimationForColoredICP()
#         else:
#             estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        
#         # Выполнение ICP с увеличенными параметрами
#         result = o3d.pipelines.registration.registration_icp(
#             source, target, threshold, current_transform,
#             estimation,
#             o3d.pipelines.registration.ICPConvergenceCriteria(
#                 relative_fitness=1e-7,
#                 relative_rmse=1e-7,
#                 max_iteration=max_iter))
        
#         current_transform = result.transformation
#         print(f"  Fitness: {result.fitness:.3f}, RMSE: {result.inlier_rmse:.5f} м")
#         print(f"  Итераций: {max_iter}, Соответствий: {len(result.correspondence_set)}")
        
#         if VISUALIZE_STEPS:
#             temp = copy.deepcopy(source)
#             temp.transform(current_transform)
#             o3d.visualization.draw_geometries([temp, target])
    
#     return current_transform

# @timeit
# def pairwise_registration(source, target, use_global=True):
#     """Попарная регистрация с полным анализом"""
#     print_header(f"Регистрация облаков")
    
#     # Глобальная регистрация (если нужно)
#     if use_global:
#         transformation = enhanced_global_registration(source, target)
#         initial_fitness = evaluate_alignment(
#             copy.deepcopy(source).transform(transformation), 
#             target)
#         print(f"Качество после глобальной регистрации: {initial_fitness:.1%}")
#     else:
#         transformation = np.identity(4)
    
#     # Многоуровневая ICP
#     final_transform = multi_scale_icp(source, target, transformation)
    
#     # Финальная оценка
#     final_fitness = evaluate_alignment(
#         copy.deepcopy(source).transform(final_transform), 
#         target)
#     print(f"\nИтоговое качество выравнивания: {final_fitness:.1%}")
    
#     return final_transform

# @timeit
# def incremental_registration(clouds):
#     """Последовательная регистрация облаков точек"""
#     print_header("Последовательная регистрация")
    
#     # Начинаем с первого облака
#     merged_cloud = clouds[0]
#     transformations = [np.identity(4)]
    
#     for i in range(1, len(clouds)):
#         print(f"\nРегистрация облака {i} ({ANGLES[i]}°) к объединенному")
        
#         # Визуализация до регистрации
#         if VISUALIZE_STEPS:
#             temp_source = copy.deepcopy(clouds[i])
#             temp_target = copy.deepcopy(merged_cloud)
#             temp_source.paint_uniform_color([1, 0, 0])
#             temp_target.paint_uniform_color([0, 0, 1])
#             o3d.visualization.draw_geometries([temp_source, temp_target])
        
#         # Регистрация текущего облака к объединенному
#         transform = pairwise_registration(clouds[i], merged_cloud, use_global=True)
        
#         # Применяем преобразование к текущему облаку
#         clouds[i].transform(transform)
        
#         # Объединяем облака
#         merged_cloud += clouds[i]
        
#         # Сохранение промежуточных результатов
#         if SAVE_INTERMEDIATE:
#             o3d.io.write_point_cloud(f"intermediate_merged_{i}.ply", merged_cloud)
        
#         # Очистка памяти
#         merged_cloud = merged_cloud.voxel_down_sample(VOXEL_SIZE)
    
#     return merged_cloud

# @timeit
# def main():
#     print_header("Начало обработки")
    
#     # 1. Загрузка и предварительная обработка
#     print("\nЗагрузка и предварительная обработка облаков...")
#     clouds = {}
#     for angle in ANGLES:
#         path = os.path.join("pointclouds", f"cloud_{angle}.ply")
#         try:
#             clouds[angle] = load_and_preprocess(path, angle)
#         except Exception as e:
#             print(f"Ошибка: {str(e)}")
#             return
    
#     # Упорядочиваем облака по углам
#     ordered_clouds = [clouds[angle] for angle in ANGLES]
    
#     # 2. Последовательная регистрация
#     print_header("Последовательная регистрация")
#     final_cloud = incremental_registration(ordered_clouds)
    
#     # 3. Финальная обработка
#     print_header("Финальная обработка")
#     final_cloud.paint_uniform_color([0.7, 0.7, 0.7])
    
#     # Удаление выбросов
#     cl, ind = final_cloud.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
#     final_cloud = final_cloud.select_by_index(ind)
#     print(f"Удалено выбросов: {1 - len(ind)/len(cl.points):.1%}")
    
#     # Сохранение
#     output_path = "final_reconstruction.ply"
#     o3d.io.write_point_cloud(output_path, final_cloud)
#     print(f"\nФинальное облако сохранено как '{output_path}'")
    
#     # Визуализация
#     axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
#     o3d.visualization.draw_geometries([final_cloud, axis], 
#                                     window_name="Финальная реконструкция",
#                                     width=1200, height=800)

# if __name__ == "__main__":
#     main()