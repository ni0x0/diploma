import open3d as o3d
import numpy as np
import os
import copy
from collections import deque
import time

# ===== НАСТРОЙКИ =====
VOXEL_SIZE = 0.003  # Базовый размер вокселя (3 мм)
ICP_DISTANCE_COARSE = 0.01  # Порог для грубой ICP
ICP_DISTANCE_FINE = 0.005   # Порог для точной ICP
VISUALIZE_STEPS = True      # Визуализация промежуточных этапов
USE_COLORED_ICP = False     # Использовать цветную информацию
SAVE_INTERMEDIATE = True    # Сохранять промежуточные результаты

# Углы сканирования
ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
BASE_TRANSFORM = np.array([
    [1, 0, 0, -0.060],
    [0, 1, 0,    0.0],
    [0, 0, 1, -1.420],
    [0, 0, 0,    1.0]
])

# Настройки для проблемных облаков
SPECIAL_GROUP_INDEX = 3  # Группа 270-300-330
SPECIAL_PARAMS = {
    'voxel_size': 0.002,
    'icp_coarse': 0.007,
    'icp_fine': 0.003,
    'normal_radius': 0.004,
    'max_nn_normal': 70
}

# === Вспомогательные функции ===
def print_header(title):
    """Красивый вывод заголовков"""
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

# === Основные функции обработки ===
@timeit
def load_and_preprocess(path, angle):
    """Загрузка и подготовка облака точек"""
    print(f"\nЗагрузка {os.path.basename(path)}...")
    cloud = o3d.io.read_point_cloud(path)
    
    if not cloud.has_points():
        raise ValueError(f"Облако {path} пустое!")
    
    # Фильтрация выбросов
    cl, ind = cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    inliers = len(ind)/len(cloud.points)
    print(f"Фильтрация выбросов: сохранено {inliers:.1%} точек")
    cloud = cloud.select_by_index(ind)
    
    # Применение начальных преобразований
    cloud.transform(BASE_TRANSFORM)
    if angle != 0:
        rotate_cloud(cloud, angle)
    
    # Расчет нормалей
    radius_normal = VOXEL_SIZE * 1.5
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_normal, max_nn=50))
    
    return cloud

def rotate_cloud(cloud, angle_deg):
    """Точный поворот облака вокруг оси Y"""
    angle_rad = np.deg2rad(angle_deg)
    rotation = np.array([
        [ np.cos(angle_rad), 0, np.sin(angle_rad), 0],
        [ 0,                 1, 0,                 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
        [ 0,                 0, 0,                 1]
    ])
    cloud.transform(rotation)

@timeit
def enhanced_global_registration(source, target):
    """Улучшенная глобальная регистрация"""
    print("\nГлобальная регистрация (RANSAC):")
    voxel_sizes = [VOXEL_SIZE * 3, VOXEL_SIZE * 1.5, VOXEL_SIZE]
    best_result = None
    
    for i, voxel_size in enumerate(voxel_sizes):
        print(f"Попытка {i+1} с вокселем {voxel_size*1000:.1f} мм")
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        
        # Оценка нормалей
        radius_normal = voxel_size * 2
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=70))  # Увеличено max_nn
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=70))
        
        # Вычисление FPFH признаков
        radius_feature = voxel_size * 5
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down, o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_feature, max_nn=150))  # Увеличено max_nn
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down, o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_feature, max_nn=150))
        
        # Глобальная регистрация с более строгими параметрами
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            voxel_size * 1.5,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.7),  # Более мягкий порог
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 2),  # Увеличенный порог
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(np.deg2rad(15))  # Более мягкий угол
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999))  # Больше итераций
        
        print(f"  Fitness: {result.fitness:.3f}, Inliers: {result.fitness*100:.1f}%")
        if best_result is None or result.fitness > best_result.fitness:
            best_result = result
    
    print(f"Лучший результат: fitness={best_result.fitness:.3f}")
    return best_result.transformation if best_result else np.identity(4)

@timeit
def multi_scale_icp(source, target, initial_transform, max_iterations=[100, 50]):
    """Многоуровневая ICP-регистрация"""
    thresholds = [ICP_DISTANCE_COARSE, ICP_DISTANCE_FINE]
    current_transform = initial_transform
    
    for level, (threshold, max_iter) in enumerate(zip(thresholds, max_iterations)):
        print(f"\nICP Уровень {level+1} (порог: {threshold*1000:.1f} мм):")
        
        # Обновление нормалей
        radius_normal = threshold * 2
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=50))
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=50))
        
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
                max_iteration=max_iter))
        
        current_transform = result.transformation
        print(f"  Fitness: {result.fitness:.3f}, RMSE: {result.inlier_rmse:.5f} м")
        
        # Исправленная строка - используем len() вместо .size
        print(f"  Итераций: {max_iter}, Соответствий: {len(result.correspondence_set)}")
        
        if VISUALIZE_STEPS:
            temp = copy.deepcopy(source)
            temp.transform(current_transform)
            o3d.visualization.draw_geometries([temp, target])
    
    return current_transform

def evaluate_alignment(source, target, threshold=0.005):
    """Комплексная оценка качества выравнивания"""
    print("\nОценка качества выравнивания:")
    
    # Вычисление расстояний
    dist_source = np.asarray(source.compute_point_cloud_distance(target))
    dist_target = np.asarray(target.compute_point_cloud_distance(source))
    
    # Статистика
    inliers_source = np.sum(dist_source < threshold)
    inliers_target = np.sum(dist_target < threshold)
    overlap_source = inliers_source / len(dist_source)
    overlap_target = inliers_target / len(dist_target)
    mean_dist = (np.mean(dist_source) + np.mean(dist_target)) / 2
    median_dist = (np.median(dist_source) + np.median(dist_target)) / 2
    
    print(f"Перекрытие точек:")
    print(f"  Source -> Target: {overlap_source:.1%} (порог: {threshold*1000:.1f} мм)")
    print(f"  Target -> Source: {overlap_target:.1%}")
    print(f"Среднее расстояние: {mean_dist*1000:.2f} мм")
    print(f"Медианное расстояние: {median_dist*1000:.2f} мм")
    
    # Визуализация плохо совмещенных областей
    if VISUALIZE_STEPS:
        bad_points_source = np.where(dist_source > threshold*2)[0]
        bad_points_target = np.where(dist_target > threshold*2)[0]
        
        source_colored = copy.deepcopy(source)
        target_colored = copy.deepcopy(target)
        
        # Если нет цветов, инициализируем их
        if not source_colored.has_colors():
            source_colored.paint_uniform_color([0.5, 0.5, 0.5])
        if not target_colored.has_colors():
            target_colored.paint_uniform_color([0.5, 0.5, 0.5])
        
        colors = np.asarray(source_colored.colors)
        colors[bad_points_source] = [1, 0, 0]  # Красный для плохих точек
        source_colored.colors = o3d.utility.Vector3dVector(colors)
        
        colors = np.asarray(target_colored.colors)
        colors[bad_points_target] = [0, 0, 1]  # Синий для плохих точек
        target_colored.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.visualization.draw_geometries([source_colored, target_colored])
    
    return (overlap_source + overlap_target) / 2

@timeit
def pairwise_registration(source, target, use_global=True):
    """Попарная регистрация с полным анализом"""
    print_header(f"Регистрация облаков")
    
    # Глобальная регистрация (если нужно)
    if use_global:
        transformation = enhanced_global_registration(source, target)
        initial_fitness = evaluate_alignment(
            copy.deepcopy(source).transform(transformation), 
            target)
        print(f"Качество после глобальной регистрации: {initial_fitness:.1%}")
    else:
        transformation = np.identity(4)
    
    # Многоуровневая ICP
    final_transform = multi_scale_icp(source, target, transformation)
    
    # Финальная оценка
    final_fitness = evaluate_alignment(
        copy.deepcopy(source).transform(final_transform), 
        target)
    print(f"\nИтоговое качество выравнивания: {final_fitness:.1%}")
    
    return final_transform

@timeit
def merge_clouds(clouds_to_merge, group_name="", use_global=True):
    """Иерархическое объединение облаков"""
    print_header(f"Объединение группы {group_name}")
    queue = deque(clouds_to_merge)
    iteration = 1
    
    while len(queue) > 1:
        print(f"\nИтерация объединения {iteration}")
        source = queue.popleft()
        target = queue.popleft()
        
        # Визуализация до регистрации
        if VISUALIZE_STEPS:
            temp_source = copy.deepcopy(source)
            temp_target = copy.deepcopy(target)
            temp_source.paint_uniform_color([1, 0, 0])
            temp_target.paint_uniform_color([0, 0, 1])
            o3d.visualization.draw_geometries([temp_source, temp_target])
        
        # Регистрация
        transformation = pairwise_registration(
            source, target, 
            use_global=(use_global and iteration == 1))
        
        source.transform(transformation)
        merged = source + target
        
        # Сохранение промежуточных результатов
        if SAVE_INTERMEDIATE:
            o3d.io.write_point_cloud(
                f"intermediate_{group_name}_iter{iteration}.ply", 
                merged)
        
        queue.append(merged)
        iteration += 1
    
    return queue[0]

@timeit
def merge_groups_strategy(groups):
    """Стратегия сшивания групп с анализом перекрытий"""
    print_header("Стратегия объединения групп")
    
    # 1. Анализ перекрытий между группами
    print("\nАнализ перекрытий между группами:")
    overlap_matrix = np.zeros((len(groups), len(groups)))
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            overlap = evaluate_alignment(groups[i], groups[j], threshold=0.02)
            overlap_matrix[i,j] = overlap
            overlap_matrix[j,i] = overlap
    
    print("\nМатрица перекрытий:")
    print("      0     1     2     3")
    for i, row in enumerate(overlap_matrix):
        print(f"{i} | " + " ".join(f"{val:.2f}" if val > 0 else "  - " for val in row))
    
    # 2. Определение порядка сшивания на основе перекрытий
    best_pair = np.unravel_index(np.argmax(overlap_matrix), overlap_matrix.shape)
    print(f"\nНаибольшее перекрытие между группами {best_pair}")
    
    # 3. Объединение пар с наибольшим перекрытием
    print("\nОбъединение групп с наибольшим перекрытием...")
    merged_pairs = []
    remaining = set(range(len(groups)))
    
    while len(remaining) >= 2:
        # Находим пару с максимальным перекрытием
        max_overlap = -1
        best_pair = None
        for i in remaining:
            for j in remaining:
                if i != j and overlap_matrix[i,j] > max_overlap:
                    max_overlap = overlap_matrix[i,j]
                    best_pair = (i, j)
        
        if best_pair is None:
            break
            
        print(f"\nОбъединение групп {best_pair} (перекрытие: {max_overlap:.1%})")
        merged = merge_clouds([groups[best_pair[0]], groups[best_pair[1]]], 
                            f"{best_pair[0]}-{best_pair[1]}", 
                            use_global=True)
        merged_pairs.append(merged)
        remaining.remove(best_pair[0])
        remaining.remove(best_pair[1])
    
    # Добавляем оставшиеся группы
    for idx in remaining:
        merged_pairs.append(groups[idx])
    
    # 4. Финальное объединение
    print("\nФинальное объединение...")
    if len(merged_pairs) == 1:
        return merged_pairs[0]
    
    return merge_clouds(merged_pairs, "final", use_global=True)

# === Основной процесс ===
@timeit
def main():
    print_header("Начало обработки")
    
    # 1. Загрузка и предварительная обработка
    print("\nЗагрузка и предварительная обработка облаков...")
    angle_groups = [
        [0, 30, 60],      # Группа 0
        [90, 120, 150],   # Группа 1
        [180, 210, 240],  # Группа 2
        [270, 300, 330]   # Группа 3
    ]
    
    # Сохраняем глобальные параметры в локальные переменные
    global VOXEL_SIZE, ICP_DISTANCE_COARSE, ICP_DISTANCE_FINE
    original_voxel = VOXEL_SIZE
    original_coarse = ICP_DISTANCE_COARSE
    original_fine = ICP_DISTANCE_FINE
    
    clouds = {}
    for angle in ANGLES:
        path = os.path.join("pointclouds", f"cloud_{angle}.ply")
        try:
            clouds[angle] = load_and_preprocess(path, angle)
        except Exception as e:
            print(f"Ошибка: {str(e)}")
            return

    # 2. Объединение внутри групп
    print_header("Объединение внутри групп")
    group_results = []
    
    for i, group in enumerate(angle_groups):
        print(f"\nОбработка группы {i}: {group}")
        
        # Специальные параметры для проблемной группы
        if i == SPECIAL_GROUP_INDEX:
            print("Применение специальных параметров для проблемной группы")
            # Сохраняем оригинальные параметры
            orig_params = {
                'voxel': VOXEL_SIZE,
                'coarse': ICP_DISTANCE_COARSE,
                'fine': ICP_DISTANCE_FINE
            }
            # Устанавливаем специальные параметры
            VOXEL_SIZE = SPECIAL_PARAMS['voxel_size']
            ICP_DISTANCE_COARSE = SPECIAL_PARAMS['icp_coarse']
            ICP_DISTANCE_FINE = SPECIAL_PARAMS['icp_fine']
        
        group_clouds = [clouds[angle] for angle in group]
        merged_group = merge_clouds(group_clouds, f"group_{i}")
        group_results.append(merged_group)
        
        # Восстановление параметров
        if i == SPECIAL_GROUP_INDEX:
            VOXEL_SIZE = orig_params['voxel']
            ICP_DISTANCE_COARSE = orig_params['coarse']
            ICP_DISTANCE_FINE = orig_params['fine']
        
        # Сохранение промежуточных результатов
        if SAVE_INTERMEDIATE:
            o3d.io.write_point_cloud(f"group_{i}_merged.ply", merged_group)
    
    # Восстановление глобальных параметров на случай, если что-то пошло не так
    VOXEL_SIZE = original_voxel
    ICP_DISTANCE_COARSE = original_coarse
    ICP_DISTANCE_FINE = original_fine
    
    # 3. Объединение групп
    print_header("Объединение групп")
    final_cloud = merge_groups_strategy(group_results)
    
    # 4. Финальная обработка
    print_header("Финальная обработка")
    final_cloud.paint_uniform_color([0.7, 0.7, 0.7])
    
    # Удаление выбросов
    cl, ind = final_cloud.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    final_cloud = final_cloud.select_by_index(ind)
    print(f"Удалено выбросов: {1 - len(ind)/len(cl.points):.1%}")
    
    # Сохранение
    output_path = "final_reconstruction.ply"
    o3d.io.write_point_cloud(output_path, final_cloud)
    print(f"\nФинальное облако сохранено как '{output_path}'")
    
    # Визуализация
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([final_cloud, axis], 
                                    window_name="Финальная реконструкция",
                                    width=1200, height=800)

if __name__ == "__main__":
    main()


# import open3d as o3d
# import numpy as np
# import os
# import copy
# from collections import deque
# import time

# # ===== НАСТРОЙКИ =====
# VOXEL_SIZE = 0.003  # Базовый размер вокселя (3 мм)
# ICP_DISTANCE_COARSE = 0.01  # Порог для грубой ICP
# ICP_DISTANCE_FINE = 0.005   # Порог для точной ICP
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

# # Настройки для проблемных облаков
# SPECIAL_GROUP_INDEX = 3  # Группа 270-300-330
# SPECIAL_PARAMS = {
#     'voxel_size': 0.002,
#     'icp_coarse': 0.007,
#     'icp_fine': 0.003,
#     'normal_radius': 0.004,
#     'max_nn_normal': 70
# }

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
    
#     # Фильтрация выбросов
#     cl, ind = cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
#     inliers = len(ind)/len(cloud.points)
#     print(f"Фильтрация выбросов: сохранено {inliers:.1%} точек")
#     cloud = cloud.select_by_index(ind)
    
#     # Применение начальных преобразований
#     cloud.transform(BASE_TRANSFORM)
#     if angle != 0:
#         rotate_cloud(cloud, angle)
    
#     # Расчет нормалей
#     radius_normal = VOXEL_SIZE * 1.5
#     cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
#         radius=radius_normal, max_nn=50))
    
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
#     """Улучшенная глобальная регистрация"""
#     print("\nГлобальная регистрация (RANSAC):")
#     voxel_sizes = [VOXEL_SIZE * 3, VOXEL_SIZE * 1.5, VOXEL_SIZE]
#     best_result = None
    
#     for i, voxel_size in enumerate(voxel_sizes):
#         print(f"Попытка {i+1} с вокселем {voxel_size*1000:.1f} мм")
#         source_down = source.voxel_down_sample(voxel_size)
#         target_down = target.voxel_down_sample(voxel_size)
        
#         # Оценка нормалей
#         radius_normal = voxel_size * 2
#         source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
#             radius=radius_normal, max_nn=70))  # Увеличено max_nn
#         target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
#             radius=radius_normal, max_nn=70))
        
#         # Вычисление FPFH признаков
#         radius_feature = voxel_size * 5
#         source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#             source_down, o3d.geometry.KDTreeSearchParamHybrid(
#                 radius=radius_feature, max_nn=150))  # Увеличено max_nn
#         target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#             target_down, o3d.geometry.KDTreeSearchParamHybrid(
#                 radius=radius_feature, max_nn=150))
        
#         # Глобальная регистрация с более строгими параметрами
#         result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#             source_down, target_down, source_fpfh, target_fpfh, True,
#             voxel_size * 1.5,
#             o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#             4, [
#                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.7),  # Более мягкий порог
#                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 2),  # Увеличенный порог
#                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(np.deg2rad(15))  # Более мягкий угол
#             ],
#             o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999))  # Больше итераций
        
#         print(f"  Fitness: {result.fitness:.3f}, Inliers: {result.fitness*100:.1f}%")
#         if best_result is None or result.fitness > best_result.fitness:
#             best_result = result
    
#     print(f"Лучший результат: fitness={best_result.fitness:.3f}")
#     return best_result.transformation if best_result else np.identity(4)

# @timeit
# def multi_scale_icp(source, target, initial_transform, max_iterations=[100, 50]):
#     """Многоуровневая ICP-регистрация"""
#     thresholds = [ICP_DISTANCE_COARSE, ICP_DISTANCE_FINE]
#     current_transform = initial_transform
    
#     for level, (threshold, max_iter) in enumerate(zip(thresholds, max_iterations)):
#         print(f"\nICP Уровень {level+1} (порог: {threshold*1000:.1f} мм):")
        
#         # Обновление нормалей
#         radius_normal = threshold * 2
#         source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
#             radius=radius_normal, max_nn=50))
#         target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
#             radius=radius_normal, max_nn=50))
        
#         # Выбор метода ICP
#         if USE_COLORED_ICP and source.has_colors() and target.has_colors():
#             print("Используется Colored ICP")
#             estimation = o3d.pipelines.registration.TransformationEstimationForColoredICP()
#         else:
#             estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        
#         # Выполнение ICP
#         result = o3d.pipelines.registration.registration_icp(
#             source, target, threshold, current_transform,
#             estimation,
#             o3d.pipelines.registration.ICPConvergenceCriteria(
#                 relative_fitness=1e-6,
#                 relative_rmse=1e-6,
#                 max_iteration=max_iter))
        
#         current_transform = result.transformation
#         print(f"  Fitness: {result.fitness:.3f}, RMSE: {result.inlier_rmse:.5f} м")
        
#         # Исправленная строка - используем len() вместо .size
#         print(f"  Итераций: {max_iter}, Соответствий: {len(result.correspondence_set)}")
        
#         if VISUALIZE_STEPS:
#             temp = copy.deepcopy(source)
#             temp.transform(current_transform)
#             o3d.visualization.draw_geometries([temp, target])
    
#     return current_transform

# def evaluate_alignment(source, target, threshold=0.005):
#     """Комплексная оценка качества выравнивания"""
#     print("\nОценка качества выравнивания:")
    
#     # Вычисление расстояний
#     dist_source = np.asarray(source.compute_point_cloud_distance(target))
#     dist_target = np.asarray(target.compute_point_cloud_distance(source))
    
#     # Статистика
#     inliers_source = np.sum(dist_source < threshold)
#     inliers_target = np.sum(dist_target < threshold)
#     overlap_source = inliers_source / len(dist_source)
#     overlap_target = inliers_target / len(dist_target)
#     mean_dist = (np.mean(dist_source) + np.mean(dist_target)) / 2
#     median_dist = (np.median(dist_source) + np.median(dist_target)) / 2
    
#     print(f"Перекрытие точек:")
#     print(f"  Source -> Target: {overlap_source:.1%} (порог: {threshold*1000:.1f} мм)")
#     print(f"  Target -> Source: {overlap_target:.1%}")
#     print(f"Среднее расстояние: {mean_dist*1000:.2f} мм")
#     print(f"Медианное расстояние: {median_dist*1000:.2f} мм")
    
#     # Визуализация плохо совмещенных областей
#     if VISUALIZE_STEPS:
#         bad_points_source = np.where(dist_source > threshold*2)[0]
#         bad_points_target = np.where(dist_target > threshold*2)[0]
        
#         source_colored = copy.deepcopy(source)
#         target_colored = copy.deepcopy(target)
        
#         # Если нет цветов, инициализируем их
#         if not source_colored.has_colors():
#             source_colored.paint_uniform_color([0.5, 0.5, 0.5])
#         if not target_colored.has_colors():
#             target_colored.paint_uniform_color([0.5, 0.5, 0.5])
        
#         colors = np.asarray(source_colored.colors)
#         colors[bad_points_source] = [1, 0, 0]  # Красный для плохих точек
#         source_colored.colors = o3d.utility.Vector3dVector(colors)
        
#         colors = np.asarray(target_colored.colors)
#         colors[bad_points_target] = [0, 0, 1]  # Синий для плохих точек
#         target_colored.colors = o3d.utility.Vector3dVector(colors)
        
#         o3d.visualization.draw_geometries([source_colored, target_colored])
    
#     return (overlap_source + overlap_target) / 2

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
# def merge_clouds(clouds_to_merge, group_name="", use_global=True):
#     """Иерархическое объединение облаков"""
#     print_header(f"Объединение группы {group_name}")
#     queue = deque(clouds_to_merge)
#     iteration = 1
    
#     while len(queue) > 1:
#         print(f"\nИтерация объединения {iteration}")
#         source = queue.popleft()
#         target = queue.popleft()
        
#         # Визуализация до регистрации
#         if VISUALIZE_STEPS:
#             temp_source = copy.deepcopy(source)
#             temp_target = copy.deepcopy(target)
#             temp_source.paint_uniform_color([1, 0, 0])
#             temp_target.paint_uniform_color([0, 0, 1])
#             o3d.visualization.draw_geometries([temp_source, temp_target])
        
#         # Регистрация
#         transformation = pairwise_registration(
#             source, target, 
#             use_global=(use_global and iteration == 1))
        
#         source.transform(transformation)
#         merged = source + target
        
#         # Сохранение промежуточных результатов
#         if SAVE_INTERMEDIATE:
#             o3d.io.write_point_cloud(
#                 f"intermediate_{group_name}_iter{iteration}.ply", 
#                 merged)
        
#         queue.append(merged)
#         iteration += 1
    
#     return queue[0]

# @timeit
# def merge_groups_strategy(groups):
#     """Стратегия сшивания групп с анализом перекрытий"""
#     print_header("Стратегия объединения групп")
    
#     # 1. Анализ перекрытий между группами
#     print("\nАнализ перекрытий между группами:")
#     overlap_matrix = np.zeros((len(groups), len(groups)))
#     for i in range(len(groups)):
#         for j in range(i+1, len(groups)):
#             overlap = evaluate_alignment(groups[i], groups[j], threshold=0.02)
#             overlap_matrix[i,j] = overlap
#             overlap_matrix[j,i] = overlap
    
#     print("\nМатрица перекрытий:")
#     print("      0     1     2     3")
#     for i, row in enumerate(overlap_matrix):
#         print(f"{i} | " + " ".join(f"{val:.2f}" if val > 0 else "  - " for val in row))
    
#     # 2. Определение порядка сшивания на основе перекрытий
#     best_pair = np.unravel_index(np.argmax(overlap_matrix), overlap_matrix.shape)
#     print(f"\nНаибольшее перекрытие между группами {best_pair}")
    
#     # 3. Объединение пар с наибольшим перекрытием
#     print("\nОбъединение групп с наибольшим перекрытием...")
#     merged_pairs = []
#     remaining = set(range(len(groups)))
    
#     while len(remaining) >= 2:
#         # Находим пару с максимальным перекрытием
#         max_overlap = -1
#         best_pair = None
#         for i in remaining:
#             for j in remaining:
#                 if i != j and overlap_matrix[i,j] > max_overlap:
#                     max_overlap = overlap_matrix[i,j]
#                     best_pair = (i, j)
        
#         if best_pair is None:
#             break
            
#         print(f"\nОбъединение групп {best_pair} (перекрытие: {max_overlap:.1%})")
#         merged = merge_clouds([groups[best_pair[0]], groups[best_pair[1]]], 
#                             f"{best_pair[0]}-{best_pair[1]}", 
#                             use_global=True)
#         merged_pairs.append(merged)
#         remaining.remove(best_pair[0])
#         remaining.remove(best_pair[1])
    
#     # Добавляем оставшиеся группы
#     for idx in remaining:
#         merged_pairs.append(groups[idx])
    
#     # 4. Финальное объединение
#     print("\nФинальное объединение...")
#     if len(merged_pairs) == 1:
#         return merged_pairs[0]
    
#     return merge_clouds(merged_pairs, "final", use_global=True)

# # === Основной процесс ===

# @timeit
# def main():
#     print_header("Начало обработки")
    
#     # 1. Загрузка и предварительная обработка
#     print("\nЗагрузка и предварительная обработка облаков...")
#     angle_groups = [
#         [0, 30, 60],      # Группа 0
#         [90, 120, 150],   # Группа 1
#         [180, 210, 240],  # Группа 2
#         [270, 300, 330]   # Группа 3
#     ]
    
#     # Сохраняем глобальные параметры в локальные переменные
#     global VOXEL_SIZE, ICP_DISTANCE_COARSE, ICP_DISTANCE_FINE
#     original_voxel = VOXEL_SIZE
#     original_coarse = ICP_DISTANCE_COARSE
#     original_fine = ICP_DISTANCE_FINE
    
#     clouds = {}
#     for angle in ANGLES:
#         path = os.path.join("pointclouds", f"cloud_{angle}.ply")
#         try:
#             clouds[angle] = load_and_preprocess(path, angle)
#         except Exception as e:
#             print(f"Ошибка: {str(e)}")
#             return

#     # 2. Объединение внутри групп
#     print_header("Объединение внутри групп")
#     group_results = []
    
#     for i, group in enumerate(angle_groups):
#         print(f"\nОбработка группы {i}: {group}")
        
#         # Специальные параметры для проблемной группы
#         if i == SPECIAL_GROUP_INDEX:
#             print("Применение специальных параметров для проблемной группы")
#             # Сохраняем оригинальные параметры
#             orig_params = {
#                 'voxel': VOXEL_SIZE,
#                 'coarse': ICP_DISTANCE_COARSE,
#                 'fine': ICP_DISTANCE_FINE
#             }
#             # Устанавливаем специальные параметры
#             VOXEL_SIZE = SPECIAL_PARAMS['voxel_size']
#             ICP_DISTANCE_COARSE = SPECIAL_PARAMS['icp_coarse']
#             ICP_DISTANCE_FINE = SPECIAL_PARAMS['icp_fine']
        
#         group_clouds = [clouds[angle] for angle in group]
#         merged_group = merge_clouds(group_clouds, f"group_{i}")
#         group_results.append(merged_group)
        
#         # Восстановление параметров
#         if i == SPECIAL_GROUP_INDEX:
#             VOXEL_SIZE = orig_params['voxel']
#             ICP_DISTANCE_COARSE = orig_params['coarse']
#             ICP_DISTANCE_FINE = orig_params['fine']
        
#         # Сохранение промежуточных результатов
#         if SAVE_INTERMEDIATE:
#             o3d.io.write_point_cloud(f"group_{i}_merged.ply", merged_group)
    
#     # Восстановление глобальных параметров на случай, если что-то пошло не так
#     VOXEL_SIZE = original_voxel
#     ICP_DISTANCE_COARSE = original_coarse
#     ICP_DISTANCE_FINE = original_fine
    
#     # 3. Объединение групп
#     print_header("Объединение групп")
#     final_cloud = merge_groups_strategy(group_results)
    
#     # 4. Финальная обработка
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



# import open3d as o3d
# import numpy as np
# import os
# import copy
# from collections import deque
# import time

# # ===== НАСТРОЙКИ =====
# VOXEL_SIZE = 0.003  # Базовый размер вокселя (3 мм)
# ICP_DISTANCE_COARSE = 0.01  # Порог для грубой ICP
# ICP_DISTANCE_FINE = 0.005   # Порог для точной ICP
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

# # Настройки для проблемных облаков
# SPECIAL_GROUP_INDEX = 3  # Группа 270-300-330
# SPECIAL_PARAMS = {
#     'voxel_size': 0.002,
#     'icp_coarse': 0.007,
#     'icp_fine': 0.003,
#     'normal_radius': 0.004,
#     'max_nn_normal': 70
# }

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
    
#     # Фильтрация выбросов
#     cl, ind = cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
#     inliers = len(ind)/len(cloud.points)
#     print(f"Фильтрация выбросов: сохранено {inliers:.1%} точек")
#     cloud = cloud.select_by_index(ind)
    
#     # Применение начальных преобразований
#     cloud.transform(BASE_TRANSFORM)
#     if angle != 0:
#         rotate_cloud(cloud, angle)
    
#     # Расчет нормалей
#     radius_normal = VOXEL_SIZE * 1.5
#     cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
#         radius=radius_normal, max_nn=50))
    
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
#     """Улучшенная глобальная регистрация"""
#     print("\nГлобальная регистрация (RANSAC):")
#     voxel_sizes = [VOXEL_SIZE * 3, VOXEL_SIZE * 1.5, VOXEL_SIZE]
#     best_result = None
    
#     for i, voxel_size in enumerate(voxel_sizes):
#         print(f"Попытка {i+1} с вокселем {voxel_size*1000:.1f} мм")
#         source_down = source.voxel_down_sample(voxel_size)
#         target_down = target.voxel_down_sample(voxel_size)
        
#         # Оценка нормалей
#         radius_normal = voxel_size * 2
#         source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
#             radius=radius_normal, max_nn=50))
#         target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
#             radius=radius_normal, max_nn=50))
        
#         # Вычисление FPFH признаков
#         radius_feature = voxel_size * 5
#         source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#             source_down, o3d.geometry.KDTreeSearchParamHybrid(
#                 radius=radius_feature, max_nn=100))
#         target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#             target_down, o3d.geometry.KDTreeSearchParamHybrid(
#                 radius=radius_feature, max_nn=100))
        
#         # Глобальная регистрация
#         result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#             source_down, target_down, source_fpfh, target_fpfh, True,
#             voxel_size * 1.5,
#             o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#             4, [
#                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
#                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size),
#                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(np.deg2rad(10))
#             ],
#             o3d.pipelines.registration.RANSACConvergenceCriteria(500000, 0.999))
        
#         print(f"  Fitness: {result.fitness:.3f}, Inliers: {result.fitness*100:.1f}%")
#         if best_result is None or result.fitness > best_result.fitness:
#             best_result = result
    
#     print(f"Лучший результат: fitness={best_result.fitness:.3f}")
#     return best_result.transformation if best_result else np.identity(4)

# @timeit
# def multi_scale_icp(source, target, initial_transform, max_iterations=[100, 50]):
#     """Многоуровневая ICP-регистрация"""
#     thresholds = [ICP_DISTANCE_COARSE, ICP_DISTANCE_FINE]
#     current_transform = initial_transform
    
#     for level, (threshold, max_iter) in enumerate(zip(thresholds, max_iterations)):
#         print(f"\nICP Уровень {level+1} (порог: {threshold*1000:.1f} мм):")
        
#         # Обновление нормалей
#         radius_normal = threshold * 2
#         source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
#             radius=radius_normal, max_nn=50))
#         target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
#             radius=radius_normal, max_nn=50))
        
#         # Выбор метода ICP
#         if USE_COLORED_ICP and source.has_colors() and target.has_colors():
#             print("Используется Colored ICP")
#             estimation = o3d.pipelines.registration.TransformationEstimationForColoredICP()
#         else:
#             estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        
#         # Выполнение ICP
#         result = o3d.pipelines.registration.registration_icp(
#             source, target, threshold, current_transform,
#             estimation,
#             o3d.pipelines.registration.ICPConvergenceCriteria(
#                 relative_fitness=1e-6,
#                 relative_rmse=1e-6,
#                 max_iteration=max_iter))
        
#         current_transform = result.transformation
#         print(f"  Fitness: {result.fitness:.3f}, RMSE: {result.inlier_rmse:.5f} м")
#         print(f"  Итераций: {max_iter}")
        
#         if VISUALIZE_STEPS:
#             temp = copy.deepcopy(source)
#             temp.transform(current_transform)
#             o3d.visualization.draw_geometries([temp, target])
    
#     return current_transform

# def evaluate_alignment(source, target, threshold=0.005):
#     """Комплексная оценка качества выравнивания"""
#     print("\nОценка качества выравнивания:")
    
#     # Вычисление расстояний
#     dist_source = np.asarray(source.compute_point_cloud_distance(target))
#     dist_target = np.asarray(target.compute_point_cloud_distance(source))
    
#     # Статистика
#     inliers_source = np.sum(dist_source < threshold)
#     inliers_target = np.sum(dist_target < threshold)
#     overlap_source = inliers_source / len(dist_source)
#     overlap_target = inliers_target / len(dist_target)
#     mean_dist = (np.mean(dist_source) + np.mean(dist_target)) / 2
#     median_dist = (np.median(dist_source) + np.median(dist_target)) / 2
    
#     print(f"Перекрытие точек:")
#     print(f"  Source -> Target: {overlap_source:.1%} (порог: {threshold*1000:.1f} мм)")
#     print(f"  Target -> Source: {overlap_target:.1%}")
#     print(f"Среднее расстояние: {mean_dist*1000:.2f} мм")
#     print(f"Медианное расстояние: {median_dist*1000:.2f} мм")
    
#     # Визуализация плохо совмещенных областей
#     if VISUALIZE_STEPS:
#         bad_points_source = np.where(dist_source > threshold*2)[0]
#         bad_points_target = np.where(dist_target > threshold*2)[0]
        
#         source_colored = copy.deepcopy(source)
#         target_colored = copy.deepcopy(target)
        
#         colors = np.asarray(source_colored.colors if source_colored.has_colors() else 
#                            np.tile([0.5, 0.5, 0.5], (len(source_colored.points), 1)))
#         colors[bad_points_source] = [1, 0, 0]  # Красный для плохих точек
#         source_colored.colors = o3d.utility.Vector3dVector(colors)
        
#         colors = np.asarray(target_colored.colors if target_colored.has_colors() else 
#                            np.tile([0.5, 0.5, 0.5], (len(target_colored.points), 1)))
#         colors[bad_points_target] = [0, 0, 1]  # Синий для плохих точек
#         target_colored.colors = o3d.utility.Vector3dVector(colors)
        
#         o3d.visualization.draw_geometries([source_colored, target_colored])
    
#     return (overlap_source + overlap_target) / 2

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
# def merge_clouds(clouds_to_merge, group_name="", use_global=True):
#     """Иерархическое объединение облаков"""
#     print_header(f"Объединение группы {group_name}")
#     queue = deque(clouds_to_merge)
#     iteration = 1
    
#     while len(queue) > 1:
#         print(f"\nИтерация объединения {iteration}")
#         source = queue.popleft()
#         target = queue.popleft()
        
#         # Визуализация до регистрации
#         if VISUALIZE_STEPS:
#             temp_source = copy.deepcopy(source)
#             temp_target = copy.deepcopy(target)
#             temp_source.paint_uniform_color([1, 0, 0])
#             temp_target.paint_uniform_color([0, 0, 1])
#             o3d.visualization.draw_geometries([temp_source, temp_target])
        
#         # Регистрация
#         transformation = pairwise_registration(
#             source, target, 
#             use_global=(use_global and iteration == 1))
        
#         source.transform(transformation)
#         merged = source + target
        
#         # Сохранение промежуточных результатов
#         if SAVE_INTERMEDIATE:
#             o3d.io.write_point_cloud(
#                 f"intermediate_{group_name}_iter{iteration}.ply", 
#                 merged)
        
#         queue.append(merged)
#         iteration += 1
    
#     return queue[0]

# @timeit
# def merge_groups_strategy(groups):
#     """Стратегия сшивания групп с анализом перекрытий"""
#     print_header("Стратегия объединения групп")
    
#     # 1. Анализ перекрытий между группами
#     print("\nАнализ перекрытий между группами:")
#     overlap_matrix = np.zeros((len(groups), len(groups)))
#     for i in range(len(groups)):
#         for j in range(i+1, len(groups)):
#             overlap = evaluate_alignment(groups[i], groups[j], threshold=0.02)
#             overlap_matrix[i,j] = overlap
#             overlap_matrix[j,i] = overlap
    
#     print("\nМатрица перекрытий:")
#     print("      0     1     2     3")
#     for i, row in enumerate(overlap_matrix):
#         print(f"{i} | " + " ".join(f"{val:.2f}" if val > 0 else "  - " for val in row))
    
#     # 2. Определение порядка сшивания на основе перекрытий
#     best_pair = np.unravel_index(np.argmax(overlap_matrix), overlap_matrix.shape)
#     print(f"\nНаибольшее перекрытие между группами {best_pair}")
    
#     # 3. Объединение пар с наибольшим перекрытием
#     print("\nОбъединение групп с наибольшим перекрытием...")
#     merged_pairs = []
#     remaining = set(range(len(groups)))
    
#     while len(remaining) >= 2:
#         # Находим пару с максимальным перекрытием
#         max_overlap = -1
#         best_pair = None
#         for i in remaining:
#             for j in remaining:
#                 if i != j and overlap_matrix[i,j] > max_overlap:
#                     max_overlap = overlap_matrix[i,j]
#                     best_pair = (i, j)
        
#         if best_pair is None:
#             break
            
#         print(f"\nОбъединение групп {best_pair} (перекрытие: {max_overlap:.1%})")
#         merged = merge_clouds([groups[best_pair[0]], groups[best_pair[1]]], 
#                             f"{best_pair[0]}-{best_pair[1]}", 
#                             use_global=True)
#         merged_pairs.append(merged)
#         remaining.remove(best_pair[0])
#         remaining.remove(best_pair[1])
    
#     # Добавляем оставшиеся группы
#     for idx in remaining:
#         merged_pairs.append(groups[idx])
    
#     # 4. Финальное объединение
#     print("\nФинальное объединение...")
#     if len(merged_pairs) == 1:
#         return merged_pairs[0]
    
#     return merge_clouds(merged_pairs, "final", use_global=True)

# # === Основной процесс ===
# @timeit
# def main():
#     print_header("Начало обработки")
    
#     # 1. Загрузка и предварительная обработка
#     print("\nЗагрузка и предварительная обработка облаков...")
#     angle_groups = [
#         [0, 30, 60],      # Группа 0
#         [90, 120, 150],   # Группа 1
#         [180, 210, 240],  # Группа 2
#         [270, 300, 330]   # Группа 3
#     ]
    
#     clouds = {}
#     for angle in ANGLES:
#         path = os.path.join("pointclouds", f"cloud_{angle}.ply")
#         try:
#             clouds[angle] = load_and_preprocess(path, angle)
#         except Exception as e:
#             print(f"Ошибка: {str(e)}")
#             return

#     # 2. Объединение внутри групп
#     print_header("Объединение внутри групп")
#     group_results = []
    
#     for i, group in enumerate(angle_groups):
#         print(f"\nОбработка группы {i}: {group}")
        
#         # Специальные параметры для проблемной группы
#         if i == SPECIAL_GROUP_INDEX:
#             print("Применение специальных параметров для проблемной группы")
#             # Сохраняем оригинальные параметры
#             orig_params = {
#                 'voxel_size': VOXEL_SIZE,
#                 'icp_coarse': ICP_DISTANCE_COARSE,
#                 'icp_fine': ICP_DISTANCE_FINE
#             }
#             # Устанавливаем специальные параметры
#             VOXEL_SIZE = SPECIAL_PARAMS['voxel_size']
#             ICP_DISTANCE_COARSE = SPECIAL_PARAMS['icp_coarse']
#             ICP_DISTANCE_FINE = SPECIAL_PARAMS['icp_fine']
        
#         group_clouds = [clouds[angle] for angle in group]
#         merged_group = merge_clouds(group_clouds, f"group_{i}")
#         group_results.append(merged_group)
        
#         # Восстановление параметров
#         if i == SPECIAL_GROUP_INDEX:
#             VOXEL_SIZE = orig_params['voxel_size']
#             ICP_DISTANCE_COARSE = orig_params['icp_coarse']
#             ICP_DISTANCE_FINE = orig_params['icp_fine']
        
#         # Сохранение промежуточных результатов
#         if SAVE_INTERMEDIATE:
#             o3d.io.write_point_cloud(f"group_{i}_merged.ply", merged_group)
    
#     # 3. Объединение групп
#     print_header("Объединение групп")
#     final_cloud = merge_groups_strategy(group_results)
    
#     # 4. Финальная обработка
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


# # import open3d as o3d
# # import numpy as np
# # import os
# # import copy
# # from collections import deque

# # # ===== НАСТРОЙКИ =====
# # VOXEL_SIZE = 0.003  # Уменьшенный размер вокселя для большей точности
# # ICP_DISTANCE_COARSE = 0.01  # Порог для грубой ICP-регистрации
# # ICP_DISTANCE_FINE = 0.005   # Порог для точной ICP-регистрации
# # VISUALIZE_STEPS = True      # Визуализация промежуточных результатов
# # USE_COLORED_ICP = False     # Использовать цветную информацию, если доступна

# # # Углы и параметры облаков
# # ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
# # BASE_TRANSFORM = np.array([
# #     [1, 0, 0, -0.060],
# #     [0, 1, 0,    0.0],
# #     [0, 0, 1, -1.420],
# #     [0, 0, 0,    1.0]
# # ])

# # # === Функции обработки ===
# # def load_and_preprocess(path, angle):
# #     """Загрузка и подготовка облака точек с улучшенной обработкой"""
# #     cloud = o3d.io.read_point_cloud(path)
# #     if not cloud.has_points():
# #         raise ValueError(f"Cloud {path} is empty!")
    
# #     # Улучшенная фильтрация выбросов
# #     cl, ind = cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
# #     cloud = cloud.select_by_index(ind)
    
# #     # Применение начальных преобразований
# #     cloud.transform(BASE_TRANSFORM)
# #     if angle != 0:
# #         rotate_cloud(cloud, angle)
    
# #     # Более точный расчет нормалей
# #     radius_normal = VOXEL_SIZE * 1.5
# #     cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
# #         radius=radius_normal, max_nn=50))
    
# #     return cloud

# # def rotate_cloud(cloud, angle_deg):
# #     """Прецизионный поворот облака вокруг оси Y"""
# #     angle_rad = np.deg2rad(angle_deg)
# #     rotation = np.array([
# #         [ np.cos(angle_rad), 0, np.sin(angle_rad), 0],
# #         [ 0,                 1, 0,                 0],
# #         [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
# #         [ 0,                 0, 0,                 1]
# #     ])
# #     cloud.transform(rotation)

# # def multi_scale_icp(source, target, initial_transform=np.identity(4)):
# #     """Многоуровневая ICP-регистрация с разной точностью"""
# #     current_transform = initial_transform
# #     thresholds = [ICP_DISTANCE_COARSE, ICP_DISTANCE_FINE]  # Два уровня точности
    
# #     for threshold in thresholds:
# #         # Обновляем нормали для текущего уровня точности
# #         radius_normal = threshold * 2
# #         source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
# #             radius=radius_normal, max_nn=50))
# #         target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
# #             radius=radius_normal, max_nn=50))
        
# #         if USE_COLORED_ICP and source.has_colors() and target.has_colors():
# #             # Используем Colored ICP если есть цветная информация
# #             icp_result = o3d.pipelines.registration.registration_colored_icp(
# #                 source, target, threshold, current_transform,
# #                 o3d.pipelines.registration.ICPConvergenceCriteria(
# #                     relative_fitness=1e-6, 
# #                     relative_rmse=1e-6, 
# #                     max_iteration=100))
# #         else:
# #             # Стандартный Point-to-Plane ICP
# #             icp_result = o3d.pipelines.registration.registration_icp(
# #                 source, target, threshold, current_transform,
# #                 o3d.pipelines.registration.TransformationEstimationPointToPlane(),
# #                 o3d.pipelines.registration.ICPConvergenceCriteria(
# #                     relative_fitness=1e-6, 
# #                     relative_rmse=1e-6, 
# #                     max_iteration=100))
        
# #         current_transform = icp_result.transformation
# #         fitness = icp_result.fitness
        
# #         print(f"ICP уровень {threshold:.4f} м: fitness={fitness:.3f}")
    
# #     return current_transform

# # def pairwise_registration(source, target, use_global_registration=True):
# #     """Улучшенная попарная регистрация"""
# #     # Даунсемплинг с более агрессивными параметрами
# #     source_down = source.voxel_down_sample(VOXEL_SIZE)
# #     target_down = target.voxel_down_sample(VOXEL_SIZE)
    
# #     # Оценка нормалей для даунсемплинга
# #     radius_normal = VOXEL_SIZE * 1.5
# #     source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
# #         radius=radius_normal, max_nn=50))
# #     target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
# #         radius=radius_normal, max_nn=50))

# #     # Начальное преобразование
# #     if use_global_registration:
# #         print("Применение глобальной регистрации...")
# #         # Улучшенная глобальная регистрация
# #         radius_feature = VOXEL_SIZE * 3
# #         source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
# #             source_down, o3d.geometry.KDTreeSearchParamHybrid(
# #                 radius=radius_feature, max_nn=100))
# #         target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
# #             target_down, o3d.geometry.KDTreeSearchParamHybrid(
# #                 radius=radius_feature, max_nn=100))
        
# #         global_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
# #             source_down, target_down, source_fpfh, target_fpfh, True,
# #             VOXEL_SIZE * 1.5,
# #             o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
# #             4, [  # Более строгие проверки
# #                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
# #                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(VOXEL_SIZE),
# #                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(np.deg2rad(15))
# #             ], 
# #             o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999))
        
# #         initial_transformation = global_result.transformation
# #         print(f"Глобальная регистрация: fitness={global_result.fitness:.3f}")
# #     else:
# #         initial_transformation = np.identity(4)
    
# #     # Многоуровневая ICP-регистрация
# #     return multi_scale_icp(source, target, initial_transformation)

# # def merge_clouds(clouds_to_merge, group_name=""):
# #     """Улучшенное иерархическое объединение облаков"""
# #     queue = deque(clouds_to_merge)
# #     iteration = 1
    
# #     while len(queue) > 1:
# #         source = queue.popleft()
# #         target = queue.popleft()
        
# #         if VISUALIZE_STEPS:
# #             temp_source = copy.deepcopy(source)
# #             temp_target = copy.deepcopy(target)
# #             temp_source.paint_uniform_color([1, 0, 0])  # Красный
# #             temp_target.paint_uniform_color([0, 0, 1])  # Синий
# #             o3d.visualization.draw_geometries([temp_source, temp_target], 
# #                                             window_name=f"{group_name} До регистрации #{iteration}")
        
# #         # Регистрация с проверкой качества
# #         transformation = pairwise_registration(
# #             source, target, 
# #             use_global_registration=(iteration == 1))
        
# #         source.transform(transformation)
# #         merged = source + target
        
# #         # Оценка качества выравнивания
# #         alignment_score = evaluate_alignment(source, target)
# #         if alignment_score < 0.3:  # Если выравнивание плохое
# #             print("Предупреждение: низкое качество выравнивания!")
# #             # Попробуем альтернативный метод
# #             transformation = multi_scale_icp(source, target)
# #             source.transform(transformation)
# #             merged = source + target
# #             alignment_score = evaluate_alignment(source, target)
        
# #         if VISUALIZE_STEPS:
# #             o3d.visualization.draw_geometries([merged], 
# #                                             window_name=f"{group_name} После регистрации #{iteration}")
        
# #         queue.append(merged)
# #         iteration += 1
    
# #     return queue[0]

# # def evaluate_alignment(source, target, threshold=0.005):
# #     """Точная оценка качества выравнивания"""
# #     distances = source.compute_point_cloud_distance(target)
# #     inliers = np.sum(np.asarray(distances) < threshold)
# #     alignment_score = inliers / len(distances)
# #     print(f"Оценка выравнивания: {alignment_score:.2%} точек в пределах {threshold*1000:.1f} мм")
# #     return alignment_score

# # # === Основной процесс ===
# # def main():
# #     # 1. Загрузка и предварительная обработка всех облаков
# #     angle_groups = [
# #         [0, 30, 60],      # Передняя часть
# #         [90, 120, 150],    # Правая часть
# #         [180, 210, 240],   # Задняя часть
# #         [270, 300, 330]    # Левая часть
# #     ]
    
# #     clouds = {}
# #     for angle in ANGLES:
# #         path = os.path.join("pointclouds", f"cloud_{angle}.ply")
# #         try:
# #             clouds[angle] = load_and_preprocess(path, angle)
# #             print(f"Успешно загружено и обработано облако {angle}°")
# #         except Exception as e:
# #             print(f"Ошибка при обработке {path}: {str(e)}")
# #             return

# #     # 2. Иерархическое объединение по группам с улучшенными параметрами
# #     group_results = []
# #     for i, group in enumerate(angle_groups):
# #         group_clouds = [clouds[angle] for angle in group]
# #         merged_group = merge_clouds(group_clouds, group_name=f"Группа {i+1}")
# #         group_results.append(merged_group)
# #         print(f"\n=== Группа углов {group} успешно объединена ===\n")

# #     # 3. Объединение групп между собой с повышенной точностью
# #     print("\n=== Начинаем финальное сшивание групп ===\n")
    
# #     # Сначала объединяем соседние группы (0-1 и 2-3)
# #     merged_01 = merge_clouds([group_results[0], group_results[1]], "Группы 0-1")
# #     merged_23 = merge_clouds([group_results[2], group_results[3]], "Группы 2-3")
    
# #     # Затем объединяем результаты
# #     final_cloud = merge_clouds([merged_01, merged_23], "Финальное объединение")
    
# #     # 4. Финальная обработка и сохранение
# #     final_cloud.paint_uniform_color([0.6, 0.6, 0.6])  # Нейтральный цвет
    
# #     # Удаление выбросов
# #     cl, ind = final_cloud.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
# #     final_cloud = final_cloud.select_by_index(ind)
    
# #     # Сохранение
# #     output_path = "reconstructed_object_improved.ply"
# #     o3d.io.write_point_cloud(output_path, final_cloud)
# #     print(f"\nФинальное облако сохранено как '{output_path}'")
    
# #     # Визуализация с координатными осями
# #     axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
# #     o3d.visualization.draw_geometries([final_cloud, axis], 
# #                                     window_name="Улучшенная реконструкция",
# #                                     width=1024, height=768)

# # if __name__ == "__main__":
# #     main()

# # # import open3d as o3d
# # # import numpy as np
# # # import os
# # # import copy
# # # from collections import deque

# # # # ===== НАСТРОЙКИ =====
# # # VOXEL_SIZE = 0.005  # Тонкий даунсемплинг только для регистрации
# # # ICP_DISTANCE_THRESHOLD = 0.01  # Более жесткий порог для точного совмещения
# # # VISUALIZE_STEPS = True  # Показывать промежуточные результаты

# # # # Углы и параметры облаков
# # # ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
# # # BASE_TRANSFORM = np.array([
# # #     [1, 0, 0, -0.060],
# # #     [0, 1, 0,    0.0],
# # #     [0, 0, 1, -1.420],
# # #     [0, 0, 0,    1.0]
# # # ])

# # # # === Функции обработки ===
# # # def load_and_preprocess(path, angle):
# # #     """Загрузка и подготовка облака"""
# # #     cloud = o3d.io.read_point_cloud(path)
# # #     if not cloud.has_points():
# # #         raise ValueError(f"Cloud {path} is empty!")
    
# # #     # Фильтрация выбросов
# # #     cl, ind = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
# # #     cloud = cloud.select_by_index(ind)
    
# # #     # Применение начальных преобразований
# # #     cloud.transform(BASE_TRANSFORM)
# # #     if angle != 0:
# # #         rotate_cloud(cloud, angle)
    
# # #     # Расчет нормалей
# # #     radius_normal = VOXEL_SIZE * 2
# # #     cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
# # #         radius=radius_normal, max_nn=30))
    
# # #     return cloud

# # # def rotate_cloud(cloud, angle_deg):
# # #     """Поворот облака вокруг оси Y"""
# # #     angle_rad = np.deg2rad(angle_deg)
# # #     rotation = np.array([
# # #         [ np.cos(angle_rad), 0, np.sin(angle_rad), 0],
# # #         [ 0,                 1, 0,                 0],
# # #         [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
# # #         [ 0,                 0, 0,                 1]
# # #     ])
# # #     cloud.transform(rotation)

# # # def pairwise_registration(source, target):
# # #     """Попарная регистрация двух облаков"""
# # #     # Даунсемплинг только для регистрации
# # #     source_down = source.voxel_down_sample(VOXEL_SIZE)
# # #     target_down = target.voxel_down_sample(VOXEL_SIZE)
    
# # #     # Оценка нормалей для даунсемплинга
# # #     radius_normal = VOXEL_SIZE * 2
# # #     source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
# # #         radius=radius_normal, max_nn=30))
# # #     target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
# # #         radius=radius_normal, max_nn=30))
    
# # #     # Глобальная регистрация
# # #     radius_feature = VOXEL_SIZE * 5
# # #     source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
# # #         source_down, o3d.geometry.KDTreeSearchParamHybrid(
# # #             radius=radius_feature, max_nn=100))
# # #     target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
# # #         target_down, o3d.geometry.KDTreeSearchParamHybrid(
# # #             radius=radius_feature, max_nn=100))
    
# # #     global_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
# # #         source_down, target_down, source_fpfh, target_fpfh, True,
# # #         VOXEL_SIZE * 1.5,
# # #         o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
# # #         3, [
# # #             o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
# # #             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(VOXEL_SIZE * 1.5)
# # #         ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
# # #     # Расчет нормалей для полных облаков перед ICP
# # #     radius_normal_full = VOXEL_SIZE * 3
# # #     source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
# # #         radius=radius_normal_full, max_nn=50))
# # #     target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
# # #         radius=radius_normal_full, max_nn=50))
    
# # #     # Точечная ICP-регистрация
# # #     icp_result = o3d.pipelines.registration.registration_icp(
# # #         source, target, ICP_DISTANCE_THRESHOLD, global_result.transformation,
# # #         o3d.pipelines.registration.TransformationEstimationPointToPlane(),
# # #         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    
# # #     return icp_result.transformation

# # # def merge_clouds(clouds_to_merge):
# # #     """Иерархическое объединение облаков"""
# # #     queue = deque(clouds_to_merge)
    
# # #     while len(queue) > 1:
# # #         source = queue.popleft()
# # #         target = queue.popleft()
        
# # #         if VISUALIZE_STEPS:
# # #             # Визуализация перед регистрацией
# # #             temp_source = copy.deepcopy(source)
# # #             temp_target = copy.deepcopy(target)
# # #             temp_source.paint_uniform_color([1, 0, 0])  # Красный
# # #             temp_target.paint_uniform_color([0, 0, 1])  # Синий
# # #             o3d.visualization.draw_geometries([temp_source, temp_target], 
# # #                                             window_name="До регистрации")
        
# # #         # Регистрация
# # #         transformation = pairwise_registration(source, target)
# # #         source.transform(transformation)
        
# # #         # Объединение
# # #         merged = source + target
        
# # #         if VISUALIZE_STEPS:
# # #             #merged.paint_uniform_color([0, 1, 0])  # Зеленый
# # #             o3d.visualization.draw_geometries([merged], 
# # #                                             window_name="После регистрации")
        
# # #         queue.append(merged)
    
# # #     return queue[0]

# # # # === Основной процесс ===
# # # def main():
# # #     # 1. Загрузка и предварительная обработка всех облаков
# # #     angle_groups = [
# # #         [0, 30, 60],      # Передняя часть
# # #         [90, 120, 150],   # Правая часть
# # #         [180, 210, 240],  # Задняя часть
# # #         [270, 300, 330]   # Левая часть
# # #     ]
    
# # #     clouds = {}
# # #     for angle in ANGLES:
# # #         path = os.path.join("pointclouds", f"cloud_{angle}.ply")
# # #         try:
# # #             clouds[angle] = load_and_preprocess(path, angle)
# # #             print(f"Успешно загружено и обработано облако {angle}°")
# # #         except Exception as e:
# # #             print(f"Ошибка при обработке {path}: {str(e)}")
# # #             return

# # #     # 2. Иерархическое объединение по группам
# # #     group_results = []
# # #     for group in angle_groups:
# # #         group_clouds = [clouds[angle] for angle in group]
# # #         merged_group = merge_clouds(group_clouds)
# # #         group_results.append(merged_group)
# # #         print(f"Объединена группа углов: {group}")

# # #     # 3. Объединение групп между собой
# # #     final_cloud = merge_clouds(group_results)
# # #     print("Все группы успешно объединены")

# # #     # 4. Финальная обработка и сохранение
# # #     final_cloud.paint_uniform_color([0.5, 0.5, 0.5])  # Серый цвет для всего объекта
    
# # #     # Удаление возможных оставшихся выбросов
# # #     cl, ind = final_cloud.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
# # #     final_cloud = final_cloud.select_by_index(ind)
    
# # #     # Сохранение и визуализация
# # #     o3d.io.write_point_cloud("reconstructed_object.ply", final_cloud)
# # #     print("Финальное облако сохранено как 'reconstructed_object.ply'")
    
# # #     o3d.visualization.draw_geometries([final_cloud], 
# # #                                     window_name="Реконструированный объект",
# # #                                     width=1024, height=768)

# # # if __name__ == "__main__":
# # #     main()