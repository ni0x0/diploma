import threading
import socket
import json
import time
import sys
import copy
import traceback
from functools import wraps
import open3d as o3d

HOST = '127.0.0.1'
PORT = 65433
conn = None

shutdown_flag = False
window_flag = False
was_first_cloud_view = False
view_position = None

import numpy as np

clouds_info = []

viewer = None
viewer_lock = threading.Lock()

message_queue = []

def viewer_loop():
    global conn, viewer, shutdown_flag, window_flag, was_first_cloud_view, view_position, clouds_info

    try:
        while not shutdown_flag:
            with viewer_lock:
                while message_queue:
                    cmd = message_queue.pop(0)
                    handle_viewer_command(cmd)

            if window_flag:
                try:
                    viewer = o3d.visualization.Visualizer()
                    viewer.create_window(window_name="3D Viewer", width=800, height=600)

                    # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    # viewer.add_geometry(axes)

                    print("[3D Worker] 3D viewer window created")
                    send_msg(conn, {"status": "viewer_created"})

                    for i, cloud in enumerate(clouds_info):
                        if cloud["show_flag"]:
                            try:
                                viewer.add_geometry(cloud["cloud"])
                                was_first_cloud_view = True
                            except Exception as e:
                                print(f"[Open3D] Failed to add cloud at row {i}: {e}")
             
                    while window_flag:
                        with viewer_lock:
                            while message_queue:
                                cmd = message_queue.pop(0)
                                handle_viewer_command(cmd)

                            try:
                                viewer.poll_events()
                                viewer.update_renderer()
                            except Exception as e:
                                print(f"[3D Worker] Viewer loop update error: {e}")
                        time.sleep(0.03)

                except Exception as e:
                    print(f"[3D Worker] Viewer loop error: {e}")
                finally:
                    window_flag = False
                    viewer.destroy_window()
                    viewer = None
                    was_first_cloud_view = False
                    print("[3D Worker] 3D viewer window closed")
                    send_msg(conn, {"status": "viewer_closed"})

            # print("[3D Worker] 3D viewer window closed")
            time.sleep(0.03)

    except Exception as e:
        print(f"[3D Worker] Main loop error: {e}")

def handle_viewer_command(msg):
    global conn, window_flag

    cmd = msg.get("cmd")
    conn = msg.get("conn")

    if cmd == "ping":
        send_msg(conn, {"reply": "pong"})

    elif cmd == "shutdown":
        global shutdown_flag
        shutdown_flag = True
        window_flag = False
        send_msg(conn, {"reply": "goodbye"})

    elif cmd == "create_window":
        if not window_flag:
            window_flag = True
        else:
            send_msg(conn, {"status": "already_running"})

    elif cmd == "close_window":
        if window_flag:
            window_flag = False
            
    elif cmd == "upload_clouds":
        load_clouds(msg)

    elif cmd == "set_visible":
        set_cloud_visible(msg)

    elif cmd == "delete_row":
        delete_cloud(msg)

    elif cmd == "move_cloud":
        move_cloud(msg)

    elif cmd == "process_clouds":
        process_clouds(msg)

    else:
        send_msg(conn, {"error": f"Unknown command: {cmd}"})

# clouds_info.append({
#     "cloud": cloud,
#     "file": file_path,
#     "angle": angle,
#     "position": {
#         "shift_x": shift_x,  
#         "shift_y": shift_y,
#         "shift_z": shift_z,
#         "rot_xy": rot_xy,
#         "rot_xz": rot_xz,
#         "rot_yz": rot_yz,
#     },
#     "show": show_flag,
#     "merge": merge_flag,
# })

def load_clouds(msg):
    global clouds_info

    new_clouds = msg.get("clouds_new_data", [])
    print(f"[3D Worker] Received {len(new_clouds)} clouds for upload")

    do_shift = msg.get("do_preload_shift", False)
    do_filter = msg.get("do_filtration", False)
    filter_neighbors = int(msg.get("neighbors", 20))
    filter_sensitivity = float(msg.get("sensivity", 1))
    do_distance_filter = msg.get("do_distance_filter", False)
    distance_min = msg.get("min_distance", 0)
    distance_max = msg.get("max_distance", 8)

    preload_shift_x = msg.get("shift_x", 0)
    preload_shift_y = msg.get("shift_y", 0)
    preload_shift_z = msg.get("shift_z", 0)
    preload_rot_xy = msg.get("rot_xy", 0)
    preload_rot_xz = msg.get("rot_xz", 0)
    preload_rot_yz = msg.get("rot_yz", 0)

    for cloud_data in new_clouds:
        try:
            path = cloud_data["file"]
            angle = cloud_data["angle"]

            cloud_distance = load_and_filter_cloud(path, do_distance_filter, distance_min, distance_max)
            cloud_filtered = filter_outliers(cloud_distance, do_filter, filter_neighbors, filter_sensitivity)
            cloud_turned = apply_preload_transforms(cloud_filtered, do_shift, preload_shift_x, preload_shift_y, preload_shift_z, preload_rot_xy, preload_rot_xz, preload_rot_yz, angle)

            cloud_data["show_flag"] = False
            cloud_data["merge_flag"] = False
            cloud_data["cloud"] = cloud_turned

            clouds_info.append(cloud_data)
            
        except Exception as e:
            print(f"[3D Worker] Failed to load cloud {cloud_data.get('file')}: {e}")

    if len(new_clouds):
        send_msg(conn, {"status": "clouds loaded", "value" : len(new_clouds)})

    print(f"[3D Worker] Num of clouds became {len(clouds_info)}")

def load_and_filter_cloud(path, do_filter=False, distance_min=0.6, distance_max=10):
    cloud = o3d.io.read_point_cloud(path)
    if do_filter:
        distances = np.asarray(cloud.points)[:, 2]
        indices = np.where((distances >= distance_min) & (distances <= distance_max))[0]
        return cloud.select_by_index(indices)
    return cloud

def filter_outliers(cloud, do_filter=False, nb_neighbors=20, std_ratio=2.0):
    if do_filter:
        filtered_pcd, _ = cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return filtered_pcd
    return cloud

def apply_preload_transforms(cloud, do_shift=False, shift_x=0, shift_y=0, shift_z=0,
                             rot_xy=0, rot_xz=0, rot_yz=0,
                             angle_y=0):
    
    if do_shift == False:
        return cloud 

    """
    Применяет к облаку последовательно:
    1. Поворот вокруг Z на rot_xy (градусы)  — плоскость XY → ось Z
    2. Поворот вокруг Y на rot_xz (градусы)  — плоскость XZ → ось Y
    3. Поворот вокруг X на rot_yz (градусы)  — плоскость YZ → ось X
    4. Сдвиг по X, Y, Z
    5. Поворот вокруг Y на angle_y (градусы)
    """
    def rotation_matrix(axis, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        R = np.eye(4)
        if axis == 'x':
            R[1:3, 1:3] = [[np.cos(angle_rad), -np.sin(angle_rad)],
                           [np.sin(angle_rad),  np.cos(angle_rad)]]
        elif axis == 'y':
            R[[0,0,2,2],[0,2,0,2]] = [np.cos(angle_rad), np.sin(angle_rad),
                                      -np.sin(angle_rad), np.cos(angle_rad)]
        elif axis == 'z':
            R[0:2,0:2] = [[np.cos(angle_rad), -np.sin(angle_rad)],
                          [np.sin(angle_rad),  np.cos(angle_rad)]]
        return R

    # Повороты
    cloud.transform(rotation_matrix('x', rot_yz))  # rot_yz → вокруг X
    cloud.transform(rotation_matrix('y', rot_xz))  # rot_xz → вокруг Y
    cloud.transform(rotation_matrix('z', rot_xy))  # rot_xy → вокруг Z

    # Сдвиг
    translation = np.eye(4)
    translation[0, 3] = shift_x
    translation[1, 3] = shift_y
    translation[2, 3] = shift_z
    cloud.transform(translation)

    # Дополнительный поворот вокруг Y (например, 180°)
    cloud.transform(rotation_matrix('y', angle_y))

    return cloud


def preload_process_cloud(cloud_info, do_shift, do_filter, filter_neighbors, filter_sensitivity, do_distance_filter, distance_min, distance_max):
    try:
        if do_filter:
            cloud = cloud_info["cloud"]

            cloud, ind = cloud.remove_statistical_outlier(nb_neighbors=filter_neighbors, std_ratio=filter_sensitivity)
            cloud = cloud.select_by_index(ind)

            if do_distance_filter:
                distances = np.asarray(cloud.points)[:, 2]
                indices = np.where((distances >= distance_min) & (distances <= distance_max))[0]
                cloud = cloud.select_by_index(indices)    

            cloud_info["cloud"] = cloud

        if do_shift:
            if cloud_info["position"]["shift_x"] != 0:
                translate(cloud_info["cloud"], 0, cloud_info["position"]["shift_x"])
            if cloud_info["position"]["shift_y"] != 0:
                translate(cloud_info["cloud"], 1, cloud_info["position"]["shift_y"])
            if cloud_info["position"]["shift_z"] != 0:
                translate(cloud_info["cloud"], 2, cloud_info["position"]["shift_z"])
            
        if cloud_info["angle"] != 0:
            rotate(cloud_info["cloud"], 'y', cloud_info["angle"])

        if do_shift:
            if cloud_info["position"]["rot_xy"] != 0:
                rotate(cloud_info["cloud"], 'z', cloud_info["position"]["rot_xy"])
            if cloud_info["position"]["rot_xz"] != 0:
                rotate(cloud_info["cloud"], 'y', cloud_info["position"]["rot_xz"]) 
            if cloud_info["position"]["rot_yz"] != 0:
                rotate(cloud_info["cloud"], 'x', cloud_info["position"]["rot_yz"])

    except Exception as e:
        print(f"[3D Worker] Error load preprocess: {e}")

def translate(cloud, axis, delta):
    T = np.eye(4)
    T[axis, 3] = delta
    cloud.transform(T)

def rotate(cloud, axis, angle_deg):
    angle = np.deg2rad(angle_deg)
    R = np.eye(4)
    if axis == 'x':
        R[1:3, 1:3] = [[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]]
    elif axis == 'y':
        R[::2, ::2] = [[np.cos(angle), np.sin(angle)],
                    [-np.sin(angle), np.cos(angle)]]
    elif axis == 'z':
        R[0:2, 0:2] = [[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]]
    cloud.transform(R)

def set_cloud_visible(msg):
    global clouds_info, was_first_cloud_view, viewer
    ind = msg.get("index", 0)
    set_visible = msg.get("value", False)

    clouds_info[ind]["show_flag"] = set_visible
    cloud = clouds_info[ind]["cloud"]

    if not window_flag:
        return

    if (cloud and (cloud.has_points() == False)):
        print("no points in cloud")
        return
    
    try:
        if viewer:
            if set_visible:
                if was_first_cloud_view == False:
                    viewer.add_geometry(cloud)
                    was_first_cloud_view = True
                else:
                    viewer.add_geometry(cloud, reset_bounding_box=False)
            else:
                if was_first_cloud_view == False:
                    viewer.remove_geometry(cloud)
                    was_first_cloud_view = True
                else:
                    viewer.remove_geometry(cloud, reset_bounding_box=False)
    except Exception as e:
        print(f"Error toggling cloud visibility: {e}")

def delete_cloud(msg):
    try:
        global clouds_info, was_first_cloud_view, viewer

        ind = msg.get("index", 0)

        if viewer:
            if clouds_info[ind]["show_flag"]:
                if was_first_cloud_view == False:
                    viewer.remove_geometry(clouds_info[ind]["cloud"])
                    was_first_cloud_view = True
                else:
                    viewer.remove_geometry(clouds_info[ind]["cloud"], reset_bounding_box=False)
        
        del clouds_info[ind]
        print(f"[Open3D] Delete row: {ind}")
    except Exception as e:
        print(f"[Open3D] Can't remove geometry: {e}")

def move_cloud(msg):
    try:
        global clouds_info, viewer

        ind = msg.get("index", 0)
        delta = msg.get("delta", [])

        cloud = clouds_info[ind]["cloud"]

        if delta["shift_x"]: translate(cloud, 0, delta["shift_x"])
        if delta["shift_y"]: translate(cloud, 1, delta["shift_y"])
        if delta["shift_z"]: translate(cloud, 2, delta["shift_z"])

        if delta["rot_xy"]: rotate(cloud, 'z', delta["rot_xy"])
        if delta["rot_yz"]: rotate(cloud, 'y', delta["rot_yz"])
        if delta["rot_xz"]: rotate(cloud, 'x', delta["rot_xz"])

        if viewer:
            if clouds_info[ind]["show_flag"]:
                viewer.update_geometry(cloud)
        
        # print(f"[Open3D] Updated row: {ind}")
    except Exception as e:
        print(f"[Open3D] Can't update cloud position: {e}")    

def send_reg_info(msg: str):
    send_msg(conn, { "reply": "reg_info", "registration_info": msg})


def process_clouds(msg): 
    try: 
        global clouds_info, viewer

        indexes = msg.get("indexes", [])
        file = msg.get("file", "")

        voxel_size = msg.get("voxel_size", 0.0)
        calc_normals = msg.get("calc_normals", False)
        normals_radius = msg.get("normals_radius", 1.0)
        do_global = msg.get("do_global", False)

        feature_multipliers = msg.get("feature_multipliers", [])
        # align_method = msg.get("align_method", "RANSAC")
        ransac_iter = msg.get("ransac_iter", 4000)
        ransac_conf = msg.get("ransac_conf", 0.999)
        ransac_samples = msg.get("ransac_samples", 4)

        # fgr_iter = msg.get("fgr_iter", 64)
        # fgr_div = msg.get("fgr_div", 1.4)
        # check_edges = msg.get("check_edges", False)
        # check_dist = msg.get("check_dist", False)
        # check_normals = msg.get("check_normals", False)

        # icp_method = msg.get("icp_method", "PointToPlane")
        use_coarse_icp = msg.get("use_coarse_icp", False)
        coarse_icp_iter = msg.get("coarse_icp_iter", 50)
        # coarse_abs = msg.get("coarse_abs", False)
        # coarse_abs_thresh = msg.get("coarse_abs_thresh", 0.05)
        # coarse_mult_thresh = msg.get("coarse_mult_thresh", 1.0)
        coarse_rel_fit = msg.get("coarse_rel_fit", 0.999)
        coarse_rel_rmse = msg.get("coarse_rel_rmse", 0.01)

        use_fine_icp = msg.get("use_fine_icp", False)
        fine_icp_iter = msg.get("fine_icp_iter", 50)
        # fine_abs = msg.get("fine_abs", False)
        # fine_abs_thresh = msg.get("fine_abs_thresh", 0.01)
        # fine_mult_thresh = msg.get("fine_mult_thresh", 1.0)
        fine_rel_fit = msg.get("fine_rel_fit", 0.999)
        fine_rel_rmse = msg.get("fine_rel_rmse", 0.01)

        use_color_icp = msg.get("use_color_icp", False)
        shape_importance = msg.get("shape_importance", 0.5)
        color_importance = msg.get("color_importance", 0.5)

        # post_voxel_size = msg.get("post_voxel_size", 0.005)
        # fill_holes = msg.get("fill_holes", False)
        # hole_size = msg.get("hole_size", 0.01)
        # filtering = msg.get("filtering", False)
        # min_artifact_size = msg.get("min_artifact_size", 0.001)
        # artifact_sensitivity = msg.get("artifact_sensitivity", 0.5)
        # keep_largest = msg.get("keep_largest", True)

        # smoothing_method = msg.get("smoothing_method", "None")
        # smoothing_radius = msg.get("smoothing_radius", 0.01)
        # smoothing_iter = msg.get("smoothing_iter", 10)
        # edge_preserve = msg.get("edge_preserve", False)
        # taubin_lambda = msg.get("taubin_lambda", 0.5)
        # taubin_mu = msg.get("taubin_mu", -0.53)

        # optimize = msg.get("optimize", False)
        # opt_algo = msg.get("opt_algo", "None")
        # loop_closure = msg.get("loop_closure", False)
        # min_overlap = msg.get("min_overlap", 30)
        # edge_prune = msg.get("edge_prune", 0.25)
        # geodesic = msg.get("geodesic", False)

        # export_method = msg.get("export_method", "Poisson")
        # export_resolution = msg.get("export_resolution", "High")
        # export_smoothing = msg.get("export_smoothing", "None")


        config = {
            'normal_radius': 0.02,  # Радиус для расчета нормалей
            'do_global': False,  # Выполнять глобальную регистрацию
            'global_voxel_sizes': [0.1, 0.05, 0.025],  # Размеры вокселей для глобальной регистрации
            'global_confidence': 0.999,  # Уверенность для RANSAC
            'global_correspondence_randomness': 5,  # Параметр случайности соответствий
            'do_coarse_icp': True,  # Выполнять грубую ICP
            'coarse_distance_threshold': 0.015,  # Порог расстояния для грубой ICP
            'coarse_relative_fitness': 1e-6,  # Критерии сходимости
            'coarse_relative_rmse': 1e-6,
            'coarse_max_iteration': 150,
            'do_fine_icp': True,  # Выполнять точную ICP
            'fine_distance_threshold': 0.005,  # Порог расстояния для точной ICP
            'fine_relative_fitness': 1e-7,  # Критерии сходимости
            'fine_relative_rmse': 1e-7,
            'fine_max_iteration': 250,
            'use_color_icp': False,  # Использовать цветную ICP
            'postprocess_voxel_size': 0.002,  # Размер вокселя для постобработки
            'save_intermediate': False,  # Сохранять промежуточные результаты
        }

        # config = {
        #     'normal_radius': normals_radius,  # Радиус для расчета нормалей
        #     'do_global': do_global,  # Выполнять глобальную регистрацию
        #     'global_voxel_sizes': feature_multipliers,  # Размеры вокселей для глобальной регистрации
        #     'global_confidence': ransac_conf,  # Уверенность для RANSAC
        #     'global_correspondence_randomness': ransac_samples,  # Параметр случайности соответствий

        #     'do_coarse_icp': use_coarse_icp,  # Выполнять грубую ICP
        #     'coarse_distance_threshold':0.05,  # Порог расстояния для грубой ICP
        #     'coarse_relative_fitness': coarse_rel_fit,  # Критерии сходимости
        #     'coarse_relative_rmse': coarse_rel_rmse,
        #     'coarse_max_iteration': coarse_icp_iter,

        #     'do_fine_icp': use_fine_icp,  # Выполнять точную ICP
        #     'fine_distance_threshold': 0.02,  # Порог расстояния для точной ICP
        #     'fine_relative_fitness': fine_rel_fit,  # Критерии сходимости
        #     'fine_relative_rmse': fine_rel_rmse,
        #     'fine_max_iteration': fine_icp_iter,

        #     'use_color_icp': use_color_icp,  # Использовать цветную ICP
        #     'shape_importance': shape_importance,
        #     'color_importance': color_importance,

        #     'postprocess_voxel_size': 0.02,  # Размер вокселя для постобработки
        #     'save_intermediate': False,  # Сохранять промежуточные результаты
        # }

        global VISUALIZE_STEPS
        VISUALIZE_STEPS = False

        clouds = []
        angles = []
        for ind in indexes:
            clouds.append(clouds_info[ind]["cloud"])
            angles.append(clouds_info[ind]["angle"])

        # Registration
        final_cloud, transformations = incremental_registration(clouds, angles, config)
        
        # Post-processing
        final_cloud = postprocess_merged_cloud(
            final_cloud, 
            voxel_size=config['postprocess_voxel_size'])
        
        # Final result
        print_header("Final result")
        send_reg_info("Final result")
        
        # # Color final cloud
        # final_cloud.paint_uniform_color([0.7, 0.7, 0.7])
        
        # Create coordinate axes
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        # Visualization with axes
        o3d.visualization.draw_geometries([final_cloud, axis], 
                                        window_name="Final reconstruction",
                                        width=1200, height=800)
        
        # Saving
        output_path = "0_final_reconstruction_final_after.ply"
        o3d.io.write_point_cloud(output_path, final_cloud)
        print(f"\nFinal cloud saved as '{output_path}'")
        send_reg_info(f"\nFinal cloud saved as '{output_path}'")
        
        # Completion message
        print("\nProcessing completed successfully!")
        send_reg_info("\nProcessing completed successfully!")
        print("Final cloud saved and displayed.")
        send_reg_info("Final cloud saved and displayed.")

        cloud_data = {}
        cloud_data["cloud"] = final_cloud
        cloud_data["show_flag"] = False
        cloud_data["merge_flag"] = False

        clouds_info.append(cloud_data)

    except Exception as e:
        print(f"[Open3D] Can't update cloud position: {e}")   
        send_reg_info(f"[Open3D] Can't update cloud position: {e}")

# Decorators and helper functions
def timeit(func):
    """Decorator for measuring function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"\nStarting execution: {func.__name__}")
        send_reg_info(f"\nStarting execution: {func.__name__}")
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Completed: {func.__name__}. Execution time: {elapsed_time:.2f} seconds")
        send_reg_info(f"Completed: {func.__name__}. Execution time: {elapsed_time:.2f} seconds")
        return result
    return wrapper

def print_header(message):
    """Print stage header"""
    print(f"\n{'='*10}")
    print(f"{message.upper()}")
    print(f"{'='*10}")
    send_reg_info(f"\n{'='*10}")
    send_reg_info(f"{message.upper()}")
    send_reg_info(f"{'='*10}")

def visualize_registration_step(source, target, transformation, title):
    """Visualize registration step"""
    if not VISUALIZE_STEPS:
        return
    
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    if transformation is not None:
        source_temp.transform(transformation)
    
    source_temp.paint_uniform_color([1, 0.706, 0])  # Orange
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Blue
    
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=title)

def visualize_merged(clouds, title):
    """Visualize merged clouds"""
    if not VISUALIZE_STEPS:
        return
    
    clouds_to_show = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    
    for i, cloud in enumerate(clouds):
        cloud_temp = copy.deepcopy(cloud)
        if len(clouds) > 1 and i < len(colors):
            cloud_temp.paint_uniform_color(colors[i])
        clouds_to_show.append(cloud_temp)
    
    o3d.visualization.draw_geometries(clouds_to_show, window_name=title)

# Main processing functions
@timeit
def preprocess_cloud(cloud, normal_radius):
    """Preprocess point cloud"""
    print(f"Preprocessing cloud (normals with radius {normal_radius:.3f})")
    send_reg_info(f"Preprocessing cloud (normals with radius {normal_radius:.3f})")
    cloud = calculate_normals(cloud, normal_radius)
    return cloud

def calculate_normals(cloud, radius_normal, max_nn=80):
    """Calculate normals for point cloud"""
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn))
    return cloud

@timeit
def pairwise_global_registration(source, target, config):
    """Pairwise global registration"""
    print_header("Global registration (RANSAC)")
    
    best_result = None
    voxel_sizes = config['global_voxel_sizes']
    
    for i, voxel_size in enumerate(voxel_sizes):
        print(f"Attempt {i+1} with voxel {voxel_size:.3f}")
        send_reg_info(f"Attempt {i+1} with voxel {voxel_size:.3f}")
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        
        # Calculate normals
        radius_normal = voxel_size * 3
        source_down = calculate_normals(source_down, radius_normal)
        target_down = calculate_normals(target_down, radius_normal)
        
        # Compute FPFH features
        radius_feature = voxel_size * 6
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down, o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_feature, max_nn=200))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down, o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_feature, max_nn=200))
        
        # Global registration
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            voxel_size * 3,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            config['global_correspondence_randomness'], [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 3),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(np.deg2rad(20))
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(2000000, config['global_confidence']))
        
        print(f"  Fitness: {result.fitness:.3f}, Inliers: {result.fitness*100:.1f}%")
        send_reg_info(f"  Fitness: {result.fitness:.3f}, Inliers: {result.fitness*100:.1f}%")
        if best_result is None or result.fitness > best_result.fitness:
            best_result = result
    
    if best_result is None or best_result.fitness < 0.3:
        print("Global registration failed, using rough initialization")
        send_reg_info("Global registration failed, using rough initialization")
        source_center = source.get_center()
        target_center = target.get_center()
        translation = target_center - source_center
        rough_transform = np.identity(4)
        rough_transform[:3, 3] = translation
        return rough_transform
    
    print(f"Best result: fitness={best_result.fitness:.3f}")
    send_reg_info(f"Best result: fitness={best_result.fitness:.3f}")
    return best_result.transformation

@timeit
def pairwise_icp_registration(source, target, initial_transform, config, icp_type='fine'):
    """Pairwise ICP registration"""
    if icp_type == 'coarse':
        print_header("Coarse ICP registration")
        threshold = config['coarse_distance_threshold']
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=config['coarse_relative_fitness'],
            relative_rmse=config['coarse_relative_rmse'],
            max_iteration=config['coarse_max_iteration'])
    else:
        print_header("Fine ICP registration")
        threshold = config['fine_distance_threshold']
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=config['fine_relative_fitness'],
            relative_rmse=config['fine_relative_rmse'],
            max_iteration=config['fine_max_iteration'])
    
    if config['use_color_icp'] and source.has_colors() and target.has_colors():
        print("Using Colored ICP")
        send_reg_info("Using Colored ICP")
        estimation = o3d.pipelines.registration.TransformationEstimationForColoredICP()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, initial_transform,
        estimation, criteria)
    
    print(f"  Fitness: {result.fitness:.3f}, RMSE: {result.inlier_rmse:.5f} m")
    send_reg_info(f"  Fitness: {result.fitness:.3f}, RMSE: {result.inlier_rmse:.5f} m")
    return result.transformation

@timeit
def pairwise_registration(source, target, config):
    """Full pairwise registration of two clouds"""
    print_header(f"Cloud registration")
    
    # Visualization before registration
    visualize_registration_step(source, target, None, "Before registration")
    
    # Global registration
    if config['do_global']:
        transformation = pairwise_global_registration(source, target, config)
        visualize_registration_step(source, target, transformation, "After global registration")
    else:
        transformation = np.identity(4)
    
    # Coarse ICP registration
    if config['do_coarse_icp']:
        transformation = pairwise_icp_registration(source, target, transformation, config, 'coarse')
        visualize_registration_step(source, target, transformation, "After coarse ICP")
    
    # Fine ICP registration
    if config['do_fine_icp']:
        transformation = pairwise_icp_registration(source, target, transformation, config, 'fine')
        visualize_registration_step(source, target, transformation, "After fine ICP")
    
    return transformation

@timeit
def incremental_registration(clouds, angles, config):
    """Sequential registration with visualization"""
    print_header("Sequential registration")
    
    # Preprocess all clouds
    processed_clouds = [preprocess_cloud(cloud, config['normal_radius']) for cloud in clouds]
    
    merged_cloud = processed_clouds[0]
    transformations = [np.identity(4)]
    
    for i in range(1, len(processed_clouds)):
        print(f"\nRegistering cloud {i} ({angles[i]}°)")
        send_reg_info(f"\nRegistering cloud {i} ({angles[i]}°)")
        
        # Visualization before merging
        visualize_merged([merged_cloud, processed_clouds[i]], f"Before registration {i}")
        
        # Registration
        transform = pairwise_registration(processed_clouds[i], merged_cloud, config)
        transformations.append(transform)
        processed_clouds[i].transform(transform)
        
        # Merging
        merged_cloud += processed_clouds[i]
        
        # Visualization after merging
        visualize_merged([merged_cloud], f"After merging {i}")
        
        if config.get('save_intermediate', False):
            o3d.io.write_point_cloud(f"intermediate_{i}.ply", merged_cloud)
    
    return merged_cloud, transformations

@timeit
def postprocess_merged_cloud(cloud, voxel_size, nb_neighbors=200, std_ratio=5):
    """Post-process merged cloud"""
    print_header("Post-processing")
    
    # Visualization before processing
    visualize_merged([cloud], "Before post-processing")
    
    # 1. Remove duplicates
    cloud = cloud.voxel_down_sample(voxel_size)
    visualize_merged([cloud], "After removing duplicates")
    
    # 2. Remove outliers
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    visualize_merged([cloud], "After removing outliers")

    # 3. Improve normals
    cloud = calculate_normals(cloud, voxel_size*3)
    
    return cloud

# -----------------------------------------------------------------------------------------------

def handle_client(conn):
    print("[3D Worker] Client connected")
    send_msg(conn, {"reply": "hello"})
    buffer = ""

    with conn:
        while not shutdown_flag:
            try:
                data = conn.recv(1024)
                if not data:
                    break
                buffer += data.decode()
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    msg = json.loads(line)
                    print("[3D Worker] Got:", msg)
                    msg["conn"] = conn
                    with viewer_lock:
                        message_queue.append(msg)
            except Exception as e:
                print(f"[3D Worker] Error: {e}")
                traceback.print_exc()
                break

    print("[3D Worker] Client disconnected")


def send_msg(conn, obj):
    try:
        msg = json.dumps(obj) + "\n"
        conn.sendall(msg.encode())
    except Exception as e:
        print(f"[3D Worker] Failed to send message: {e}")

def run_server():
    global conn
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[3D Worker] Listening on {HOST}:{PORT}")
        conn, _ = s.accept()
        handle_client(conn)

if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    viewer_loop()

    print("[3D Worker] Exiting")
    sys.exit(0)
