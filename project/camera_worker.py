# camera_worker.py

import socket
import threading
import time
import os
import json
import sys
import cv2
import numpy as np
from primesense import openni2

from dataclasses import dataclass, asdict

@dataclass
class DeviceConfig:
    registration: bool = True
    sync: bool = True
    mirror: bool = False

    def to_dict(self):
        return asdict(self)
    

HOST = '127.0.0.1'
PORT = 65432

shutdown_flag = False

camera_stop_event = threading.Event()
camera_turn_off_event = threading.Event()

camera_running = False
camera_thread = None

dev = None
depth_stream = None
color_stream = None

last_depth = None
last_color = None

camera_cycle_is_on = False

device_config = DeviceConfig()

COLORMAPS = {
    "jet": cv2.COLORMAP_JET,
    "bone": cv2.COLORMAP_BONE,
    "plasma": cv2.COLORMAP_PLASMA,
}

current_colormap = "jet"
camera_crop_width = 640
camera_crop_height = 480

background_path = ""
background_frame = None
sh_contour = False
sh_cntr_shift_x = 0
sh_cntr_shift_y = 0
sh_cntr_stretch_x = 0
sh_cntr_stretch_y = 0
kernel_value = 5
depth_trunk = 10 # m

not_video_mode_event = threading.Event()
not_video_mode_event.clear()
video_mode = True
not_video_depth_frame = None
not_video_color_frame = None
not_video_crop_width = 640
not_video_crop_height = 480

def send_msg(conn, obj):
    try:
        msg = json.dumps(obj) + "\n"
        conn.sendall(msg.encode())
    except Exception as e:
        print(f"[Camera Worker] Failed to send message: {e}")

def apply_camera_config(config: DeviceConfig, conn=None):
    global dev, depth_stream, color_stream
    
    try:
        if dev is None or depth_stream is None or color_stream is None:
            raise RuntimeError("Camera not initialized")

        if config.registration:
            dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        else:
            dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_OFF)

        dev.set_depth_color_sync_enabled(config.sync)

        depth_stream.set_mirroring_enabled(config.mirror)
        color_stream.set_mirroring_enabled(config.mirror)

        print("[Camera Worker] Updated camera config:", config.to_dict())
        # if conn:
        #     send_msg(conn, {"status": "config_applied", "config": config.to_dict()})

    except Exception as e:
        err = f"[Camera Worker] Failed to apply config: {e}"
        print(err)
        if conn:
            send_msg(conn, {"error": err})

def init_camera(config: DeviceConfig, conn=None):
    global dev, depth_stream, color_stream

    try:
        dev = openni2.Device.open_any()

        depth_stream = dev.create_depth_stream()
        color_stream = dev.create_color_stream()

        depth_stream.set_video_mode(openni2.VideoMode(
            pixelFormat=openni2.PIXEL_FORMAT_DEPTH_1_MM,
            resolutionX=640, resolutionY=480, fps=30))
        color_stream.set_video_mode(openni2.VideoMode(
            pixelFormat=openni2.PIXEL_FORMAT_RGB888,
            resolutionX=640, resolutionY=480, fps=30))

        apply_camera_config(config, conn)

        depth_stream.start()
        color_stream.start()

    except Exception as e:
        error_msg = f"[Camera Worker] Camera init error: {e}"
        print(error_msg)
        if conn:
            send_msg(conn, {"error": error_msg})
        raise

def stop_camera():     
    global dev, depth_stream, color_stream
    time.sleep(0.2)

    try:
        if depth_stream:
            depth_stream.stop()
        if color_stream:
            color_stream.stop()
        if dev:
            dev.close()
    except Exception as e:
        print(f"[Camera Worker] Error during camera stop: {e}")

    depth_stream = None
    color_stream = None
    dev = None

# Закидывать уже сдвинутый rgb
def draw_obj_contour(depth_background_data, depth_data, depth_frame, color_frame, kernel_value):
    kernel = np.ones((kernel_value, kernel_value), np.uint8)

    depth_filtered = cv2.medianBlur(depth_data, kernel_value)
    depth_diff = cv2.absdiff(depth_filtered, depth_background_data)
    _, depth_thresh = cv2.threshold(depth_diff, 50, 255, cv2.THRESH_BINARY)
    depth_thresh = depth_thresh.astype(np.uint8)

    depth_thresh = cv2.morphologyEx(depth_thresh, cv2.MORPH_OPEN, kernel)
    depth_thresh = cv2.morphologyEx(depth_thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(depth_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea) if contours else None 

    depth_frame_new = depth_frame.copy()
    color_frame_new = color_frame.copy()

    if main_contour is not None and len(main_contour) > 0:
        cv2.drawContours(depth_frame_new, [main_contour], -1, (0, 255, 0), 2)
        cv2.drawContours(color_frame_new, [main_contour], -1, (0, 255, 0), 2)

    return depth_frame_new, color_frame_new

def camera_loop(conn=None):
    global camera_turn_off_event, camera_stop_event, camera_running, camera_cycle_is_on, last_depth, last_color, not_video_mode_event
    global background_frame, sh_contour, sh_cntr_shift_x, sh_cntr_shift_y, sh_cntr_stretch_x, sh_cntr_stretch_y, kernel_value, depth_trunk
    global not_video_depth_frame, not_video_color_frame, not_video_crop_width, not_video_crop_height 

    # камера работает
    while not camera_turn_off_event.is_set():
        # камеру не надо стопать
        while not camera_stop_event.is_set(): 
            camera_cycle_is_on = True
            try: 
                if dev is None or depth_stream is None or color_stream is None:
                    camera_running = False

                    error_msg = f"[Camera Worker] Error reading frame: {e}"
                    print(error_msg)
                    if conn:
                        send_msg(conn, {"error": error_msg})
                    
                    camera_stop_event.set()
                    camera_turn_off_event.set()
                    break

                depth_frame = depth_stream.read_frame()
                color_frame = color_stream.read_frame()

                depth_data_full = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape((480, 640))
                color_data_full = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8).reshape((480, 640, 3))
                color_bgr_full = cv2.cvtColor(color_data_full, cv2.COLOR_RGB2BGR)

                h = min(camera_crop_height, 480)
                w = min(camera_crop_width, 640)

                depth_data = depth_data_full[0:h, 0:w]
                color_bgr = color_bgr_full[0:h, 0:w]

                depth_norm = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
                colormap_code = COLORMAPS.get(current_colormap, cv2.COLORMAP_JET)
                depth_colored = cv2.applyColorMap(depth_norm.astype(np.uint8), colormap_code)

                last_depth = depth_data.copy()
                last_color = color_bgr.copy()

                # ------- Смещение ------- 
                center_x = w / 2
                center_y = h / 2

                scale_x = 1 + (sh_cntr_stretch_x / 100.0)
                scale_y = 1 + (sh_cntr_stretch_y / 100.0)

                # Матрица масштабирования с привязкой к центру изображения
                M_scale = np.array([
                    [scale_x, 0, center_x * (1 - scale_x)],
                    [0, scale_y, center_y * (1 - scale_y)]
                ], dtype=np.float32)

                # Матрица сдвига
                M_shift = np.array([
                    [1, 0, sh_cntr_shift_x],
                    [0, 1, sh_cntr_shift_y]
                ], dtype=np.float32)

                # # Итоговая матрица — сначала масштабирование от центра, потом сдвиг
                # M_combined = M_shift @ M_scale
                # color_bgr = cv2.warpAffine(color_bgr, M_combined, (w, h))

                color_bgr = cv2.warpAffine(color_bgr, M_scale, (w, h))
                color_bgr = cv2.warpAffine(color_bgr, M_shift, (w, h))

                # ------- Обрезание -------
                depth_trunk_mm = depth_trunk * 1000 

                mask_far = depth_data > depth_trunk_mm
                color_bgr[mask_far] = (255, 255, 255)

                if sh_contour == True: 
                    depth_colored, color_bgr = draw_obj_contour(background_frame, depth_data, depth_colored, color_bgr, kernel_value)

                cv2.imshow("Color", color_bgr)
                cv2.imshow("Depth", depth_colored)
                cv2.waitKey(1)

            except Exception as e:
                error_msg = f"[Camera Worker] Error reading frame: {e}"
                print(error_msg)
                if conn:
                    send_msg(conn, {"error": error_msg})

        camera_cycle_is_on = False
        print("[Camera Worker] Cam stop event") 
        time.sleep(0.5)
        
        while not_video_mode_event.is_set():
            # print("[Camera Worker] Is not video mode")     
            # time.sleep(2)
            try: 
                # not_video_depth_frame, not_video_color_frame, not_video_crop_height, not_video_crop_width

                h = min(not_video_crop_height, 480)
                w = min(not_video_crop_width, 640)

                depth_data = not_video_depth_frame[0:h, 0:w]
                color_bgr = not_video_color_frame[0:h, 0:w]

                depth_norm = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
                colormap_code = COLORMAPS.get(current_colormap, cv2.COLORMAP_JET)
                depth_colored = cv2.applyColorMap(depth_norm.astype(np.uint8), colormap_code)

                last_depth = depth_data.copy()
                last_color = color_bgr.copy()

                # ------- Смещение ------- 
                # M = np.float32([
                #     [1 + sh_cntr_stretch_x, 0, sh_cntr_shift_x],
                #     [0, 1 + sh_cntr_stretch_y, sh_cntr_shift_y]
                # ])

                # color_bgr = cv2.warpAffine(color_bgr, M, (w, h))

                center_x = w / 2
                center_y = h / 2

                scale_x = 1 + (sh_cntr_stretch_x / 100.0)
                scale_y = 1 + (sh_cntr_stretch_y / 100.0)

                # Матрица масштабирования с привязкой к центру изображения
                M_scale = np.array([
                    [scale_x, 0, center_x * (1 - scale_x)],
                    [0, scale_y, center_y * (1 - scale_y)]
                ], dtype=np.float32)

                # Матрица сдвига
                M_shift = np.array([
                    [1, 0, sh_cntr_shift_x],
                    [0, 1, sh_cntr_shift_y]
                ], dtype=np.float32)

                # # Итоговая матрица — сначала масштабирование от центра, потом сдвиг
                # M_combined = M_shift @ M_scale
                # color_bgr = cv2.warpAffine(color_bgr, M_combined, (w, h))

                color_bgr = cv2.warpAffine(color_bgr, M_scale, (w, h))
                color_bgr = cv2.warpAffine(color_bgr, M_shift, (w, h))

                # ------- Обрезание -------
                depth_trunk_mm = depth_trunk * 1000 

                mask_far = depth_data > depth_trunk_mm
                color_bgr[mask_far] = (255, 255, 255)

                if sh_contour == True: 
                    # depth_colored, color_bgr = draw_obj_contour(background_frame, depth_data, depth_colored, color_bgr, kernel_value)
                    depth_colored, color_bgr = draw_obj_contour2(background_frame, depth_data, depth_colored, color_bgr, 3)

                cv2.imshow("Color", color_bgr)
                cv2.imshow("Depth", depth_colored)
                cv2.waitKey(1)

            except Exception as e:
                error_msg = f"[Camera Worker] Error reading frame in not video: {e}"
                print(error_msg)
                if conn:
                    send_msg(conn, {"error": error_msg})

        print("[Camera Worker] Not video")

    # камеру выключили
    try:
        cv2.destroyAllWindows()
    except Exception as e:
        error_msg = f"[Camera Worker] Error destroying windows: {e}"
        print(error_msg)
        if conn:
            send_msg(conn, {"error": error_msg})
    print("[Camera Worker] ended camera loop")


import numpy as np
import cv2

def draw_obj_contour2(depth_background, depth_object, depth_colored, color_bgr, kernel_value=3):
    # === Порог фильтрации ===
    max_distance_mm = 1000

    # === Копия карты глубины для редактирования ===
    depth_colored = depth_colored.copy()

    # === Визуализация фона ===
    # depth_bg_normalized = cv2.normalize(depth_background, None, 0, 255, cv2.NORM_MINMAX)
    # depth_bg_colored = cv2.applyColorMap(depth_bg_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    # depth_bg_colored[depth_background > max_distance_mm] = [255, 255, 255]

    # === Фильтрация по дальности ===
    # depth_colored[depth_object > max_distance_mm] = [255, 255, 255]

    # === Определение значения верхнего левого пикселя ===
    top_left_value = depth_object[0, 0]
    # print(f"[Camera Thread] Top left pixel: {top_left_value}")

    # === Маска совпадений с этим значением ===
    mask_top_left = (depth_object == top_left_value)
    highlighted_depth_colored = depth_colored.copy()
    # highlighted_depth_colored[mask_top_left] = [0, 0, 255]  # Чисто красный

    # === ВЫЧИТАНИЕ ФОНА ===
    depth_diff = cv2.absdiff(depth_object, depth_background)

    # === БИНАРИЗАЦИЯ ===
    _, binary_mask = cv2.threshold(depth_diff, 30, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)

    binary_mask[depth_object > max_distance_mm] = 0
    binary_mask[depth_object == top_left_value] = 0

    # === Морфологическая обработка ===
    kernel = np.ones((kernel_value, kernel_value), np.uint8)
    cleaned_mask = binary_mask.copy()
    cleaned_mask = cv2.erode(cleaned_mask, kernel, iterations=2)
    cleaned_mask = cv2.dilate(cleaned_mask, kernel, iterations=11)
    cleaned_mask = cv2.erode(cleaned_mask, kernel, iterations=3)

    # === Контур на маске ===
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    depth_with_contour = highlighted_depth_colored.copy()
    color_with_contour = color_bgr.copy()

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(depth_with_contour, [largest_contour], -1, (0, 255, 0), 2)
        cv2.drawContours(color_with_contour, [largest_contour], -1, (0, 255, 0), 2)
    else:
        print("[INFO] Контуры не найдены")

    return depth_with_contour, color_with_contour



def handle_client(conn):
    global shutdown_flag, camera_running, device_config, camera_thread, camera_stop_event, camera_turn_off_event, camera_cycle_is_on, not_video_mode_event, background_frame
    print("[Camera Worker] Client connected")
    send_msg(conn, {"reply": "hello"})

    with conn:
        while not shutdown_flag:
            try:
                data = conn.recv(1024)
                if not data:
                    break

                try:
                    msg = json.loads(data.decode())
                except json.JSONDecodeError:
                    send_msg(conn, {"error": "Invalid JSON"})
                    continue

                print("[Camera Worker] Got:", msg)

                cmd = msg.get("cmd")

                if cmd == "ping":
                    send_msg(conn, {"reply": "pong"})

                elif cmd == "shutdown":
                    shutdown_flag = True
                    send_msg(conn, {"reply": "goodbye"})
                    break

                elif cmd == "start":
                    if not camera_running:
                        try:
                            init_camera(config=device_config, conn=conn)
                            camera_running = True
                            camera_stop_event.clear()
                            camera_turn_off_event.clear()
                            camera_thread = threading.Thread(target=camera_loop, args=(conn,), daemon=True)
                            camera_thread.start()

                            send_msg(conn, {"status": "camera_started"})
                        except Exception:
                            camera_running = False # ошибка уже отправлена внутри init_camera
                            camera_stop_event.set()
                            camera_turn_off_event.set()
                    else:
                        send_msg(conn, {"status": "already_running"})

                elif cmd == "no_cam_start":
                    try:
                        camera_running = True
                        camera_stop_event.set()
                        camera_turn_off_event.clear()
                        not_video_mode_event.set()
                        
                        camera_thread = threading.Thread(target=camera_loop, args=(conn,), daemon=True)
                        camera_thread.start()

                        send_msg(conn, {"status": "camera_no_cam started"})
                    except Exception as e:
                        error_msg = f"[Camera Worker] No camera start error: {e}"
                        print(error_msg)
                        if conn:
                            send_msg(conn, {"error": error_msg})

                # elif cmd == "stop":
                #     if camera_running:
                #         camera_running = False
                #         camera_wait_event.set()
                #         camera_turn_off_event.set()
                        
                #         if camera_thread is not None:
                #             if camera_thread.is_alive():
                #                 camera_thread.join()
                #         stop_camera()
                #         send_msg(conn, {"status": "camera_stopped"})
                #     else:
                #         send_msg(conn, {"status": "camera_not_running"})

                elif cmd == "set_colormap":
                    name = msg.get("name", "").lower()
                    if name in COLORMAPS:
                        global current_colormap
                        current_colormap = name
                        send_msg(conn, {"status": f"colormap_set_to_{name}"})
                        print(f"[Camera Worker] Colormap changed to {name}")
                    else:
                        send_msg(conn, {"error": f"Unknown colormap: {name}"})

                elif cmd == "set_crop":
                    new_w = int(msg.get("width", 640))
                    new_h = int(msg.get("height", 480))

                    if 0 < new_w <= 640 and 0 < new_h <= 480:
                        global camera_crop_width, camera_crop_height
                        camera_crop_width = new_w
                        camera_crop_height = new_h
                        send_msg(conn, {"status": "crop_updated", "width": new_w, "height": new_h})
                        print(f"[Camera Worker] Crop updated to {new_w}x{new_h}")
                    else:
                        send_msg(conn, {"error": "Invalid crop size (must be <= 640x480)"})

                elif cmd == "set_config":
                    params = msg.get("params", {})
                    try:
                        registration = bool(params.get("registration", device_config.registration))
                        sync = bool(params.get("sync", device_config.sync))
                        mirror = bool(params.get("mirror", device_config.mirror))

                        device_config = DeviceConfig(registration=registration, sync=sync, mirror=mirror)

                        if camera_running:
                            camera_running = False
                            camera_stop_event.set()
                            not_video_mode_event.clear()
                            global depth_stream, color_stream

                            while camera_cycle_is_on == True:
                                pass

                            depth_stream.stop()
                            color_stream.stop()
                            apply_camera_config(device_config, conn)
                            depth_stream.start()
                            color_stream.start()

                            camera_running = True
                            camera_stop_event.clear()
                          
                            send_msg(conn, {"status": "config_updated", "config": device_config.to_dict()})
                            print(f"[Camera Worker] Config applied restarted")

                    except Exception as e:
                        error_msg = f"[Camera Worker] Camera reconfigure error: {e}"
                        print(error_msg)
                        if conn:
                            send_msg(conn, {"error": error_msg})

                # color_path = "dir/color_name.png" 
                # depth_path = "dir/depth_name.npy"
                elif cmd == "make_photo":
                    dir_path = msg.get("dir_path", "").strip()
                    depth_name = msg.get("depth_name", "").strip()
                    color_name = msg.get("color_name", "").strip()

                    if not dir_path or not depth_name or not color_name:
                        send_msg(conn, {"error": "Missing dir_path or name for make_photo"})
                        continue
                    
                    if camera_running:
                        camera_stop_event.set()
                        global last_depth, last_color

                        while camera_cycle_is_on == True:
                            pass
                        
                        try:
                            os.makedirs(dir_path, exist_ok=True)
                            color_path = dir_path + "/" + color_name + ".png" 
                            depth_path = dir_path + "/" + depth_name + ".npy" 
                            

                            cv2.imwrite(color_path, last_color)
                            np.save(depth_path, last_depth)

                            send_msg(conn, {"status": "good_photo", "depth_path": depth_path, "color_path": color_path})
                            print(f"[Camera Worker] Made photo")

                        except Exception as e:
                            error_msg = f"[Camera Worker] Camera make photo error: {e}"
                            print(error_msg)
                            if conn:
                                send_msg(conn, {"error": error_msg})

                        camera_stop_event.clear()
                        
                        # send_msg(conn, {"status": "config_updated", "config": device_config.to_dict()})
                        # print(f"[Camera Worker] Config applied restarted")

                # frame_proc commands
                elif cmd == "set_show_contour":
                    try:
                        global background_path, sh_contour 
                        got_sh_contour = bool(msg.get("value", False))
                        background_path = str(msg.get("dir_path", "KILL MEEEE"))

                        background_frame = np.load(background_path)

                        sh_contour = got_sh_contour

                        send_msg(conn, {"status": f"show_contour_set_to_{sh_contour}"})
                        print(f"[Camera Worker] show_contour set to {sh_contour}")
                    except Exception as e:
                        send_msg(conn, {"error": f"Failed to set show_contour: {e}"})

                elif cmd == "set_video_mode":
                    try:
                        global video_mode, not_video_depth_frame, not_video_color_frame 
                        got_video_mode = bool(msg.get("value", False))

                        if got_video_mode == False:
                            background_path = str(msg.get("background_path", "KILL MEEEE"))
                            object_depth_path = str(msg.get("object_depth_path", "KILL MEEEE"))
                            object_color_path = str(msg.get("object_color_path", "KILL MEEEE"))

                            background_frame = np.load(background_path)
                            not_video_depth_frame = np.load(object_depth_path)
                            not_video_color_frame = cv2.imread(object_color_path)

                            global not_video_crop_width, not_video_crop_height 
                            not_video_crop_height, not_video_crop_width = not_video_depth_frame.shape

                            if camera_running:
                                camera_stop_event.set()

                                while camera_cycle_is_on == True:
                                    pass
                                
                                depth_stream.stop()
                                color_stream.stop()
                                video_mode = True 
                                not_video_mode_event.set()    
                        else:
                            not_video_mode_event.clear()

                            depth_stream.start()
                            color_stream.start()

                            camera_stop_event.clear() 

                        send_msg(conn, {"status": f"set_video_mode_to_{video_mode}"})
                        print(f"[Camera Worker] set_video_mode_to_{video_mode}")
                    except Exception as e:
                        not_video_mode_event.set()
                        send_msg(conn, {"error": f"Failed to set video mode: {e}"})

                elif cmd == "set_frpr_shift_x":
                    try:
                        global sh_cntr_shift_x
                        shift_x = int(msg.get("shift_x", 0))
                        sh_cntr_shift_x = shift_x

                        send_msg(conn, {"status": "cntr_shif_x_updated", "shift_x": shift_x})
                    except Exception as e:
                        send_msg(conn, {"error": f"Failed to set contour shift x: {e}"})

                elif cmd == "set_frpr_shift_y":
                    try:
                        global sh_cntr_shift_y
                        shift_y = int(msg.get("shift_y", 0))
                        sh_cntr_shift_y = shift_y
                        send_msg(conn, {"status": "cntr_shift_y_updated", "shift_y": shift_y})
                    except Exception as e:
                        send_msg(conn, {"error": f"Failed to set contour shift y: {e}"})

                elif cmd == "set_frpr_stretch_x":
                    try:
                        global sh_cntr_stretch_x
                        stretch_x = int(msg.get("stretch_x", 0))
                        sh_cntr_stretch_x = stretch_x
                        send_msg(conn, {"status": "cntr_stretch_x_updated", "stretch_x": stretch_x})
                    except Exception as e:
                        send_msg(conn, {"error": f"Failed to set contour stretch x: {e}"})

                elif cmd == "set_frpr_stretch_y":
                    try:
                        global sh_cntr_stretch_y
                        stretch_y = int(msg.get("stretch_y", 0))
                        sh_cntr_stretch_y = stretch_y
                        send_msg(conn, {"status": "cntr_stretch_y_updated", "stretch_y": stretch_y})
                    except Exception as e:
                        send_msg(conn, {"error": f"Failed to set contour stretch y: {e}"})

                elif cmd == "set_frpr_depth_trunc":
                    try:
                        global depth_trunk
                        depth_trunc = float(msg.get("depth_trunc", 10.0))
                        depth_trunk = depth_trunc
                        send_msg(conn, {"status": "depth_trunc_updated", "depth_trunc": depth_trunc})
                    except Exception as e:
                        send_msg(conn, {"error": f"Failed to set depth_trunc: {e}"})

                elif cmd == "set_frpr_kernel_value":
                    try:
                        global kernel_value
                        kernel = int(msg.get("kernel_value", 5))
                        if kernel < 1 or kernel % 2 == 0:
                            raise ValueError("Kernel size must be odd and >= 1")
                        kernel_value = kernel
                        send_msg(conn, {"status": "kernel_value_updated", "kernel_value": kernel})
                    except Exception as e:
                        send_msg(conn, {"error": f"Failed to set kernel_value: {e}"})

                else:
                    send_msg(conn, {"error": f"Unknown command: {cmd}"})

            except Exception as e:
                error_msg = f"[Camera Worker] Exception: {e}"
                print(error_msg)
                send_msg(conn, {"error": error_msg})
                break

    print("[Camera Worker] Client disconnected")

def run_server():
    global camera_running, camera_thread, camera_stop_event, camera_turn_off_event

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[Camera Worker] Listening on {HOST}:{PORT}")
        conn, _ = s.accept()
        handle_client(conn)

    if camera_running:
        camera_running = False
        camera_stop_event.set()
        camera_turn_off_event.set()
        not_video_mode_event.clear()
        #time.sleep(0.1)

        if camera_thread is not None:
            if camera_thread.is_alive():
                camera_thread.join(timeout=5.0)

                if camera_thread.is_alive():
                    print("[Camera Worker] WARNING: Camera thread did not stop after timeout :(")
        
        stop_camera()

if __name__ == "__main__":
    try:
        openni2.initialize()
    except Exception as e:
        print(f"[Camera Worker] Failed to initialize OpenNI2: {e}")
        sys.exit(1)

    run_server()

    try:
        openni2.unload()
        print("[Camera Worker] OpenNI2 unloaded.")
    except Exception as e:
        print(f"[Camera Worker] Failed to unload OpenNI2: {e}")  
    print("[Camera Worker] Exiting")
    sys.exit(0)


