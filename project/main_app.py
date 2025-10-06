import numpy as np

import sys
import json
import os
import socket
import subprocess
import threading
import time
import re
from threading import Lock

# import cv2
# from primesense import openni2

import numpy as np
import open3d as o3d

from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QCheckBox, QWidget, QHBoxLayout, QLineEdit, QMessageBox, QHeaderView, QTableWidget
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeyEvent

from mainwindow import Ui_MainWindow

CAM_HOST = '127.0.0.1'
CAM_PORT = 65432

TD_HOST = '127.0.0.1'
TD_PORT = 65433

class MyForm(QMainWindow, Ui_MainWindow):
    camera_show_error_signal = pyqtSignal(str)
    reload_camera_signal = pyqtSignal(str)
    made_good_photo_signal = pyqtSignal(str, str)

    esp_message_received = pyqtSignal(str)
    esp_disconnected_signal = pyqtSignal()

    scanning_good_end_signal = pyqtSignal(str)
    scanning_made_photo_signal = pyqtSignal(str)
    
    cloudsproc_deletePressed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.CONFIG_FILE = "config.json"

        # ---- Platform connection ---- 
        self.esp_sock = None
        self.esp_connected = False
        self.esp_listener_thread = None
        self.esp_listener_running = False

        self.esp_disconnected_signal.connect(self.disconnect_from_platform)

        # ---- Camera check ----
        self.camera_sock = None
        self.camera_show_error_signal.connect(self.show_camera_error_message)
        self.camera_running = False
        self.reload_camera_signal.connect(self.reload_camera)

        self.made_good_photo_signal.connect(self.made_good_photo_proc)

        self.colormaps = [
            ("Colormap Jet", "jet"),
            ("Colormap Bone", "bone"),
            ("Colormap Plasma", "plasma")
        ]

        # ---- Scanning proc ----
        self.scanning_process_thread = None
        self.scanning_process_is_on = False
        self.scanning_process_photo_maden = False
        self.scanning_process_rotate_maden = False
        self.scanning_process_rotate_num = -1

        self.scanning_good_end_signal.connect(self.scanning_good_end)
        self.scanning_made_photo_signal.connect(self.scanning_made_photo)

        # ---- 3d tab ----
        self.init_td_variables()

        # ---- Common ----
        self.shots_folder_line_syncing = False
        self.shots_object_line_syncing = False

        # ---- Инициализация полей ---- 
        self.set_default_platform_tab()
        self.set_default_cam_check_tab()
        self.set_default_scanning_tab()
        self.set_default_framesprocessing_tab()
        self.set_default_cloudsprocessing_tab()


        # ---- Настройка сигналов и связей ----
        self.set_connections_platform_tab()
        self.set_connections_cam_check_tab()
        self.set_connections_scanning_tab()
        self.set_connections_framesprocessing_tab()
        self.set_connections_cloudsprocessing_tab()

    # Загрузка конфигов
    def load_config(self):
        if os.path.exists(self.CONFIG_FILE):
            with open(self.CONFIG_FILE, "r") as f:
                return json.load(f)
        return {}
    
    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Delete:
            if self.options_tabWidget.currentIndex() == 4:
                self.cloudsproc_deletePressed.emit()

        super().keyPressEvent(event)

    # ---- Platform tab init ----
    def set_default_platform_tab(self):
        config = self.load_config().get("esp_connect_tab", {})

        self.platformConnectionSettings_platformPort_lineEdit.setText(config.get("platformPort", ""))
        self.platformConnectionSettings_platformIP_lineEdit.setText(config.get("platformIP", ""))
        self.platformConnectionSettings_ApSSID_lineEdit.setText(config.get("apSSID", ""))
        self.platformConnectionSettings_APPAssword_lineEdit.setText(config.get("apPassword", ""))

    def set_connections_platform_tab(self):
        self.platformConnectionSettings_connect_pushButton.clicked.connect(self.connect_to_platform)

        self.platformConnectionSettings_connect_pushButton_2.clicked.connect(self.set_new_user_ap_data)

    # ---- Cam check init ----
    def set_default_cam_check_tab(self):
        config = self.load_config().get("cam_check_tab", {})

        self.cameraCheck_cameraOutput_checkBox.setChecked(False)
        
        self.cameraCheck_depthToColor_checkBox.setChecked(True)
        self.cameraCheck_syncIsEnabled_checkBox.setChecked(True)
        self.cameraCheck_mirroringIsEnables_checkBox.setChecked(False)
        
        self.cameraCheck_resolutionX_spinBox.setMinimum(1)
        self.cameraCheck_resolutionX_spinBox.setMaximum(640)
        self.cameraCheck_resolutionX_spinBox.setSingleStep(1)
        #self.cameraCheck_resolutionX_spinBox.setValue(640)
        self.cameraCheck_resolutionX_spinBox.setValue(config.get("resolutionX", 640))
        
        self.cameraCheck_resolutionY_spinBox.setMinimum(1)
        self.cameraCheck_resolutionY_spinBox.setMaximum(480)
        self.cameraCheck_resolutionY_spinBox.setSingleStep(1)
        #self.cameraCheck_resolutionY_spinBox.setValue(480)
        self.cameraCheck_resolutionY_spinBox.setValue(config.get("resolutionY", 480))

        self.cameraCheck_chooseFolder_lineEdit.setText(config.get("folderName", ""))
        self.cameraCheck_photoName_lineEdit.setText(config.get("objectName", ""))

        self.cameraCheck_depthDisplayMethod_comboBox.clear()

        for display_name, internal_name in self.colormaps:
            self.cameraCheck_depthDisplayMethod_comboBox.addItem(display_name, internal_name)

    def set_connections_cam_check_tab(self):
        self.cameraCheck_cameraOutput_checkBox.stateChanged.connect(self.cam_check_camera_outp_on_off)

        self.cameraCheck_depthDisplayMethod_comboBox.currentIndexChanged.connect(self.update_colormap)
        self.cameraCheck_resolutionX_spinBox.valueChanged.connect(self.update_crop_width)
        self.cameraCheck_resolutionY_spinBox.valueChanged.connect(self.update_crop_height)
        self.cameraCheck_depthToColor_checkBox.stateChanged.connect(self.cam_check_depth_to_color_proc)
        self.cameraCheck_syncIsEnabled_checkBox.stateChanged.connect(self.cam_check_sync_set)
        self.cameraCheck_mirroringIsEnables_checkBox.stateChanged.connect(self.cam_check_mirror_set)

        self.cameraCheck_chooseFolder_pushButton.clicked.connect(self.choose_object_folder)
        self.cameraCheck_chooseFolder_lineEdit.textChanged.connect(self.sync_folder_lineedits)
        
        self.cameraCheck_makePhoto_pushButton.clicked.connect(self.make_photo)

    # ---- Scanning tab init ----
    def set_default_scanning_tab(self):
        config = self.load_config().get("scanning_tab", {})

        self.scanningSettings_chooseFolder_lineEdit.setText(config.get("folderName", ""))
        self.scanningSettings_objectName_lineEdit.setText(config.get("objectName", ""))

        self.scanningSettings_speed_spinBox.setRange(1, 90)
        self.scanningSettings_speed_spinBox.setSingleStep(1)
        self.scanningSettings_speed_spinBox.setValue(config.get("speed", 5))

        self.scanningSettings_acceeration_spinBox.setRange(1, 90)
        self.scanningSettings_acceeration_spinBox.setSingleStep(1)
        self.scanningSettings_acceeration_spinBox.setValue(config.get("acceleration", 2))

        self.scanningSettings_numverOfShots_spinBox.setRange(1, 99)
        self.scanningSettings_numverOfShots_spinBox.setSingleStep(1)
        self.scanningSettings_numverOfShots_spinBox.setValue(config.get("numberOfShots", 1))

        self.scanningSettings_rotationAngle_spinBox.setRange(1, 500)
        self.scanningSettings_rotationAngle_spinBox.setSingleStep(1)
        self.scanningSettings_rotationAngle_spinBox.setValue(config.get("rotationAngle", 10))

        
    def set_connections_scanning_tab(self):
        self.scanningSettings_chooseFolder_pushButton.clicked.connect(self.choose_object_folder)
        self.scanningSettings_chooseFolder_lineEdit.textChanged.connect(self.sync_folder_lineedits)
        self.scanningSettings_objectName_lineEdit.textChanged.connect(self.sync_object_lineedits)

        self.scanningSettings_showCamOutput_checkBox.stateChanged.connect(self.scanning_camera_outp_on_off)

        self.scanningSettings_start_pushButton.clicked.connect(self.start_scan_proc)
        self.scanningSettings_start_pushButton_2.clicked.connect(self.make_background_photo)

        
    # ---- Frames processing tab init ----
    def set_default_framesprocessing_tab(self):
        config = self.load_config().get("framesprocessing_tab", {})
        
        self.framesProcessing_folder_lineEdit.setText(config.get("folderName", ""))
        self.framesProcessing_objectName_lineEdit.setText(config.get("objectName", ""))   

        self.framesProcessing_manualAdjustment_checkBox.setChecked(True)   

        self.framesProcessing_depthTrunc_doubleSpinBox.setDecimals(2)
        self.framesProcessing_depthTrunc_doubleSpinBox.setRange(0.4, 10.0)
        self.framesProcessing_depthTrunc_doubleSpinBox.setSingleStep(0.01)
        self.framesProcessing_depthTrunc_doubleSpinBox.setValue(config.get("depthTrunk", 10))

        self.framesProcessing_morphKernelSize_doubleSpinBox.setDecimals(0)  
        self.framesProcessing_morphKernelSize_doubleSpinBox.setRange(1, 99)
        self.framesProcessing_morphKernelSize_doubleSpinBox.setSingleStep(2)  
        self.framesProcessing_morphKernelSize_doubleSpinBox.setValue(config.get("kernelSize", 5))

        self.framesProcessing_shiftX_doubleSpinBox.setDecimals(0)
        self.framesProcessing_shiftX_doubleSpinBox.setRange(-700, 700)
        self.framesProcessing_shiftX_doubleSpinBox.setSingleStep(1)
        self.framesProcessing_shiftX_doubleSpinBox.setValue(config.get("shift_x", 0))

        self.framesProcessing_shiftY_doubleSpinBox.setDecimals(0)
        self.framesProcessing_shiftY_doubleSpinBox.setRange(-700, 700)
        self.framesProcessing_shiftY_doubleSpinBox.setSingleStep(1)
        self.framesProcessing_shiftY_doubleSpinBox.setValue(config.get("shift_y", 0))

        self.framesProcessing_stretchX_doubleSpinBox.setDecimals(0)
        self.framesProcessing_stretchX_doubleSpinBox.setRange(-700, 700)
        self.framesProcessing_stretchX_doubleSpinBox.setSingleStep(1)
        self.framesProcessing_stretchX_doubleSpinBox.setValue(config.get("stretch_x", 0))

        self.framesProcessing_stretchY_doubleSpinBox.setDecimals(0)
        self.framesProcessing_stretchY_doubleSpinBox.setRange(-700, 700)
        self.framesProcessing_stretchY_doubleSpinBox.setSingleStep(1)
        self.framesProcessing_stretchY_doubleSpinBox.setValue(config.get("stretch_y", 0))

        self.framesProcessing_videoMode_checkBox.setChecked(True)

        self.framesProcessing_photoReferenceAngle_doubleSpinBox.setValue(config.get("ref_angle", 0))
        self.framesProcessing_photoReferenceAngle_doubleSpinBox.setDecimals(0)
        self.framesProcessing_photoReferenceAngle_doubleSpinBox.setRange(0, 999)
        self.framesProcessing_photoReferenceAngle_doubleSpinBox.setSingleStep(1)

        self.framesProcessing_folderToSave_lineEdit.setText(config.get("folderName_3ds", ""))

        self.framesProcessing_depthCameraIntrinsics_checkBox.setChecked(True)

        # Depth Camera Intrinsics - width и height (целые числа)
        self.framesProcessing_depthCameraIntrinsics_width_doubleSpinBox.setDecimals(0)
        self.framesProcessing_depthCameraIntrinsics_width_doubleSpinBox.setRange(1, 640)
        self.framesProcessing_depthCameraIntrinsics_width_doubleSpinBox.setSingleStep(1)
        self.framesProcessing_depthCameraIntrinsics_width_doubleSpinBox.setValue(config.get("depth_width", 640))

        self.framesProcessing_depthCameraIntrinsics_heigth_doubleSpinBox.setDecimals(0)
        self.framesProcessing_depthCameraIntrinsics_heigth_doubleSpinBox.setRange(1, 480)
        self.framesProcessing_depthCameraIntrinsics_heigth_doubleSpinBox.setSingleStep(1)
        self.framesProcessing_depthCameraIntrinsics_heigth_doubleSpinBox.setValue(config.get("depth_height", 480))

        # Depth Camera Intrinsics - fx, fy, cx, cy (2 знака после запятой, 0.00 .. 999.99)
        self.framesProcessing_depthCameraIntrinsics_fx_doubleSpinBox.setDecimals(2)
        self.framesProcessing_depthCameraIntrinsics_fx_doubleSpinBox.setRange(0.00, 999.99)
        self.framesProcessing_depthCameraIntrinsics_fx_doubleSpinBox.setSingleStep(0.01)
        self.framesProcessing_depthCameraIntrinsics_fx_doubleSpinBox.setValue(config.get("depth_fx", 554.26))

        self.framesProcessing_depthCameraIntrinsics_fy_doubleSpinBox.setDecimals(2)
        self.framesProcessing_depthCameraIntrinsics_fy_doubleSpinBox.setRange(0.00, 999.99)
        self.framesProcessing_depthCameraIntrinsics_fy_doubleSpinBox.setSingleStep(0.01)
        self.framesProcessing_depthCameraIntrinsics_fy_doubleSpinBox.setValue(config.get("depth_fy", 579.41))

        self.framesProcessing_depthCameraIntrinsics_cx_doubleSpinBox.setDecimals(2)
        self.framesProcessing_depthCameraIntrinsics_cx_doubleSpinBox.setRange(0.00, 999.99)
        self.framesProcessing_depthCameraIntrinsics_cx_doubleSpinBox.setSingleStep(0.01)
        self.framesProcessing_depthCameraIntrinsics_cx_doubleSpinBox.setValue(config.get("depth_cx", 320.0))

        self.framesProcessing_depthCameraIntrinsics_cy_doubleSpinBox.setDecimals(2)
        self.framesProcessing_depthCameraIntrinsics_cy_doubleSpinBox.setRange(0.00, 999.99)
        self.framesProcessing_depthCameraIntrinsics_cy_doubleSpinBox.setSingleStep(0.01)
        self.framesProcessing_depthCameraIntrinsics_cy_doubleSpinBox.setValue(config.get("depth_cy", 240.0))


        # RGB Camera Intrinsics - width и height (целые числа)
        self.framesProcessing_RGBCameraIntrinsics_width_doubleSpinBox.setDecimals(0)
        self.framesProcessing_RGBCameraIntrinsics_width_doubleSpinBox.setRange(1, 640)
        self.framesProcessing_RGBCameraIntrinsics_width_doubleSpinBox.setSingleStep(1)
        self.framesProcessing_RGBCameraIntrinsics_width_doubleSpinBox.setValue(config.get("rgb_width", 640))

        self.framesProcessing_RGBCameraIntrinsics_height_doubleSpinBox.setDecimals(0)
        self.framesProcessing_RGBCameraIntrinsics_height_doubleSpinBox.setRange(1, 480)
        self.framesProcessing_RGBCameraIntrinsics_height_doubleSpinBox.setSingleStep(1)
        self.framesProcessing_RGBCameraIntrinsics_height_doubleSpinBox.setValue(config.get("rgb_height", 480))

        # RGB Camera Intrinsics - fx, fy, cx, cy (2 знака после запятой, 0.00 .. 999.99)
        self.framesProcessing_RGBCameraIntrinsics_fx_doubleSpinBox.setDecimals(2)
        self.framesProcessing_RGBCameraIntrinsics_fx_doubleSpinBox.setRange(0.00, 999.99)
        self.framesProcessing_RGBCameraIntrinsics_fx_doubleSpinBox.setSingleStep(0.01)
        self.framesProcessing_RGBCameraIntrinsics_fx_doubleSpinBox.setValue(config.get("rgb_fx", 554.26))

        self.framesProcessing_RGBCameraIntrinsics_fy_doubleSpinBox.setDecimals(2)
        self.framesProcessing_RGBCameraIntrinsics_fy_doubleSpinBox.setRange(0.00, 999.99)
        self.framesProcessing_RGBCameraIntrinsics_fy_doubleSpinBox.setSingleStep(0.01)
        self.framesProcessing_RGBCameraIntrinsics_fy_doubleSpinBox.setValue(config.get("rgb_fy", 579.41))

        self.framesProcessing_RGBCameraIntrinsics_cx_doubleSpinBox.setDecimals(2)
        self.framesProcessing_RGBCameraIntrinsics_cx_doubleSpinBox.setRange(0.00, 999.99)
        self.framesProcessing_RGBCameraIntrinsics_cx_doubleSpinBox.setSingleStep(0.01)
        self.framesProcessing_RGBCameraIntrinsics_cx_doubleSpinBox.setValue(config.get("rgb_cx", 320.0))

        self.framesProcessing_RGBCameraIntrinsics_cy_doubleSpinBox.setDecimals(2)
        self.framesProcessing_RGBCameraIntrinsics_cy_doubleSpinBox.setRange(0.00, 999.99)
        self.framesProcessing_RGBCameraIntrinsics_cy_doubleSpinBox.setSingleStep(0.01)
        self.framesProcessing_RGBCameraIntrinsics_cy_doubleSpinBox.setValue(config.get("rgb_cy", 240.0))


        # Number of clouds (целое число)
        self.framesProcessing_numberOfClouds_doubleSpinBox.setDecimals(0)
        self.framesProcessing_numberOfClouds_doubleSpinBox.setRange(1, 999)
        self.framesProcessing_numberOfClouds_doubleSpinBox.setSingleStep(1)
        self.framesProcessing_numberOfClouds_doubleSpinBox.setValue(config.get("numberOfClouds", 1))


    def set_connections_framesprocessing_tab(self):
        self.framesProcessing_folder_pushButton.clicked.connect(self.choose_object_folder)
        self.framesProcessing_folder_lineEdit.textChanged.connect(self.sync_folder_lineedits)
        self.framesProcessing_objectName_lineEdit.textChanged.connect(self.sync_object_lineedits)

        self.framesProcessing_manualAdjustment_checkBox.stateChanged.connect(self.frames_manual_adjustment_activation)

        self.framesProcessing_onOffImage_checkBox.stateChanged.connect(self.frames_proc_outp_on_off)
        self.framesProcessing_showCountour_checkBox.stateChanged.connect(self.send_show_contour)
        self.framesProcessing_videoMode_checkBox.stateChanged.connect(self.framesproc_videomode_on_off)

        self.framesProcessing_shiftX_doubleSpinBox.valueChanged.connect(self.framesproc_send_shift_x)
        self.framesProcessing_shiftY_doubleSpinBox.valueChanged.connect(self.framesproc_send_shift_y)
        self.framesProcessing_stretchX_doubleSpinBox.valueChanged.connect(self.framesproc_send_stretch_x)
        self.framesProcessing_stretchY_doubleSpinBox.valueChanged.connect(self.framesproc_send_stretch_y)

        self.framesProcessing_depthTrunc_doubleSpinBox.valueChanged.connect(self.framesproc_send_depth_trunc)
        self.framesProcessing_morphKernelSize_doubleSpinBox.valueChanged.connect(self.framesproc_send_kernel_value)

        self.framesProcessing_folderToSave_pushButton.clicked.connect(self.choose_3d_folder)
        self.framesProcessing_apply_pushButton.clicked.connect(self.get_o3d_datas)

    # ---- Clouds processing tab init ----
    def set_default_cloudsprocessing_tab(self):
        config = self.load_config().get("cloudsprocessing_tab", {})

        self.last_td_dir = config.get("last_tddir", "")

        self.cloudsProcessing_cloudsList_tableWidget.verticalHeader().setVisible(False) 
        self.cloudsProcessing_cloudsList_tableWidget.setColumnWidth(0, 45) 
        self.cloudsProcessing_cloudsList_tableWidget.setColumnWidth(1, 45)
        self.cloudsProcessing_cloudsList_tableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.cloudsProcessing_cloudsList_tableWidget.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)

        self.cloudsProcessing_cloudsList_tableWidget.horizontalHeader().setStretchLastSection(True)

        self.cloudsProcessing_uploadFiltration_checkBox.setChecked(config.get("cloudsProcessing_uploadFiltration", True))

        self.cloudsProcessing_filtrationNeighbors_doubleSpinBox.setRange(1, 100)
        self.cloudsProcessing_filtrationNeighbors_doubleSpinBox.setDecimals(0)
        self.cloudsProcessing_filtrationNeighbors_doubleSpinBox.setValue(config.get("cloudsProcessing_filtrationNeighbors", 40))
        
        self.cloudsProcessing_outlierSensitivity_doubleSpinBox.setRange(0.1, 2.0)
        self.cloudsProcessing_outlierSensitivity_doubleSpinBox.setSingleStep(0.01)
        self.cloudsProcessing_outlierSensitivity_doubleSpinBox.setDecimals(2)
        self.cloudsProcessing_outlierSensitivity_doubleSpinBox.setValue(config.get("cloudsProcessing_outlierSensitivity", 1.0))
        
        self.cloudsProcessing_enableDistanceFilter_checkBox.setChecked(config.get("cloudsProcessing_enableDistanceFilter", True))

        self.cloudsProcessing_uploadFilterMaxDistance_doubleSpinBox.setRange(0.6, 8.0)
        self.cloudsProcessing_uploadFilterMaxDistance_doubleSpinBox.setSingleStep(0.01)
        self.cloudsProcessing_uploadFilterMaxDistance_doubleSpinBox.setDecimals(2)
        self.cloudsProcessing_uploadFilterMaxDistance_doubleSpinBox.setValue(config.get("cloudsProcessing_uploadFilterMaxDistance", 8.0))

        self.cloudsProcessing_uploadFilterMinDistance_doubleSpinBox.setRange(0.6, 8.0)
        self.cloudsProcessing_uploadFilterMaxDistance_doubleSpinBox.setSingleStep(0.01)
        self.cloudsProcessing_uploadFilterMaxDistance_doubleSpinBox.setDecimals(2)
        self.cloudsProcessing_uploadFilterMinDistance_doubleSpinBox.setValue(config.get("cloudsProcessing_uploadFilterMinDistance", 0.6))

        self.cloudsProcessing_preloadShift_checkBox.setChecked(config.get("cloudsProcessing_preloadShift", False))

        for spinbox in [
            self.cloudsProcessing_preloadAxisShiftX_doubleSpinBox,
            self.cloudsProcessing_preloadAxisShiftY_doubleSpinBox,
            self.cloudsProcessing_preloadAxisShiftZ_doubleSpinBox
        ]:
            spinbox.setDecimals(2)
            spinbox.setRange(-999.00, 999.00)
            spinbox.setSingleStep(0.01)
            spinbox.setValue(0.00)

        for spinbox in [
            self.cloudsProcessing_preloadRotationShiftXY_doubleSpinBox,
            self.cloudsProcessing_preloadRotationShiftXZ_doubleSpinBox,
            self.cloudsProcessing_preloadRotationShiftYZ_doubleSpinBox
        ]:
            spinbox.setDecimals(0)
            spinbox.setRange(-360, 360)
            spinbox.setSingleStep(1)
            spinbox.setValue(0)

        self.cloudsProcessing_preloadAxisShiftX_doubleSpinBox.setValue(config.get("preload_shift_x", 0.0))
        self.cloudsProcessing_preloadAxisShiftY_doubleSpinBox.setValue(config.get("preload_shift_y", 0.0))
        self.cloudsProcessing_preloadAxisShiftZ_doubleSpinBox.setValue(config.get("preload_shift_z", 0.0))

        self.cloudsProcessing_preloadRotationShiftXY_doubleSpinBox.setValue(config.get("preload_rot_xy", 0.0))
        self.cloudsProcessing_preloadRotationShiftXZ_doubleSpinBox.setValue(config.get("preload_rot_xz", 0.0))
        self.cloudsProcessing_preloadRotationShiftYZ_doubleSpinBox.setValue(config.get("preload_rot_yz", 0.0))

        for slider in [
            self.cloudsProcessing_manualAdjustmentShiftX_horizontalSlider,
            self.cloudsProcessing_manualAdjustmentShiftY_horizontalSlider,
            self.cloudsProcessing_manualAdjustmentShiftZ_horizontalSlider,
        ]:
            slider.setMinimum(-100)  
            slider.setMaximum(100)   
            slider.setSingleStep(1)  # шаг 0.01 м
            slider.setValue(0)       # центр

        for slider in [
            self.cloudsProcessing_manualAdjustmentRotShiftXY_horizontalSlider,
            self.cloudsProcessing_manualAdjustmentRotShiftXZ_horizontalSlider,
            self.cloudsProcessing_manualAdjustmentRotShiftYZ_horizontalSlider,
        ]:
            slider.setMinimum(-180)
            slider.setMaximum(180)
            slider.setSingleStep(1)
            slider.setValue(0)

        self.cloudsProcessing_manualAdjustmentShiftX_textBrowser.setText("0.00")
        self.cloudsProcessing_manualAdjustmentShiftY_textBrowser.setText("0.00")
        self.cloudsProcessing_manualAdjustmentShiftZ_textBrowser.setText("0.00")
        self.cloudsProcessing_manualAdjustmentRotShiftXY_textBrowser.setText("0")
        self.cloudsProcessing_manualAdjustmentRotShiftXZ_textBrowser.setText("0")
        self.cloudsProcessing_manualAdjustmentRotShiftYZ_textBrowser.setText("0")

        def init_combobox(widget, items, default):
            widget.clear()
            for name in items:
                widget.addItem(name, userData=name)
            index = widget.findData(config.get(widget.objectName(), default))
            if index != -1:
                widget.setCurrentIndex(index)
            else:
                widget.setCurrentIndex(0)

        init_combobox(self.cloudsProcessing_downsamplingMode_comboBox, ["fixed", "adaptive"], "fixed")

        self.cloudsProcessing_downsampling_vovelSize_doubleSpinBox.setDecimals(3)
        self.cloudsProcessing_downsampling_vovelSize_doubleSpinBox.setRange(0.001, 0.1)
        self.cloudsProcessing_downsampling_vovelSize_doubleSpinBox.setSingleStep(0.001)
        self.cloudsProcessing_downsampling_vovelSize_doubleSpinBox.setValue(config.get("cloudsProcessing_downsampling_vovelSize", 0.002))

        self.cloudsProcessing_downsamplingMinPointsPerVoxel_doubleSpinBox.setDecimals(0)
        self.cloudsProcessing_downsamplingMinPointsPerVoxel_doubleSpinBox.setRange(1, 20)
        self.cloudsProcessing_downsamplingMinPointsPerVoxel_doubleSpinBox.setSingleStep(1)
        self.cloudsProcessing_downsamplingMinPointsPerVoxel_doubleSpinBox.setValue(config.get("cloudsProcessing_downsamplingMinPntsPerVoxel", 5))
        
        self.cloudsProcessing_normalsCalculation_checkBox.setChecked(config.get("cloudsProcessing_normalsCalculation", True))
        self.cloudsProcessing_normalEstimationRadius_doubleSpinBox.setRange(1.0, 10.0)
        self.cloudsProcessing_normalEstimationRadius_doubleSpinBox.setValue(config.get("cloudsProcessing_normalEstimationRadius", 2.0))
        self.cloudsProcessing_normalEstimationRadius_doubleSpinBox.setDecimals(1)
        self.cloudsProcessing_normalEstimationRadius_doubleSpinBox.setSingleStep(0.1)

        self.cloudsProcessing_globalRegistrationIsEnabled_checkBox.setChecked(config.get("cloudsProcessing_globalRegistrationIsEnabled", True))
        self.cloudsProcessing_featureRadiusMultipliers_lineEdit.setText(config.get("cloudsProcessing_featureRadiusMultipliers", "x3.0, x5.0, x8.0"))
        init_combobox(self.cloudsProcessing_aligmentMethod_comboBox, ["RANSAC", "FGR", "HYBRID"], "RANSAC")

        self.cloudsProcessing_ransacMaxIterations_doubleSpinBox.setRange(10, 5000)
        self.cloudsProcessing_ransacMaxIterations_doubleSpinBox.setValue(config.get("cloudsProcessing_ransacMaxIterations", 1000))  # k
        self.cloudsProcessing_ransacMaxIterations_doubleSpinBox.setDecimals(0)
        self.cloudsProcessing_ransacMaxIterations_doubleSpinBox.setSingleStep(100)

        self.cloudsProcessing_ransacConfidenceLevel_doubleSpinBox.setRange(0.8, 0.999)
        self.cloudsProcessing_ransacConfidenceLevel_doubleSpinBox.setValue(config.get("cloudsProcessing_ransacConfidenceLevel", 0.999))
        self.cloudsProcessing_ransacConfidenceLevel_doubleSpinBox.setDecimals(3)
        self.cloudsProcessing_ransacConfidenceLevel_doubleSpinBox.setSingleStep(0.001)

        self.cloudsProcessing_ransacRandomSamples_doubleSpinBox.setRange(5, 20)
        self.cloudsProcessing_ransacRandomSamples_doubleSpinBox.setValue(config.get("cloudsProcessing_ransacRandomSamples", 7))
        self.cloudsProcessing_ransacRandomSamples_doubleSpinBox.setDecimals(0)
        self.cloudsProcessing_ransacRandomSamples_doubleSpinBox.setSingleStep(1)

        self.cloudsProcessing_fgrMaxIterations_doubleSpinBox.setRange(30, 200)
        self.cloudsProcessing_fgrMaxIterations_doubleSpinBox.setValue(config.get("cloudsProcessing_fgrMaxIterations", 64))
        self.cloudsProcessing_fgrMaxIterations_doubleSpinBox.setDecimals(0)
        self.cloudsProcessing_fgrMaxIterations_doubleSpinBox.setSingleStep(10)

        self.cloudsProcessing_fgrDivisionFactor_doubleSpinBox.setRange(1.2, 4.0)
        self.cloudsProcessing_fgrDivisionFactor_doubleSpinBox.setValue(config.get("cloudsProcessing_fgrDivisionFactor", 1.4))
        self.cloudsProcessing_fgrDivisionFactor_doubleSpinBox.setDecimals(1)
        self.cloudsProcessing_fgrDivisionFactor_doubleSpinBox.setSingleStep(0.1)

        self.cloudsProcessing_edgeLengthVerificationCheck_checkBox.setChecked(config.get("cloudsProcessing_edgeLengthVerificationCheck", True))
        self.cloudsProcessing_distanceVerificationCheck_checkBox.setChecked(config.get("cloudsProcessing_distanceVerificationCheck", True))
        self.cloudsProcessing_normalsVerificationCheck_checkBox.setChecked(config.get("cloudsProcessing_normalsVerificationCheck", True))

        init_combobox(self.cloudsProcessing_icpMethod_comboBox, ["P2Point", "P2Plane"], "P2Plane")

        self.cloudsProcessing_useCoarseIcpRegistration_checkBox.setChecked(config.get("cloudsProcessing_useCoarseIcpRegistration", True))
        self.cloudsProcessing_coarseIcpMaxIterations_doubleSpinBox.setRange(30, 100)
        self.cloudsProcessing_coarseIcpMaxIterations_doubleSpinBox.setValue(config.get("cloudsProcessing_coarseIcpMaxIterations", 60))
        self.cloudsProcessing_coarseIcpMaxIterations_doubleSpinBox.setDecimals(0)
        self.cloudsProcessing_coarseIcpMaxIterations_doubleSpinBox.setSingleStep(5)

        self.cloudsProcessing_useAbsoluteValues_checkBox_2.setChecked(config.get("cloudsProcessing_useAbsoluteValues_coarse", False))
        self.cloudsProcessing_coarseThresholdAbsolute_doubleSpinBox.setRange(1, 50)
        self.cloudsProcessing_coarseThresholdAbsolute_doubleSpinBox.setValue(config.get("cloudsProcessing_coarseThresholdAbsolute", 1))
        self.cloudsProcessing_coarseThresholdAbsolute_doubleSpinBox.setDecimals(0)
        self.cloudsProcessing_coarseThresholdAbsolute_doubleSpinBox.setSingleStep(1)

        self.cloudsProcessing_coarseThresholdMultiplier_doubleSpinBox.setRange(3.0, 6.0)
        self.cloudsProcessing_coarseThresholdMultiplier_doubleSpinBox.setValue(config.get("cloudsProcessing_coarseThresholdMultiplier", 3.0))
        self.cloudsProcessing_coarseThresholdMultiplier_doubleSpinBox.setDecimals(1)
        self.cloudsProcessing_coarseThresholdMultiplier_doubleSpinBox.setSingleStep(0.1)

        self.cloudsProcessing_coarseIcpRelativeFitness_doubleSpinBox.setRange(0.1, 1.0)
        self.cloudsProcessing_coarseIcpRelativeFitness_doubleSpinBox.setValue(config.get("cloudsProcessing_coarseIcpRelativeFitness", 0.4))
        self.cloudsProcessing_coarseIcpRelativeFitness_doubleSpinBox.setDecimals(2)
        self.cloudsProcessing_coarseIcpRelativeFitness_doubleSpinBox.setSingleStep(0.05)

        self.cloudsProcessing_coarseIcpRelativeRmse_doubleSpinBox.setRange(1, 40)
        self.cloudsProcessing_coarseIcpRelativeRmse_doubleSpinBox.setValue(config.get("cloudsProcessing_coarseIcpRelativeRmse", 5))
        self.cloudsProcessing_coarseIcpRelativeRmse_doubleSpinBox.setDecimals(0)
        self.cloudsProcessing_coarseIcpRelativeRmse_doubleSpinBox.setSingleStep(1)

        self.cloudsProcessing_useFineIcpRegistration_checkBox.setChecked(config.get("cloudsProcessing_useFineIcpRegistration", True))

        self.cloudsProcessing_fineIcpMaxIterations_doubleSpinBox.setRange(20, 50)
        self.cloudsProcessing_fineIcpMaxIterations_doubleSpinBox.setValue(config.get("cloudsProcessing_fineIcpMaxIterations", 30))
        self.cloudsProcessing_fineIcpMaxIterations_doubleSpinBox.setDecimals(0)
        self.cloudsProcessing_fineIcpMaxIterations_doubleSpinBox.setSingleStep(1)

        self.cloudsProcessing_useAbsoluteValues_checkBox.setChecked(config.get("cloudsProcessing_useAbsoluteValues", True))

        self.cloudsProcessing_coarseThresholdAbsolute_doubleSpinBox_2.setRange(0.1, 5)
        self.cloudsProcessing_coarseThresholdAbsolute_doubleSpinBox_2.setValue(config.get("cloudsProcessing_fineThresholdAbsolute", 0.5))
        self.cloudsProcessing_coarseThresholdAbsolute_doubleSpinBox_2.setDecimals(1)
        self.cloudsProcessing_coarseThresholdAbsolute_doubleSpinBox_2.setSingleStep(0.1)

        self.cloudsProcessing_fineThreshold_doubleSpinBox.setRange(1, 5)
        self.cloudsProcessing_fineThreshold_doubleSpinBox.setValue(config.get("cloudsProcessing_fineThresholdMultiplier", 1))
        self.cloudsProcessing_fineThreshold_doubleSpinBox.setDecimals(1)
        self.cloudsProcessing_fineThreshold_doubleSpinBox.setSingleStep(0.1)

        self.cloudsProcessing_fineIcpRelativeFitness_doubleSpinBox.setRange(0.1, 1)
        self.cloudsProcessing_fineIcpRelativeFitness_doubleSpinBox.setValue(config.get("cloudsProcessing_fineIcpRelativeFitness", 0.9))
        self.cloudsProcessing_fineIcpRelativeFitness_doubleSpinBox.setDecimals(2)
        self.cloudsProcessing_fineIcpRelativeFitness_doubleSpinBox.setSingleStep(0.05)

        self.cloudsProcessing_fineIcpRelativeRmse_doubleSpinBox.setRange(0.5, 3.0)
        self.cloudsProcessing_fineIcpRelativeRmse_doubleSpinBox.setValue(config.get("cloudsProcessing_fineIcpRelativeRmse", 1.0))
        self.cloudsProcessing_fineIcpRelativeRmse_doubleSpinBox.setDecimals(2)
        self.cloudsProcessing_fineIcpRelativeRmse_doubleSpinBox.setSingleStep(0.1)

        self.cloudsProcessing_useColorIcp_checkBox.setChecked(config.get("cloudsProcessing_useColorIcp", True))
        self.cloudsProcessing_shapeImportance_doubleSpinBox.setRange(0.5, 1.0)
        self.cloudsProcessing_shapeImportance_doubleSpinBox.setValue(config.get("cloudsProcessing_shapeImportance", 0.9))
        self.cloudsProcessing_shapeImportance_doubleSpinBox.setDecimals(2)
        self.cloudsProcessing_shapeImportance_doubleSpinBox.setSingleStep(0.05)

        self.cloudsProcessing_colorImportance_doubleSpinBox.setRange(0.0, 0.5)
        self.cloudsProcessing_colorImportance_doubleSpinBox.setValue(config.get("cloudsProcessing_colorImportance", 0.1))
        self.cloudsProcessing_colorImportance_doubleSpinBox.setDecimals(2)
        self.cloudsProcessing_colorImportance_doubleSpinBox.setSingleStep(0.01)

        self.cloudsProcessing_postprocessingVoxelSize_doubleSpinBox.setRange(0.0, 4.0)
        self.cloudsProcessing_postprocessingVoxelSize_doubleSpinBox.setValue(config.get("cloudsProcessing_postprocessingVoxelSize", 1.0))
        self.cloudsProcessing_postprocessingVoxelSize_doubleSpinBox.setDecimals(2)
        self.cloudsProcessing_postprocessingVoxelSize_doubleSpinBox.setSingleStep(0.1)

        self.cloudsProcessing_fillHoles_checkBox.setChecked(config.get("cloudsProcessing_fillHoles", True))
        self.cloudsProcessing_postprocessingHoleSize_doubleSpinBox.setRange(0.01, 0.1)
        self.cloudsProcessing_postprocessingHoleSize_doubleSpinBox.setValue(config.get("cloudsProcessing_postprocessingHoleSize", 0.05))
        self.cloudsProcessing_postprocessingHoleSize_doubleSpinBox.setDecimals(3)
        self.cloudsProcessing_postprocessingHoleSize_doubleSpinBox.setSingleStep(0.005)

        self.cloudsProcessing_filtering_checkBox.setChecked(config.get("cloudsProcessing_filtering", True))
        self.cloudsProcessing_minArtifactSize_doubleSpinBox.setRange(100, 500)
        self.cloudsProcessing_minArtifactSize_doubleSpinBox.setValue(config.get("cloudsProcessing_minArtifactSize", 100))
        self.cloudsProcessing_minArtifactSize_doubleSpinBox.setDecimals(0)
        self.cloudsProcessing_minArtifactSize_doubleSpinBox.setSingleStep(50)

        self.cloudsProcessing_filteringArtifactSensitivity_doubleSpinBox.setRange(0.01, 0.05)
        self.cloudsProcessing_filteringArtifactSensitivity_doubleSpinBox.setValue(config.get("cloudsProcessing_filtrArtifactSensitivity", 0.02))
        self.cloudsProcessing_filteringArtifactSensitivity_doubleSpinBox.setDecimals(3)
        self.cloudsProcessing_filteringArtifactSensitivity_doubleSpinBox.setSingleStep(0.005)

        self.cloudsProcessing_keepLargest_checkBox.setChecked(config.get("cloudsProcessing_keepLargest", True))

        init_combobox(self.cloudsProcessing_smoothingMethod_comboBox, ["none", "mls", "laplasian"], "none")
        self.cloudsProcessing_smoothingRadius_doubleSpinBox.setRange(0.01, 0.05)
        self.cloudsProcessing_smoothingRadius_doubleSpinBox.setValue(config.get("cloudsProcessing_smoothingRadius", 0.03))
        self.cloudsProcessing_smoothingRadius_doubleSpinBox.setDecimals(3)
        self.cloudsProcessing_smoothingRadius_doubleSpinBox.setSingleStep(0.005)

        self.cloudsProcessing_smoothingIterations_doubleSpinBox.setRange(1, 5)
        self.cloudsProcessing_smoothingIterations_doubleSpinBox.setValue(config.get("cloudsProcessing_smoothingIterations", 2))
        self.cloudsProcessing_smoothingIterations_doubleSpinBox.setDecimals(0)
        self.cloudsProcessing_smoothingIterations_doubleSpinBox.setSingleStep(1)

        self.cloudsProcessing_edgePreserve_checkBox.setChecked(config.get("cloudsProcessing_edgePreserve", True))

        self.cloudsProcessing_smoothingTaubinLambda_doubleSpinBox.setRange(0.1, 0.5)
        self.cloudsProcessing_smoothingTaubinLambda_doubleSpinBox.setValue(config.get("cloudsProcessing_smoothingTaubinLambda", 0.3))
        self.cloudsProcessing_smoothingTaubinLambda_doubleSpinBox.setDecimals(2)
        self.cloudsProcessing_smoothingTaubinLambda_doubleSpinBox.setSingleStep(0.05)

        self.cloudsProcessing_smoothingTaubinMu_doubleSpinBox.setRange(-0.5, 0.0)
        self.cloudsProcessing_smoothingTaubinMu_doubleSpinBox.setValue(config.get("cloudsProcessing_smoothingTaubinMu", -0.3))
        self.cloudsProcessing_smoothingTaubinMu_doubleSpinBox.setDecimals(2)
        self.cloudsProcessing_smoothingTaubinMu_doubleSpinBox.setSingleStep(0.05)

        self.cloudsProcessing_optimization_checkBox.setChecked(config.get("cloudsProcessing_optimization", True))
        init_combobox(self.cloudsProcessing_optimizationAlgorithm_comboBox, ["L-M", "G_N", "Levenberg"], "L-M")
        self.cloudsProcessing_useLoopClosure_checkBox.setChecked(config.get("cloudsProcessing_useLoopClosure", True))

        self.cloudsProcessing_minOverlapPoints_doubleSpinBox.setRange(100, 1000)
        self.cloudsProcessing_minOverlapPoints_doubleSpinBox.setValue(config.get("cloudsProcessing_minOverlapPoints", 500))
        self.cloudsProcessing_minOverlapPoints_doubleSpinBox.setDecimals(0)
        self.cloudsProcessing_minOverlapPoints_doubleSpinBox.setSingleStep(100)

        self.cloudsProcessing_edgePruneThreshold_doubleSpinBox.setRange(0.1, 0.5)
        self.cloudsProcessing_edgePruneThreshold_doubleSpinBox.setValue(config.get("cloudsProcessing_edgePruneThreshold", 0.25))
        self.cloudsProcessing_edgePruneThreshold_doubleSpinBox.setDecimals(2)
        self.cloudsProcessing_edgePruneThreshold_doubleSpinBox.setSingleStep(0.01)

        self.cloudsProcessing_useGeodesic_checkBox.setChecked(config.get("cloudsProcessing_useGeodesic", True))

        self.cloudsProcessing_exportReconstructionMethod_comboBox.clear()
        self.cloudsProcessing_exportReconstructionMethod_comboBox.addItems([
            "Ball-Pivot", "Poisson", "Alpha-Shapes", "Delaunay"
        ])
        self.cloudsProcessing_exportReconstructionMethod_comboBox.setCurrentText("Poisson")

        self.cloudsProcessing_exportMeshResolution_comboBox.clear()
        self.cloudsProcessing_exportMeshResolution_comboBox.addItems([
            "Low", "Average", "High"
        ])
        self.cloudsProcessing_exportMeshResolution_comboBox.setCurrentText("Average")

        self.cloudsProcessing_exportFinalSmoothing_comboBox.clear()
        self.cloudsProcessing_exportFinalSmoothing_comboBox.addItems([
            "None", "Laplacian", "Taubin"
        ])
        self.cloudsProcessing_exportFinalSmoothing_comboBox.setCurrentText("None")
        
        self.cloudsProcessing_saveFormat_comboBox.addItems([
            ".PLY", ".STL", ".OBJ"
        ])


    def set_connections_cloudsprocessing_tab(self):
        self.cloudsProcessing_open3dSpace_pushButton.clicked.connect(self.open_3dspace)
        self.cloudsProcessing_chooseFilesNUpload_pushButton.clicked.connect(self.choose_n_upload3ds)

        self.cloudsProcessing_cloudsList_tableWidget.itemSelectionChanged.connect(self.on_cloud_table_selection_changed)

        self.cloudsProcessing_manualAdjustmentShiftX_horizontalSlider.valueChanged.connect(self.manual_position_correction)
        self.cloudsProcessing_manualAdjustmentShiftY_horizontalSlider.valueChanged.connect(self.manual_position_correction)
        self.cloudsProcessing_manualAdjustmentShiftZ_horizontalSlider.valueChanged.connect(self.manual_position_correction)
        self.cloudsProcessing_manualAdjustmentRotShiftXY_horizontalSlider.valueChanged.connect(self.manual_position_correction)
        self.cloudsProcessing_manualAdjustmentRotShiftXZ_horizontalSlider.valueChanged.connect(self.manual_position_correction)
        self.cloudsProcessing_manualAdjustmentRotShiftYZ_horizontalSlider.valueChanged.connect(self.manual_position_correction)

        self.cloudsProcessing_manualAdjustmentIsEnabled_checkBox.setChecked(True)
        self.cloudsProcessing_manualAdjustmentIsEnabled_checkBox.toggled.connect(self.set_manual_correction_mode)

        self.cloudsProcessing_applyNProcess_pushButton.clicked.connect(self.applyNProcess)

    # Обработка закрытия    
    def closeEvent(self, event):
        #openni2.unload()

        config = self.load_config()
        self.save_platform_tab(config)
        self.save_cam_check_tab(config)
        self.save_scanning_tab(config)
        self.save_framesprocessing_tab(config)
        self.save_cloudsprocessing_tab(config)
        self.save_config(config)

        if self.td_sock:
            self.send_td_shutdown()

        event.accept()

    def save_config(self, config):
        with open(self.CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)

    def save_platform_tab(self, config):
        config["esp_connect_tab"] = {
            "platformPort": self.platformConnectionSettings_platformPort_lineEdit.text(),
            "platformIP": self.platformConnectionSettings_platformIP_lineEdit.text(),
            "apSSID": self.platformConnectionSettings_ApSSID_lineEdit.text(),
            "apPassword": self.platformConnectionSettings_APPAssword_lineEdit.text(),
        }

    def save_cam_check_tab(self, config):
        config["cam_check_tab"] = {
            "resolutionX": self.cameraCheck_resolutionX_spinBox.value(),
            "resolutionY": self.cameraCheck_resolutionY_spinBox.value(),
            "folderName": self.cameraCheck_chooseFolder_lineEdit.text(),
            "objectName": self.cameraCheck_photoName_lineEdit.text()
        }

    def save_scanning_tab(self, config):
        config["scanning_tab"] = {
            "folderName": self.scanningSettings_chooseFolder_lineEdit.text(),
            "objectName": self.scanningSettings_objectName_lineEdit.text(),
            "folderName_3ds": self.framesProcessing_folderToSave_lineEdit.text(),

            "speed": self.scanningSettings_speed_spinBox.value(),
            "acceleration": self.scanningSettings_acceeration_spinBox.value(),
            "numberOfShots": self.scanningSettings_numverOfShots_spinBox.value(),
            "rotationAngle": self.scanningSettings_rotationAngle_spinBox.value()
        }     

    def save_framesprocessing_tab(self, config):
        config["framesprocessing_tab"] = {
            "folderName": self.framesProcessing_folder_lineEdit.text(),
            "objectName": self.framesProcessing_objectName_lineEdit.text(),
            "folderName_3ds": self.framesProcessing_folderToSave_lineEdit.text(),

            "depthTrunk": self.framesProcessing_depthTrunc_doubleSpinBox.value(),
            "kernelSize": self.framesProcessing_morphKernelSize_doubleSpinBox.value(),

            "shift_x": self.framesProcessing_shiftX_doubleSpinBox.value(),
            "shift_y": self.framesProcessing_shiftY_doubleSpinBox.value(),
            "stretch_x": self.framesProcessing_stretchX_doubleSpinBox.value(),
            "stretch_y": self.framesProcessing_stretchY_doubleSpinBox.value(),

            "ref_angle": self.framesProcessing_photoReferenceAngle_doubleSpinBox.value(),

            "depth_width": self.framesProcessing_depthCameraIntrinsics_width_doubleSpinBox.value(),
            "depth_height": self.framesProcessing_depthCameraIntrinsics_heigth_doubleSpinBox.value(),
            "depth_fx": self.framesProcessing_depthCameraIntrinsics_fx_doubleSpinBox.value(),
            "depth_fy": self.framesProcessing_depthCameraIntrinsics_fy_doubleSpinBox.value(),
            "depth_cx": self.framesProcessing_depthCameraIntrinsics_cx_doubleSpinBox.value(),
            "depth_cy": self.framesProcessing_depthCameraIntrinsics_cy_doubleSpinBox.value(),

            "rgb_width": self.framesProcessing_RGBCameraIntrinsics_width_doubleSpinBox.value(),
            "rgb_height": self.framesProcessing_RGBCameraIntrinsics_height_doubleSpinBox.value(),
            "rgb_fx": self.framesProcessing_RGBCameraIntrinsics_fx_doubleSpinBox.value(),
            "rgb_fy": self.framesProcessing_RGBCameraIntrinsics_fy_doubleSpinBox.value(),
            "rgb_cx": self.framesProcessing_RGBCameraIntrinsics_cx_doubleSpinBox.value(),
            "rgb_cy": self.framesProcessing_RGBCameraIntrinsics_cy_doubleSpinBox.value(),

            "numberOfClouds": self.framesProcessing_numberOfClouds_doubleSpinBox.value()
        }  

    def save_cloudsprocessing_tab(self, config):
        config["cloudsprocessing_tab"] = {
            "last_tddir": self.last_td_dir,

            "cloudsProcessing_uploadFiltration": self.cloudsProcessing_uploadFiltration_checkBox.isChecked(),
            "cloudsProcessing_filtrationNeighbors": self.cloudsProcessing_filtrationNeighbors_doubleSpinBox.value(),
            "cloudsProcessing_outlierSensitivity": self.cloudsProcessing_outlierSensitivity_doubleSpinBox.value(),
            "cloudsProcessing_enableDistanceFilter": self.cloudsProcessing_enableDistanceFilter_checkBox.isChecked(),
            "cloudsProcessing_uploadFilterMaxDistance": self.cloudsProcessing_uploadFilterMaxDistance_doubleSpinBox.value(),
            "cloudsProcessing_uploadFilterMinDistance": self.cloudsProcessing_uploadFilterMinDistance_doubleSpinBox.value(),
            "cloudsProcessing_preloadShift": self.cloudsProcessing_preloadShift_checkBox.isChecked(),

            "preload_shift_x": self.cloudsProcessing_preloadAxisShiftX_doubleSpinBox.value(),
            "preload_shift_y": self.cloudsProcessing_preloadAxisShiftY_doubleSpinBox.value(),
            "preload_shift_z": self.cloudsProcessing_preloadAxisShiftZ_doubleSpinBox.value(),

            "preload_rot_xy": self.cloudsProcessing_preloadRotationShiftXY_doubleSpinBox.value(),
            "preload_rot_xz": self.cloudsProcessing_preloadRotationShiftXZ_doubleSpinBox.value(),
            "preload_rot_yz": self.cloudsProcessing_preloadRotationShiftYZ_doubleSpinBox.value(), 


            "downsamplingMode_comboBox": self.cloudsProcessing_downsamplingMode_comboBox.currentText(),
            "downsampling_vovelSize_doubleSpinBox": self.cloudsProcessing_downsampling_vovelSize_doubleSpinBox.value(),
            "downsamplingMinPntsPerVoxel_spinBox": self.cloudsProcessing_downsamplingMinPointsPerVoxel_doubleSpinBox.value(),
            "normalsCalculation_checkBox": self.cloudsProcessing_normalsCalculation_checkBox.isChecked(),
            "normalEstimationRadius_doubleSpinBox": self.cloudsProcessing_normalEstimationRadius_doubleSpinBox.value(),

            "globalRegistrationIsEnabled_checkBox": self.cloudsProcessing_globalRegistrationIsEnabled_checkBox.isChecked(),
            "featureRadiusMultipliers_lineEdit": self.cloudsProcessing_featureRadiusMultipliers_lineEdit.text(),
            "aligmentMethod_comboBox": self.cloudsProcessing_aligmentMethod_comboBox.currentText(),
            "ransacMaxIterations_spinBox": self.cloudsProcessing_ransacMaxIterations_doubleSpinBox.value(),
            "ransacConfidenceLevel_doubleSpinBox": self.cloudsProcessing_ransacConfidenceLevel_doubleSpinBox.value(),
            "ransacRandomSamples_spinBox": self.cloudsProcessing_ransacRandomSamples_doubleSpinBox.value(),
            "fgrMaxIterations_spinBox": self.cloudsProcessing_fgrMaxIterations_doubleSpinBox.value(),
            "fgrDivisionFactor_doubleSpinBox": self.cloudsProcessing_fgrDivisionFactor_doubleSpinBox.value(),
            "edgeLengthVerificationCheck_checkBox": self.cloudsProcessing_edgeLengthVerificationCheck_checkBox.isChecked(),
            "distanceVerificationCheck_checkBox": self.cloudsProcessing_distanceVerificationCheck_checkBox.isChecked(),
            "normalsVerificationCheck_checkBox": self.cloudsProcessing_normalsVerificationCheck_checkBox.isChecked(),

            "icpMethod_comboBox": self.cloudsProcessing_icpMethod_comboBox.currentText(),
            "useCoarseIcpRegistration_checkBox": self.cloudsProcessing_useCoarseIcpRegistration_checkBox.isChecked(),
            "coarseIcpMaxIterations_spinBox": self.cloudsProcessing_coarseIcpMaxIterations_doubleSpinBox.value(),
            "useAbsoluteValues_coarse_checkBox": self.cloudsProcessing_useAbsoluteValues_checkBox_2.isChecked(),
            "coarseThresholdAbsolute_spinBox": self.cloudsProcessing_coarseThresholdAbsolute_doubleSpinBox.value(),
            "coarseThresholdMultiplier_doubleSpinBox": self.cloudsProcessing_coarseThresholdMultiplier_doubleSpinBox.value(),
            "coarseIcpRelativeFitness_doubleSpinBox": self.cloudsProcessing_coarseIcpRelativeFitness_doubleSpinBox.value(),
            "coarseIcpRelativeRmse_spinBox": self.cloudsProcessing_coarseIcpRelativeRmse_doubleSpinBox.value(),
            "useFineIcpRegistration_checkBox": self.cloudsProcessing_useFineIcpRegistration_checkBox.isChecked(),
            "fineIcpMaxIterations_spinBox": self.cloudsProcessing_fineIcpMaxIterations_doubleSpinBox.value(),
            "useAbsoluteValues_checkBox": self.cloudsProcessing_useAbsoluteValues_checkBox.isChecked(),
            "fineThresholdAbsolute_doubleSpinBox": self.cloudsProcessing_fineThreshold_doubleSpinBox.value(),
            "fineThresholdMultiplier_spinBox": self.cloudsProcessing_coarseThresholdMultiplier_doubleSpinBox.value(),
            "fineIcpRelativeFitness_doubleSpinBox": self.cloudsProcessing_fineIcpRelativeFitness_doubleSpinBox.value(),
            "fineIcpRelativeRmse_doubleSpinBox": self.cloudsProcessing_fineIcpRelativeRmse_doubleSpinBox.value(),
            "useColorIcp_checkBox": self.cloudsProcessing_useColorIcp_checkBox.isChecked(),
            "shapeImportance_doubleSpinBox": self.cloudsProcessing_shapeImportance_doubleSpinBox.value(),
            "colorImportance_doubleSpinBox": self.cloudsProcessing_colorImportance_doubleSpinBox.value(),
            
            "postprocessingVoxelSize_doubleSpinBox": self.cloudsProcessing_postprocessingVoxelSize_doubleSpinBox.value(),
            "fillHoles_checkBox": self.cloudsProcessing_fillHoles_checkBox.isChecked(),
            "postprocessingHoleSize_doubleSpinBox": self.cloudsProcessing_postprocessingHoleSize_doubleSpinBox.value(),
            "filtering_checkBox": self.cloudsProcessing_filtering_checkBox.isChecked(),
            "minArtifactSize_spinBox": self.cloudsProcessing_minArtifactSize_doubleSpinBox.value(),
            "filtrArtifactSensitivity_doubleSpinBox": self.cloudsProcessing_filteringArtifactSensitivity_doubleSpinBox.value(),
            "keepLargest_checkBox": self.cloudsProcessing_keepLargest_checkBox.isChecked(),
            "smoothingMethod_comboBox": self.cloudsProcessing_smoothingMethod_comboBox.currentText(),
            "smoothingRadius_doubleSpinBox": self.cloudsProcessing_smoothingRadius_doubleSpinBox.value(),
            "smoothingIterations_doubleSpinBox": self.cloudsProcessing_smoothingIterations_doubleSpinBox.value(),
            "edgePreserve_checkBox": self.cloudsProcessing_edgePreserve_checkBox.isChecked(),
            "smoothingTaubinLambda_doubleSpinBox": self.cloudsProcessing_smoothingTaubinLambda_doubleSpinBox.value(),
            "smoothingTaubinMu_doubleSpinBox": self.cloudsProcessing_smoothingTaubinMu_doubleSpinBox.value(),
            "optimization_checkBox": self.cloudsProcessing_optimization_checkBox.isChecked(),
            "optimizationAlgorithm_comboBox": self.cloudsProcessing_optimizationAlgorithm_comboBox.currentText(),
            "useLoopClosure_checkBox": self.cloudsProcessing_useLoopClosure_checkBox.isChecked(),
            "minOverlapPoints_spinBox": self.cloudsProcessing_minOverlapPoints_doubleSpinBox.value(),
            "edgePruneThreshold_doubleSpinBox": self.cloudsProcessing_edgePruneThreshold_doubleSpinBox.value(),
            "useGeodesic_checkBox": self.cloudsProcessing_useGeodesic_checkBox.isChecked(),

            "exportReconstructionMethod_comboBox": self.cloudsProcessing_exportReconstructionMethod_comboBox.currentText(),
            "exportMeshResolution_comboBox": self.cloudsProcessing_exportMeshResolution_comboBox.currentText(),
            "exportFinalSmoothing_comboBox": self.cloudsProcessing_exportFinalSmoothing_comboBox.currentText()
        }

    def sync_folder_lineedits(self, text):
        if self.shots_folder_line_syncing:
            return
        self.shots_folder_line_syncing = True
        self.cameraCheck_chooseFolder_lineEdit.setText(text)
        self.scanningSettings_chooseFolder_lineEdit.setText(text)
        self.framesProcessing_folder_lineEdit.setText(text)
        self.shots_folder_line_syncing = False

    def sync_object_lineedits(self, text):
        if self.shots_object_line_syncing:
            return
        self.shots_folder_line_syncing = True
        self.scanningSettings_objectName_lineEdit.setText(text)
        self.framesProcessing_objectName_lineEdit.setText(text)
        self.shots_object_line_syncing = False

    def choose_object_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory", "")
        if dir_path:
            self.cameraCheck_chooseFolder_lineEdit.setText(dir_path)
            self.scanningSettings_chooseFolder_lineEdit.setText(dir_path)
            self.framesProcessing_folder_lineEdit.setText(dir_path)

    # ---------------------------------------------------------------------------------------------------------
    # -- ESP CONNECTION ---------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------

    def connect_to_platform(self):
        if self.esp_connected:
            self.disconnect_from_platform()
            return

        ip = self.platformConnectionSettings_platformIP_lineEdit.text().strip()
        port_text = self.platformConnectionSettings_platformPort_lineEdit.text().strip()

        if not ip or not port_text:
            QMessageBox.warning(self, "Input Error", "Enter IP and Port")
            return

        try:
            port = int(port_text)
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Port must be integer")
            return

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((ip, port))

            self.esp_sock = sock
            self.esp_connected = True

            self.platformConnectionSettings_connect_pushButton.setText("Disconnect")
            self.platformConnectionSettings_platformStatus_label.setText("Platform Status: Connected")
            QMessageBox.information(self, "Connected", "Connected to Platform successfully.")
            print("[GUI] Connected to Platform")

            self.esp_listener_thread = threading.Thread(target=self.listen_to_platform, daemon=True)
            self.esp_listener_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Connection Error", f"Failed to connect:\n{e}")
            self.esp_sock = None
            self.esp_connected = False
            print(f"[GUI] Failed to connect: {e}")

    def disconnect_from_platform(self):
        if self.esp_sock:
            try:
                self.esp_sock.close()
            except:
                pass
        self.esp_sock = None
        self.esp_connected = False
        self.esp_listener_thread = None

        self.platformConnectionSettings_connect_pushButton.setText("Connect")
        self.platformConnectionSettings_platformStatus_label.setText("Platform Status: Disconnected")
        print("[GUI] Disconnected from Platform")
        # QMessageBox.information(self, "Disconnected", "Disconnected from Platform.")

    # def listen_to_platform(self):
    #     try:
    #         while self.platform_connected:
    #             data = self.platform_socket.recv(1024)
    #             if not data:
    #                 break
    #             message = data.decode().strip()
    #             print(f"[Platform] {message}")
    #     except Exception as e:
    #         print(f"[GUI] Listen error: {e}")
    #     finally:
    #         self.disconnect_from_platform()

    def listen_to_platform(self):
        try:
            self.esp_sock.settimeout(None)
            while self.esp_connected:
                data = self.recv_exact(self.esp_sock, 3)
                if data is None:
                    break

                if len(data) != 3:
                    print(f"[Platform] Unexpected data length: {len(data)}")
                    continue

                code = data[0]
                value = int.from_bytes(data[1:3], 'little')

                if code == 0x01:
                    self.scanning_process_rotate_num = value
                    self.scanning_process_rotate_maden = True
                    print(f"[Platform] Сделан поворот № {value}")
                elif code == 0x02:
                    print("[Platform] Процесс сканирования прерван")
                elif code == 0x03:
                    print("[Platform] Данные сети сохранены")
                else:
                    print(f"[Platform] Неизвестный код ответа: {code} значение: {value}")
        except Exception as e:
            print(f"[GUI] Listen error: {e}")
        finally:
            # self.disconnect_from_platform()
             self.esp_disconnected_signal.emit()

    def recv_exact(self, sock, size):
        data = bytearray()
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)


    # def send_to_platform(self, cmd):
    #     if not self.esp_connected or self.esp_sock is None:
    #         QMessageBox.warning(self, "Send Error", "Not connected to platform")
    #         return
    #     try:
    #         self.esp_sock.sendall((cmd + "\n").encode())
    #         print(f"[GUI] Sent: {cmd}")
    #     except Exception as e:
    #         print(f"[GUI] Send failed: {e}")
    #         QMessageBox.warning(self, "Send Error", f"Failed to send command:\n{e}")

    def send_packet(self, packet: bytes):
        if not self.esp_connected or self.esp_sock is None:
            QMessageBox.warning(self, "Send Error", "Not connected to platform")
            return
        
        try:
            self.esp_sock.sendall(packet)
            print(f"[GUI] Sent: {packet.hex()}")
        except Exception as e:
            print(f"[GUI] Send failed: {e}")
            QMessageBox.warning(self, "Send Error", f"Failed to send command:\n{e}")

    def set_new_user_ap_data(self):
        self.send_start_scan(3, 90, 10, 5)

    def send_start_scan(self, turns, angle, speed, acceleration):
        packet = bytearray(65)
        packet[0] = 0x01
        packet[1:3] = turns.to_bytes(2, 'little')
        packet[17:19] = angle.to_bytes(2, 'little')
        packet[33:35] = speed.to_bytes(2, 'little')
        packet[49:51] = acceleration.to_bytes(2, 'little')
        self.send_packet(packet)

    def send_continue_scan(self):
        packet = bytearray(65)
        packet[0] = 0x02
        self.send_packet(packet)

    # def send_abort_scan(self):
    #     packet = bytearray(65)
    #     packet[0] = 0x03
    #     self.send_packet(packet)

    # def send_set_wifi(self, ssid: str, password: str):
    #     packet = bytearray(65)
    #     packet[0] = 0x04
    #     ssid_bytes = ssid.encode()[:32]
    #     password_bytes = password.encode()[:32]
    #     packet[1:1 + len(ssid_bytes)] = ssid_bytes
    #     packet[33:33 + len(password_bytes)] = password_bytes
    #     self.send_packet(packet)

    # def send_reset_connection(self):
    #     packet = bytearray(65)
    #     packet[0] = 0x05
    #     self.send_packet(packet)

    # ---------------------------------------------------------------------------------------------------------
    # -- CAM CHECK FUNCS --------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------

    def cam_check_camera_outp_on_off(self):
        if self.cameraCheck_cameraOutput_checkBox.checkState() == Qt.CheckState.Checked:
            self.start_camera_worker()
            self.camera_running = True    
        else:
            self.send_shutdown()
            self.camera_running = False 

    def start_camera_worker(self):
        # Запускаем внешний процесс
        subprocess.Popen([sys.executable, "camera_worker.py"])
        threading.Thread(target=self.connect_to_worker, daemon=True).start()

    def connect_to_worker(self):
        self.camera_sock = socket.socket()
        while True:
            try:
                self.camera_sock.connect((CAM_HOST, CAM_PORT))
                print("[GUI] Conected to Camera Worker")
                break
            except Exception as e:
                pass
                # print(f"Waiting for connection... ({e})")
                # time.sleep(0.5)
        threading.Thread(target=self.listen_to_camera_worker, daemon=True).start()

    # def send_ping(self):
    #     msg = {"cmd": "ping"}
    #     self.camera_sock.sendall((json.dumps(msg) + "\n").encode())
    #     print("[GUI] ping is sended")

    def send_shutdown(self):
        msg = {"cmd": "shutdown"}
        self.camera_sock.sendall((json.dumps(msg) + "\n").encode())
        print("[GUI] shutdown is sended")

    def listen_to_camera_worker(self):
        buffer = ""
        while True:
            try:
                data = self.camera_sock.recv(1024).decode()
                if not data:
                    break
                buffer += data
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    msg = json.loads(line)
                    print("[GUI] Msg from Camera worker:", msg)

                # Получили REPLY
                reply = msg.get("reply")
                if reply == "hello":

                    if self.options_tabWidget.currentIndex() == 3:
                        if self.framesProcessing_videoMode_checkBox.checkState() == Qt.CheckState.Unchecked:
                            self.framesproc_videomode_on_off()
                            time.sleep(0.05)
                            self.framesproc_send_shift_x()
                            time.sleep(0.05)
                            self.framesproc_send_shift_y()
                            time.sleep(0.05)    
                            self.framesproc_send_stretch_x()
                            time.sleep(0.05)
                            self.framesproc_send_stretch_y()
                            time.sleep(0.05)

                            self.framesproc_send_depth_trunc()
                            time.sleep(0.05)
                            self.framesproc_send_kernel_value()
                            time.sleep(0.05)
                            self.camera_sock.sendall((json.dumps({"cmd": "no_cam_start"}) + "\n").encode())
                            print("[GUI] no_cam_start is sended")
                            continue

                    self.update_colormap(self.cameraCheck_depthDisplayMethod_comboBox.currentIndex())
                    time.sleep(0.05)
                    self.send_crop()
                    time.sleep(0.05)
                    self.send_camera_config()
                    time.sleep(0.05)

                    if self.options_tabWidget.currentIndex() == 3:
                        self.send_show_contour()
                        time.sleep(0.05)
                        self.framesproc_send_shift_x()
                        time.sleep(0.05)
                        self.framesproc_send_shift_y()
                        time.sleep(0.05)    
                        self.framesproc_send_stretch_x()
                        time.sleep(0.05)
                        self.framesproc_send_stretch_y()
                        time.sleep(0.05)

                        self.framesproc_send_depth_trunc()
                        time.sleep(0.05)
                        self.framesproc_send_kernel_value()
                        time.sleep(0.05)

                    self.camera_sock.sendall((json.dumps({"cmd": "start"}) + "\n").encode())
                    print("[GUI] start is sended") 

                if reply == "goodbye":
                    print("[GUI] Sock is now None")
                    self.camera_sock = None

                # Получили STATUS
                reply = msg.get("status")
                if reply == "good_photo":
                    depth_path = msg.get("depth_path")
                    color_path = msg.get("color_path")
                    self.made_good_photo_signal.emit(depth_path, color_path)

                # Получили ERROR
                err = msg.get("error")
                if err:
                    if "ONI_STATUS_NO_DEVICE" in err:
                        self.cameraCheck_cameraOutput_checkBox.setChecked(False)
                        self.camera_show_error_signal.emit("Device is not connected")

            except Exception as e:
                print("[GUI] Reading camera socket error:", e)
                self.camera_sock = None
                if self.camera_running:
                    self.reload_camera_signal.emit("AAAAAA") 
                    pass
                break

    def show_camera_error_message(self, message: str):
        QMessageBox.critical(self, "Camera Error", message)

    def reload_camera(self, message: str):
        print(message)
        self.start_camera_worker()

    def update_colormap(self, index):
        index = self.cameraCheck_depthDisplayMethod_comboBox.currentIndex()
        if index == -1 or self.camera_sock is None:
            return

        internal_name = self.cameraCheck_depthDisplayMethod_comboBox.itemData(index)
        msg = {"cmd": "set_colormap", "name": internal_name}
        try:
            self.camera_sock.sendall((json.dumps(msg) + "\n").encode())
            print(f"[GUI] Sent colormap change to: {internal_name}")
        except Exception as e:
            print(f"[GUI] Failed to send colormap change: {e}")

    def update_crop_width(self, value: int):
        self.send_crop()

    def update_crop_height(self, value: int):
        self.send_crop()

    def send_crop(self):
        if self.camera_sock is None:
            return

        width = self.cameraCheck_resolutionX_spinBox.value()
        height = self.cameraCheck_resolutionY_spinBox.value()

        msg = {
            "cmd": "set_crop",
            "width": width,
            "height": height
        }

        try:
            self.camera_sock.sendall((json.dumps(msg) + "\n").encode())
            # print(f"[GUI] Sent crop: {width}x{height}")
        except Exception as e:
            # print(f"[GUI] Failed to send crop command: {e}")
            pass

    def send_camera_config(self):
        if self.camera_sock is None:
            return

        config = {
            "cmd": "set_config",
            "params": {
                "registration": self.cameraCheck_depthToColor_checkBox.isChecked(),
                "sync": self.cameraCheck_syncIsEnabled_checkBox.isChecked(),
                "mirror": self.cameraCheck_mirroringIsEnables_checkBox.isChecked()
            }
        }

        try:
            self.camera_sock.sendall((json.dumps(config) + "\n").encode())
            print(f"[GUI] Sent camera config: {config['params']}")
        except Exception as e:
            print(f"[GUI] Failed to send camera config: {e}")

    def cam_check_depth_to_color_proc(self):
        # if self.cameraCheck_depthToColor_checkBox.checkState() == Qt.CheckState.Checked:
        #     msg = {"cmd": "start"}
        #     self.camera_sock.sendall((json.dumps(msg) + "\n").encode())
        #     print("[GUI] start is sended") 
        # else:
        #     msg = {"cmd": "stop"}
        #     self.camera_sock.sendall((json.dumps(msg) + "\n").encode())
        #     print("[GUI] stop is sended") 
        self.send_camera_config()

    def cam_check_sync_set(self):
        self.send_camera_config()

    def cam_check_mirror_set(self):
        self.send_camera_config()

    def make_photo(self):
        if self.camera_running == False:
            QMessageBox.warning(self, "Save Error", "Turn on the camera")
            return
        
        dir_path = self.cameraCheck_chooseFolder_lineEdit.text().strip()
        obj_name = self.cameraCheck_photoName_lineEdit.text().strip()

        if not dir_path or not obj_name:
            QMessageBox.warning(self, "Save Error", "Please specify both directory and filename.")
            return
        
        if self.camera_sock is None:
            return

        msg = {
            "cmd": "make_photo",
            "dir_path": dir_path,
            "depth_name": obj_name,  
            "color_name": obj_name,
        }

        try:
            self.camera_sock.sendall((json.dumps(msg) + "\n").encode())
            print(f"[GUI] Sent make_photo: dir='{dir_path}', name='{obj_name}'")
        except Exception as e:
            print(f"[GUI] Failed to send make_photo command: {e}")

    def made_good_photo_proc(self, depth_path, color_path):
        if self.options_tabWidget.currentIndex() == 1:
            QMessageBox.information(self, "Nice photo", f"Successfully saved:\n{depth_path}\n{color_path}")
        elif self.options_tabWidget.currentIndex() == 2:
            if self.scanning_process_is_on == False:
                QMessageBox.information(self, "Nice photo", f"Successfully saved:\n{depth_path}\n{color_path}")
            else:
                self.scanning_process_photo_maden = True

    # ---------------------------------------------------------------------------------------------------------
    # -- SCANNING FUNCS ---------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------

    def scanning_camera_outp_on_off(self):
        if self.scanningSettings_showCamOutput_checkBox.checkState() == Qt.CheckState.Checked:
            self.start_camera_worker()
            self.camera_running = True    
        else:
            self.send_shutdown()
            self.camera_running = False 

    # background_path = folder + "/" + obj_name + "_depth_background.npy"
    def make_background_photo(self):
        if self.camera_running == False:
            QMessageBox.warning(self, "Save Error", "Turn on the camera")
            return
        
        dir_path = self.scanningSettings_chooseFolder_lineEdit.text().strip()
        object_name = self.scanningSettings_objectName_lineEdit.text().strip()
        depth_obj_name = object_name + "_depth_background"
        color_obj_name = object_name + "_color_background"

        if not dir_path or not object_name:
            QMessageBox.warning(self, "Save Error", "Please specify both directory and filename.")
            return
        
        if self.camera_sock is None:
            return

        msg = {
            "cmd": "make_photo",
            "dir_path": dir_path,
            "depth_name": depth_obj_name,  
            "color_name": color_obj_name
        }

        try:
            self.camera_sock.sendall((json.dumps(msg) + "\n").encode())
            print(f"[GUI] Sent make_photo: dir='{dir_path}', name='{object_name}'")
        except Exception as e:
            print(f"[GUI] Failed to send make_photo command: {e}")

    def start_scan_proc(self):
        if self.scanning_process_is_on == False:
            if self.camera_running == False:
                QMessageBox.warning(self, "Save Error", "Turn on the camera")
                return
            
            if not self.esp_connected or self.esp_sock is None:
                QMessageBox.warning(self, "Send Error", "Not connected to platform")
                return
            
            dir_path = self.scanningSettings_chooseFolder_lineEdit.text().strip()
            object_name = self.scanningSettings_objectName_lineEdit.text().strip()

            if not dir_path or not object_name:
                QMessageBox.warning(self, "Save Error", "Please specify both directory and object name.")
                return
        
            self.scanning_process_thread = threading.Thread(target=self.scanninng_process, daemon=True)
            self.scanning_process_thread.start()
            self.scanningSettings_start_pushButton.setText("Stop")

        else: 
            self.scanning_process_is_on = False
            self.scanningSettings_start_pushButton.setText("Start")
    
    def scanninng_process(self):
        try:
            print(f"[GUI] Scanning process thread is started")

            speed = self.scanningSettings_speed_spinBox.value()
            acceleration = self.scanningSettings_acceeration_spinBox.value()
            number_of_shots = self.scanningSettings_numverOfShots_spinBox.value()
            delta_angle = self.scanningSettings_rotationAngle_spinBox.value()
            angle = 0

            self.scanning_process_photo_maden = False
            self.scanning_process_rotate_maden = False
            self.scanning_process_rotate_num = -1

            dir_path = self.scanningSettings_chooseFolder_lineEdit.text().strip()
            obj_name = self.scanningSettings_objectName_lineEdit.text().strip()

            self.scanning_process_is_on = True 

            while self.scanning_process_is_on:
                print(f"[GUI] Scanning process thread iteration with angle {angle}°")

                depth_obj_name = obj_name + "_depth_" + str(int(angle))
                color_obj_name = obj_name + "_color_" + str(int(angle))
                msg = {
                    "cmd": "make_photo",
                    "dir_path": dir_path,
                    "depth_name": depth_obj_name,  
                    "color_name": color_obj_name
                }
                self.camera_sock.sendall((json.dumps(msg) + "\n").encode())

                while self.scanning_process_photo_maden == False:
                    pass
                self.scanning_process_photo_maden = False
                
                print(f"[GUI] Made photo {angle}")
                self.scanning_made_photo_signal.emit(f"Made photo with angle {angle}°")

                number_of_shots -=1
                if number_of_shots == 0:
                    break

                # фотку сделали - теперь можно повернуть платформу
                if angle == 0:
                    self.send_start_scan(number_of_shots, delta_angle, speed, acceleration)
                else: 
                    self.send_continue_scan()

                # дождаться поворота
                while self.scanning_process_rotate_maden == False:
                    pass
                self.scanning_process_rotate_maden = False
                print(f"[GUI] Made rotate {self.scanning_process_rotate_num}")

                angle += delta_angle

            self.scanning_good_end_signal.emit("good_end")

        except Exception as e:
            print(f"[GUI] Scanning process error: {e}")
        finally:
            self.scanning_process_is_on = False
            # сигнал о завершении
        print(f"[GUI] Scanning process thread is over")

    def scanning_good_end(self, msg: str):
        print("[GUI] Entered scanning good end")
        self.scanning_process_is_on = False
        self.scanningSettings_start_pushButton.setText("Start")
        QMessageBox.information(self, "Scan", "Scanning process is ended\nAll data collected!")

    def scanning_made_photo(self, msg: str):
        self.statusBar().showMessage(msg, 3000)

    # ---------------------------------------------------------------------------------------------------------
    # -- FRAMES PROCESSING ------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------

    def frames_proc_outp_on_off(self):
        
        if self.framesProcessing_onOffImage_checkBox.checkState() == Qt.CheckState.Checked:
            self.start_camera_worker()
            self.camera_running = True 
            self.framesProcessing_videoMode_checkBox.setEnabled(False)   
        else:
            self.send_shutdown()
            self.camera_running = False 
            self.framesProcessing_videoMode_checkBox.setEnabled(True)
            

    def send_show_contour(self):
        if self.camera_sock is None:
            return
        
        folder = self.framesProcessing_folder_lineEdit.text().strip()
        obj_name = self.framesProcessing_objectName_lineEdit.text().strip()

        if not folder or not obj_name:
            QMessageBox.warning(self, "Input Error", "Folder and Object Name must not be empty.")
            return

        background_path = folder + "/" + obj_name + "_depth_background.npy"

        if os.path.exists(background_path) == False:
            QMessageBox.warning(self, "File Not Found", f"Background file not found:\n{background_path}")
            return
        
        msg = {"cmd": "set_show_contour", "value": self.framesProcessing_showCountour_checkBox.isChecked(), "dir_path": background_path}
        try:
            self.camera_sock.sendall((json.dumps(msg) + "\n").encode())
            print(f"[GUI] Sent show_contour: {msg['value']}")
        except Exception as e:
            print(f"[GUI] Failed to send show_contour: {e}")

    def frames_manual_adjustment_activation(self):
        if self.framesProcessing_manualAdjustment_checkBox.checkState() == Qt.CheckState.Checked:
            self.framesProcessing_shiftX_doubleSpinBox.setEnabled(True)
            self.framesProcessing_shiftY_doubleSpinBox.setEnabled(True)
            self.framesProcessing_stretchX_doubleSpinBox.setEnabled(True)
            self.framesProcessing_stretchY_doubleSpinBox.setEnabled(True)
        else:
            self.framesProcessing_shiftX_doubleSpinBox.setEnabled(False)
            self.framesProcessing_shiftY_doubleSpinBox.setEnabled(False)
            self.framesProcessing_stretchX_doubleSpinBox.setEnabled(False)
            self.framesProcessing_stretchY_doubleSpinBox.setEnabled(False)

    def framesproc_videomode_on_off(self):
        if self.camera_sock is None:
            return
        
        folder = self.framesProcessing_folder_lineEdit.text().strip()
        obj_name = self.framesProcessing_objectName_lineEdit.text().strip()
        angle = self.framesProcessing_photoReferenceAngle_doubleSpinBox.value()

        if not folder or not obj_name:
            QMessageBox.warning(self, "Input Error", "Folder and Object Name must not be empty.")
            return 

        background_path = folder + "/" + obj_name + "_depth_background.npy"
        object_depth_path = folder + "/" + obj_name + "_depth_" + str(int(angle)) + ".npy"
        object_color_path = folder + "/" + obj_name + "_color_" + str(int(angle)) + ".png"

        if os.path.exists(background_path) == False:
            QMessageBox.warning(self, "File Not Found", f"Background file not found:\n{background_path}")
            return
        
        if os.path.exists(object_depth_path) == False:
            QMessageBox.warning(self, "File Not Found", f"File not found:\n{object_depth_path}")
            return        
        
        if os.path.exists(object_color_path) == False:
            QMessageBox.warning(self, "File Not Found", f"File not found:\n{object_color_path}")
            return

        msg = {"cmd": "set_video_mode", "value": self.framesProcessing_videoMode_checkBox.isChecked(), "background_path": background_path, "object_depth_path": object_depth_path, "object_color_path": object_color_path}
        try:
            self.camera_sock.sendall((json.dumps(msg) + "\n").encode())
            print(f"[GUI] Sent set_video_mode: {msg['value']}")
        except Exception as e:
            print(f"[GUI] Failed to send set_video_mode: {e}")

    def framesproc_send_shift_x(self):
        if self.camera_sock is None:
            return

        shift_x = self.framesProcessing_shiftX_doubleSpinBox.value()
        msg = { "cmd": "set_frpr_shift_x", "shift_x": shift_x }

        try:
            self.camera_sock.sendall((json.dumps(msg) + "\n").encode())
            print(f"[GUI] Sent frpr shift_x: {shift_x}")
        except Exception as e:
            print(f"[GUI] Failed to send shift_x: {e}")

    def framesproc_send_shift_y(self):
        if self.camera_sock is None:
            return

        shift_y = self.framesProcessing_shiftY_doubleSpinBox.value()
        msg = {"cmd": "set_frpr_shift_y", "shift_y": shift_y}

        try:
            self.camera_sock.sendall((json.dumps(msg) + "\n").encode())
            print(f"[GUI] Sent frpr shift_y: {shift_y}")
        except Exception as e:
            print(f"[GUI] Failed to send shift_y: {e}")

    def framesproc_send_stretch_x(self):
        if self.camera_sock is None:
            return

        stretch_x = self.framesProcessing_stretchX_doubleSpinBox.value()
        msg = {"cmd": "set_frpr_stretch_x", "stretch_x": stretch_x}

        try:
            self.camera_sock.sendall((json.dumps(msg) + "\n").encode())
            print(f"[GUI] Sent frpr stretch_x: {stretch_x}")
        except Exception as e:
            print(f"[GUI] Failed to send stretch_x: {e}")

    def framesproc_send_stretch_y(self):
        if self.camera_sock is None:
            return

        stretch_y = self.framesProcessing_stretchY_doubleSpinBox.value()
        msg = {"cmd": "set_frpr_stretch_y", "stretch_y": stretch_y}

        try:
            self.camera_sock.sendall((json.dumps(msg) + "\n").encode())
            print(f"[GUI] Sent frpr stretch_y: {stretch_y}")
        except Exception as e:
            print(f"[GUI] Failed to send stretch_y: {e}")

    def framesproc_send_depth_trunc(self):
        if self.camera_sock is None:
            return

        depth_trunc = self.framesProcessing_depthTrunc_doubleSpinBox.value()
        msg = {"cmd": "set_frpr_depth_trunc", "depth_trunc": depth_trunc}

        try:
            self.camera_sock.sendall((json.dumps(msg) + "\n").encode())
            print(f"[GUI] Sent frpr depth_trunc: {depth_trunc}")
        except Exception as e:
            print(f"[GUI] Failed to send depth_trunc: {e}")

    def framesproc_send_kernel_value(self):
        if self.camera_sock is None:
            return

        kernel_size = int(self.framesProcessing_morphKernelSize_doubleSpinBox.value())

        if kernel_size < 1 or kernel_size % 2 == 0:
            print(f"[GUI] Invalid kernel size (must be odd >=1): {kernel_size}")
            return

        msg = {"cmd": "set_frpr_kernel_value", "kernel_value": kernel_size}

        try:
            self.camera_sock.sendall((json.dumps(msg) + "\n").encode())
            print(f"[GUI] Sent frpr kernel_value: {kernel_size}")
        except Exception as e:
            print(f"[GUI] Failed to send kernel_value: {e}")

    def choose_3d_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory", "")
        if dir_path:
            self.framesProcessing_folderToSave_lineEdit.setText(dir_path)

    def get_o3d_datas(self):
        self.statusBar().showMessage("Loading...", 4000)
        import cv2
        #import numpy as np
        #import open3d as o3d

        # self.statusBar().showMessage("Starting processing...", 2000)
        try:
            folder = self.framesProcessing_folder_lineEdit.text()
            obj_name = self.framesProcessing_objectName_lineEdit.text()

            use_depth_intrinsics = self.framesProcessing_depthCameraIntrinsics_checkBox.isChecked()
            use_rgb_intrinsics = self.framesProcessing_RGBCameraIntrinsics_checkBox.isChecked()

            # --- Camera Intrinsics ---
            if use_depth_intrinsics:
                depth_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                    width=int(self.framesProcessing_depthCameraIntrinsics_width_doubleSpinBox.value()),
                    height=int(self.framesProcessing_depthCameraIntrinsics_heigth_doubleSpinBox.value()),
                    fx=self.framesProcessing_depthCameraIntrinsics_fx_doubleSpinBox.value(),
                    fy=self.framesProcessing_depthCameraIntrinsics_fy_doubleSpinBox.value(),
                    cx=self.framesProcessing_depthCameraIntrinsics_cx_doubleSpinBox.value(),
                    cy=self.framesProcessing_depthCameraIntrinsics_cy_doubleSpinBox.value()
                )
            if use_rgb_intrinsics:
                rgb_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                    width=int(self.framesProcessing_RGBCameraIntrinsics_width_doubleSpinBox.value()),
                    height=int(self.framesProcessing_RGBCameraIntrinsics_height_doubleSpinBox.value()),
                    fx=self.framesProcessing_RGBCameraIntrinsics_fx_doubleSpinBox.value(),
                    fy=self.framesProcessing_RGBCameraIntrinsics_fy_doubleSpinBox.value(),
                    cx=self.framesProcessing_RGBCameraIntrinsics_cx_doubleSpinBox.value(),
                    cy=self.framesProcessing_RGBCameraIntrinsics_cy_doubleSpinBox.value()
                )

            num_clouds = int(self.framesProcessing_numberOfClouds_doubleSpinBox.value())
            depth_trunc = self.framesProcessing_depthTrunc_doubleSpinBox.value()  # в метрах
            morph_kernel_size = int(self.framesProcessing_morphKernelSize_doubleSpinBox.value())
            show_axes = self.framesProcessing_axesVisualiazation_checkBox.isChecked()
            save_folder = self.framesProcessing_folderToSave_lineEdit.text()
            os.makedirs(save_folder, exist_ok=True)

            sh_cntr_shift_x = self.framesProcessing_shiftX_doubleSpinBox.value()     # в пикселях
            sh_cntr_shift_y = self.framesProcessing_shiftY_doubleSpinBox.value()     # в пикселях
            sh_cntr_stretch_x = self.framesProcessing_stretchX_doubleSpinBox.value() # в процентах
            sh_cntr_stretch_y = self.framesProcessing_stretchY_doubleSpinBox.value() # в процентах

            # Загрузка фона
            background_path = os.path.join(folder, f"{obj_name}_depth_background.npy")
            background = np.load(background_path)

            # Список файлов
            depth_files = sorted([
                f for f in os.listdir(folder) if f.startswith(f"{obj_name}_depth_") and f.endswith(".npy") and "background" not in f
            ])
            angles = []
            for filename in depth_files:
                try:
                    angle = int(filename.split("_")[-1].replace(".npy", ""))
                    angles.append(angle)
                except ValueError:
                    continue

            kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)

            for idx, angle in enumerate(angles):
                depth = np.load(os.path.join(folder, f"{obj_name}_depth_{angle}.npy"))
                color_bgr = cv2.imread(os.path.join(folder, f"{obj_name}_color_{angle}.png"))

                h, w = color_bgr.shape[:2]
                center_x = w / 2
                center_y = h / 2

                scale_x = 1 + (sh_cntr_stretch_x / 100.0)
                scale_y = 1 + (sh_cntr_stretch_y / 100.0)

                M_scale = np.array([
                    [scale_x, 0, center_x * (1 - scale_x)],
                    [0, scale_y, center_y * (1 - scale_y)]
                ], dtype=np.float32)

                M_shift = np.array([
                    [1, 0, sh_cntr_shift_x],
                    [0, 1, sh_cntr_shift_y]
                ], dtype=np.float32)

                color_bgr = cv2.warpAffine(color_bgr, M_scale, (w, h))
                color_bgr = cv2.warpAffine(color_bgr, M_shift, (w, h))



                # # # --- Depth processing ---
                # # depth_filtered = cv2.medianBlur(depth, morph_kernel_size)
                # # depth_diff = cv2.absdiff(depth_filtered, background)
                # # _, depth_thresh = cv2.threshold(depth_diff, 50, 255, cv2.THRESH_BINARY)
                # # depth_thresh = depth_thresh.astype(np.uint8)
                # # depth_thresh = cv2.morphologyEx(depth_thresh, cv2.MORPH_OPEN, kernel)
                # # depth_thresh = cv2.morphologyEx(depth_thresh, cv2.MORPH_CLOSE, kernel)

                # # contours, _ = cv2.findContours(depth_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # # if not contours:
                # #     print(f"[GUI] [Warning] No object found at angle {angle}")
                # #     continue

                # # main_contour = max(contours, key=cv2.contourArea)
                # # object_mask = np.zeros_like(depth_thresh)
                # # cv2.drawContours(object_mask, [main_contour], -1, 255, thickness=-1)
                # # mask_bool = object_mask.astype(bool)

                # # depth_masked = np.zeros_like(depth)
                # # depth_masked[mask_bool] = depth_filtered[mask_bool]
                # # depth_masked = np.clip(depth_masked, 0, int(depth_trunc * 1000))  # обрезка глубины (в мм)

                # # color_bgr_masked = np.zeros_like(color_bgr)
                # # color_bgr_masked[mask_bool] = color_bgr[mask_bool]



                top_left_value = depth [0, 0]

                mask_top_left = (depth == top_left_value)

                depth_diff = cv2.absdiff(depth, background)

                # === БИНАРИЗАЦИЯ ===
                _, binary_mask = cv2.threshold(depth_diff, 30, 255, cv2.THRESH_BINARY)
                binary_mask = binary_mask.astype(np.uint8)

                binary_mask[depth > int(depth_trunc * 1000)] = 0
                binary_mask[depth == top_left_value] = 0

                # === Морфологическая обработка ===
                kernel = np.ones((3, 3), np.uint8)
                cleaned_mask = binary_mask.copy()
                cleaned_mask = cv2.erode(cleaned_mask, kernel, iterations=2)
                cleaned_mask = cv2.dilate(cleaned_mask, kernel, iterations=11)
                cleaned_mask = cv2.erode(cleaned_mask, kernel, iterations=3)

                # === Контур на маске ===
                contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours is None:
                    return

                largest_contour = max(contours, key=cv2.contourArea)

                color_with_contour = color_bgr.copy() 

                # cv2.drawContours(depth_with_contour, [largest_contour], -1, (0, 255, 0), 2)
                cv2.drawContours(color_with_contour, [largest_contour], -1, (0, 255, 0), 2)

                # >>> Добавить сюда >>>
                object_mask = np.zeros_like(cleaned_mask)
                cv2.drawContours(object_mask, [largest_contour], -1, 255, thickness=-1)
                mask_bool = object_mask.astype(bool)

                depth_masked = np.zeros_like(depth)
                depth_masked[mask_bool] = depth[mask_bool]
                depth_masked = np.clip(depth_masked, 0, int(depth_trunc * 1000))

                color_bgr_masked = np.zeros_like(color_bgr)
                color_bgr_masked[mask_bool] = color_bgr[mask_bool]

                # Можно визуализировать
                # cv2.imshow("Depth Masked", cv2.normalize(depth_masked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                #cv2.imshow("Color Masked", color_bgr_masked)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                color_rgb = cv2.cvtColor(color_bgr_masked, cv2.COLOR_BGR2RGB)
                depth_o3d = o3d.geometry.Image(depth_masked)
                color_o3d = o3d.geometry.Image(color_rgb)

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d, depth_o3d,
                    depth_scale=1000.0,
                    depth_trunc=depth_trunc,
                    convert_rgb_to_intensity=False
                )

                if use_depth_intrinsics and not use_rgb_intrinsics:
                    intrinsic_to_use = depth_intrinsics
                elif use_rgb_intrinsics and not use_depth_intrinsics:
                    intrinsic_to_use = rgb_intrinsics
                else:
                    intrinsic_to_use = depth_intrinsics  # default

                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_to_use)

                out_path = os.path.join(save_folder, f"{obj_name}_cloud_{angle}.ply")
                o3d.io.write_point_cloud(out_path, pcd)
                print(f"[GUI] Saved point cloud: {out_path}")

                if idx < num_clouds:
                    if show_axes:
                        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                        o3d.visualization.draw_geometries([pcd, coord_frame])
                    else:
                        o3d.visualization.draw_geometries([pcd])

                print(f"[GUI] Saved point cloud: {out_path}")

            QMessageBox.information(self, "Processing complete", "All point clouds have been processed and saved.")

        except Exception as e:
            print(f"[GUI] Error: {str(e)}")
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")

    # ---------------------------------------------------------------------------------------------------------
    # -- 3D space ---------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------

        # self.tdspace_is_opened = False
        # self.td_vis = None
        # self.td_timer = None

        # self.last_td_dir = ""

        # self.td_view_ctl = None
        # self.was_first_cloud_view = False

    def init_td_variables(self):
        self.tdspace_is_running = False
        self.tdspace_is_opened = False

        self.td_sock = None
        self.td_clouds_info = []


        self.cloudsProcessing_open3dSpace_pushButton.setEnabled(False)
        self.options_tabWidget.currentChanged.connect(self.run_3d_space)

        self.cloudsproc_deletePressed.connect(self.delete_tdrow)

        self.cloudsProcessing_saveResult_pushButton.clicked.connect(self.save_cloud)

    def run_3d_space(self):
        if self.options_tabWidget.currentIndex() == 4:
            if self.tdspace_is_running == False:
                self.start_3d_worker()
                self.tdspace_is_running = True

    def open_3dspace(self):
        if self.tdspace_is_opened == False:
            self.td_sock.sendall(json.dumps({"cmd": "create_window"}).encode() + b"\n")
            self.tdspace_is_opened = True 
            self.cloudsProcessing_open3dSpace_pushButton.setText("Close 3D space")
        else:
            # self.send_td_shutdown()
            self.td_sock.sendall(json.dumps({"cmd": "close_window"}).encode() + b"\n")
            self.tdspace_is_opened = False 
            self.cloudsProcessing_open3dSpace_pushButton.setText("Open 3D space")

    def start_3d_worker(self):
        subprocess.Popen([sys.executable, "a3d_worker.py"])
        threading.Thread(target=self.connect_to_3d_worker, daemon=True).start()

    def send_td_shutdown(self):
        msg = {"cmd": "shutdown"}
        self.td_sock.sendall((json.dumps(msg) + "\n").encode())
        print("[GUI] shutdown is sended")

    def connect_to_3d_worker(self):
        self.td_sock = socket.socket()
        while True:
            try:
                self.td_sock.connect((TD_HOST, TD_PORT))
                print("[GUI] Connected to 3D Worker")
                break
            except Exception:
                time.sleep(0.3)
        threading.Thread(target=self.listen_to_3d_worker, daemon=True).start()

    def listen_to_3d_worker(self):
        buffer = ""
        while True:
            try:
                data = self.td_sock.recv(1024).decode()
                if not data:
                    break
                buffer += data
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    msg = json.loads(line)
                    print("[GUI] Msg from 3D worker:", msg)

                reply = msg.get("reply")
                if reply == "hello":
                    # self.statusBar().showMessage("3D is ready to work", 2000)
                    self.cloudsProcessing_open3dSpace_pushButton.setEnabled(True)

                if reply == "reg_info":
                    self.cloudsProcessing_log_textBrowser.append(msg.get("registration_info"))


            except Exception as e:
                print("[GUI] 3D socket error:", e)
                self.tdspace_is_running = False
                break

    def choose_n_upload3ds(self):
        files, angles = self.select_cloud_ply_files() 

        print("[GUI] Collected files:", files)
        print("[GUI] Collected angles:", angles)

        if len(files) == 0:
            return

        shift_x = 0
        shift_y = 0
        shift_z = 0
        rot_xy = 0
        rot_xz = 0
        rot_yz = 0

        if self.cloudsProcessing_preloadShift_checkBox.isChecked():
            shift_x = float(self.cloudsProcessing_preloadAxisShiftX_doubleSpinBox.value())
            shift_y = float(self.cloudsProcessing_preloadAxisShiftY_doubleSpinBox.value())
            shift_z = float(self.cloudsProcessing_preloadAxisShiftZ_doubleSpinBox.value())
            rot_xy = float(self.cloudsProcessing_preloadRotationShiftXY_doubleSpinBox.value())
            rot_xz = float(self.cloudsProcessing_preloadRotationShiftXZ_doubleSpinBox.value())
            rot_yz = float(self.cloudsProcessing_preloadRotationShiftYZ_doubleSpinBox.value())

        new_data = []

        for file_path, angle in zip(files, angles):

            obj_name = os.path.basename(file_path).replace(".ply", "").replace("_cloud", "")
            self.add_cloud_row(cloud_name=obj_name)

            self.td_clouds_info.append({
                # "cloud": cloud,
                "file": file_path,
                "angle": angle,
                "position": {
                    "shift_x": 0,  
                    "shift_y": 0,
                    "shift_z": 0,
                    "rot_xy": 0,
                    "rot_xz": 0,
                    "rot_yz": 0,
                }
                # "show_flag": False,
                # "merge_flag": False
            })

            new_data.append({
                "file": file_path,
                "angle": angle,
                "position": {
                    "shift_x": 0,  
                    "shift_y": 0,
                    "shift_z": 0,
                    "rot_xy": 0,
                    "rot_xz": 0,
                    "rot_yz": 0,
                }
            })

        msg = {
            "cmd": "upload_clouds",
            "do_preload_shift": self.cloudsProcessing_preloadShift_checkBox.isChecked(),
            "do_filtration": self.cloudsProcessing_uploadFiltration_checkBox.isChecked(),
            "neighbors": self.cloudsProcessing_filtrationNeighbors_doubleSpinBox.value(),
            "sensivity": self.cloudsProcessing_outlierSensitivity_doubleSpinBox.value(),
            "do_distance_filter": self.cloudsProcessing_enableDistanceFilter_checkBox.isChecked(),
            "max_distance": self.cloudsProcessing_uploadFilterMaxDistance_doubleSpinBox.value(),
            "min_distance": self.cloudsProcessing_uploadFilterMinDistance_doubleSpinBox.value(),
            "shift_x": shift_x,  
            "shift_y": shift_y,
            "shift_z": shift_z,
            "rot_xy": rot_xy,
            "rot_xz": rot_xz,
            "rot_yz": rot_yz,
            "clouds_new_data": new_data  
        }
        self.td_sock.sendall((json.dumps(msg) + "\n").encode())
        print(f"[GUI] Num of clouds became {len(self.td_clouds_info)}")

    def select_cloud_ply_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите PLY файлы формата {name}_cloud_{angle}.ply",
            self.last_td_dir,
            "PLY файлы (*.ply)"
        )

        pattern = re.compile(r"^.+_cloud_(\d+)\.ply$")

        filtered_files = []
        angles = []

        for fpath in files:
            fname = os.path.basename(fpath)
            match = pattern.match(fname)
            if match:
                angle = int(match.group(1))
                filtered_files.append(fpath)
                angles.append(angle)

        if filtered_files:
            self.last_td_dir = os.path.dirname(filtered_files[0])

        return filtered_files, angles
    
    def add_cloud_row(self, cloud_name: str):
        table = self.cloudsProcessing_cloudsList_tableWidget
        row_position = table.rowCount()
        table.insertRow(row_position)

        show_widget = QWidget()
        show_checkbox = QCheckBox()
        layout_show = QHBoxLayout(show_widget)
        layout_show.addWidget(show_checkbox)
        layout_show.setContentsMargins(0, 0, 0, 0)
        layout_show.setAlignment(show_checkbox, Qt.AlignmentFlag.AlignCenter)
        table.setCellWidget(row_position, 0, show_widget)
        show_checkbox.stateChanged.connect(lambda state, r=row_position: self.toggle_cloud_visibility(r, state))

        merge_widget = QWidget()
        merge_checkbox = QCheckBox()
        layout_merge = QHBoxLayout(merge_widget)
        layout_merge.addWidget(merge_checkbox)
        layout_merge.setContentsMargins(0, 0, 0, 0)
        layout_merge.setAlignment(merge_checkbox, Qt.AlignmentFlag.AlignCenter)
        table.setCellWidget(row_position, 1, merge_widget)

        item = QTableWidgetItem(cloud_name)
        #item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)  # только для чтения
        table.setItem(row_position, 2, item)

        return row_position

    def toggle_cloud_visibility(self, row: int, state: int):
        if row >= len(self.td_clouds_info):
            return
            
        msg = {
            "cmd": "set_visible",
            "index": row,
            "value": (state == Qt.CheckState.Checked.value)
        }

        self.td_sock.sendall((json.dumps(msg) + "\n").encode())
        print(f"[GUI] Make {row} row visible: {(state == Qt.CheckState.Checked.value)}")


    def delete_tdrow(self):
        table = self.cloudsProcessing_cloudsList_tableWidget
        selected_ranges = table.selectedRanges()
        if not selected_ranges:
            return

        row = selected_ranges[0].topRow()

        msg = {
            "cmd": "delete_row",
            "index": row
        }

        self.td_sock.sendall((json.dumps(msg) + "\n").encode())

        table.removeRow(row)
        del self.td_clouds_info[row]
        self.rebuild_td_table()

        print(f"[GUI] Delete {row} row visible")

    def rebuild_td_table(self):
        table = self.cloudsProcessing_cloudsList_tableWidget

        # === Сохраняем состояния чекбоксов по cloud_name ===
        previous_states = {}
        for row in range(table.rowCount()):
            name_item = table.item(row, 2)
            if name_item:
                cloud_name = name_item.text()
                show_checkbox = table.cellWidget(row, 0).findChild(QCheckBox)
                merge_checkbox = table.cellWidget(row, 1).findChild(QCheckBox)
                previous_states[cloud_name] = {
                    "show": show_checkbox.isChecked() if show_checkbox else False,
                    "merge": merge_checkbox.isChecked() if merge_checkbox else False,
                }

        # === Очищаем таблицу ===
        table.setRowCount(0)

        # === Пересоздаем строки и восстанавливаем состояния ===
        for i, cloud_info in enumerate(self.td_clouds_info):
            cloud_name = os.path.basename(cloud_info["file"]).replace(".ply", "").replace("_cloud", "")
            row = self.add_cloud_row(cloud_name)

            # Восстанавливаем состояния чекбоксов, если были сохранены
            if cloud_name in previous_states:
                state = previous_states[cloud_name]
                show_checkbox = table.cellWidget(row, 0).findChild(QCheckBox)
                merge_checkbox = table.cellWidget(row, 1).findChild(QCheckBox)

                if show_checkbox:
                    show_checkbox.blockSignals(True)
                    show_checkbox.setChecked(state["show"])
                    show_checkbox.blockSignals(False)

                if merge_checkbox:
                    merge_checkbox.setChecked(state["merge"])


    def on_cloud_table_selection_changed(self):
        selected_row = self.cloudsProcessing_cloudsList_tableWidget.currentRow()
        if selected_row < 0 or selected_row >= len(self.td_clouds_info):
            return

        self.cloudsProcessing_manualAdjustmentShiftX_horizontalSlider.blockSignals(True)
        self.cloudsProcessing_manualAdjustmentShiftY_horizontalSlider.blockSignals(True)
        self.cloudsProcessing_manualAdjustmentShiftZ_horizontalSlider.blockSignals(True)
        self.cloudsProcessing_manualAdjustmentRotShiftXY_horizontalSlider.blockSignals(True)
        self.cloudsProcessing_manualAdjustmentRotShiftXZ_horizontalSlider.blockSignals(True)
        self.cloudsProcessing_manualAdjustmentRotShiftYZ_horizontalSlider.blockSignals(True)

        position = self.td_clouds_info[selected_row].get("position", {})

        self.cloudsProcessing_manualAdjustmentShiftX_horizontalSlider.setValue(int(position.get("shift_x", 0.0) * 100))
        self.cloudsProcessing_manualAdjustmentShiftY_horizontalSlider.setValue(int(position.get("shift_y", 0.0) * 100))
        self.cloudsProcessing_manualAdjustmentShiftZ_horizontalSlider.setValue(int(position.get("shift_z", 0.0) * 100))

        self.cloudsProcessing_manualAdjustmentRotShiftXY_horizontalSlider.setValue(int(position.get("rot_xy", 0.0)))
        self.cloudsProcessing_manualAdjustmentRotShiftXZ_horizontalSlider.setValue(int(position.get("rot_xz", 0.0)))
        self.cloudsProcessing_manualAdjustmentRotShiftYZ_horizontalSlider.setValue(int(position.get("rot_yz", 0.0)))

        self.cloudsProcessing_manualAdjustmentShiftX_textBrowser.setText(f"{position.get('shift_x', 0.0):.2f}")
        self.cloudsProcessing_manualAdjustmentShiftY_textBrowser.setText(f"{position.get('shift_y', 0.0):.2f}")
        self.cloudsProcessing_manualAdjustmentShiftZ_textBrowser.setText(f"{position.get('shift_z', 0.0):.2f}")

        self.cloudsProcessing_manualAdjustmentRotShiftXY_textBrowser.setText(f"{position.get('rot_xy', 0.0):.0f}")
        self.cloudsProcessing_manualAdjustmentRotShiftXZ_textBrowser.setText(f"{position.get('rot_xz', 0.0):.0f}")
        self.cloudsProcessing_manualAdjustmentRotShiftYZ_textBrowser.setText(f"{position.get('rot_yz', 0.0):.0f}")

        self.cloudsProcessing_manualAdjustmentShiftX_horizontalSlider.blockSignals(False)
        self.cloudsProcessing_manualAdjustmentShiftY_horizontalSlider.blockSignals(False)
        self.cloudsProcessing_manualAdjustmentShiftZ_horizontalSlider.blockSignals(False)
        self.cloudsProcessing_manualAdjustmentRotShiftXY_horizontalSlider.blockSignals(False)
        self.cloudsProcessing_manualAdjustmentRotShiftXZ_horizontalSlider.blockSignals(False)
        self.cloudsProcessing_manualAdjustmentRotShiftYZ_horizontalSlider.blockSignals(False)

    def manual_position_correction(self):
        table = self.cloudsProcessing_cloudsList_tableWidget
        selected_items = table.selectedItems()
        if not selected_items:
            return

        selected_row = selected_items[0].row()
        cloud_info = self.td_clouds_info[selected_row]
        
        #cloud = cloud_info["cloud"]

        new = {
            "shift_x": self.cloudsProcessing_manualAdjustmentShiftX_horizontalSlider.value() / 100.0,
            "shift_y": self.cloudsProcessing_manualAdjustmentShiftY_horizontalSlider.value() / 100.0,
            "shift_z": self.cloudsProcessing_manualAdjustmentShiftZ_horizontalSlider.value() / 100.0,
            "rot_xy": self.cloudsProcessing_manualAdjustmentRotShiftXY_horizontalSlider.value(),
            "rot_xz": self.cloudsProcessing_manualAdjustmentRotShiftXZ_horizontalSlider.value(),
            "rot_yz": self.cloudsProcessing_manualAdjustmentRotShiftYZ_horizontalSlider.value(),
        }

        # Показываем в TextBrowser
        self.cloudsProcessing_manualAdjustmentShiftX_textBrowser.setText(f"{new['shift_x']:.2f}")
        self.cloudsProcessing_manualAdjustmentShiftY_textBrowser.setText(f"{new['shift_y']:.2f}")
        self.cloudsProcessing_manualAdjustmentShiftZ_textBrowser.setText(f"{new['shift_z']:.2f}")
        self.cloudsProcessing_manualAdjustmentRotShiftXY_textBrowser.setText(f"{new['rot_xy']:.0f}")
        self.cloudsProcessing_manualAdjustmentRotShiftXZ_textBrowser.setText(f"{new['rot_xz']:.0f}")
        self.cloudsProcessing_manualAdjustmentRotShiftYZ_textBrowser.setText(f"{new['rot_yz']:.0f}")

        # Считаем дельту
        old = cloud_info["position"]
        delta = {key: new[key] - old[key] for key in new}

        
        msg = {
            "cmd": "move_cloud",
            "index": selected_row,
            "delta": delta
        }

        self.td_sock.sendall((json.dumps(msg) + "\n").encode())

        # Обновляем last_transform
        cloud_info["position"] = new

    def set_manual_correction_mode(self, enabled: bool):
        self.cloudsProcessing_manualAdjustmentShiftX_horizontalSlider.setEnabled(enabled)
        self.cloudsProcessing_manualAdjustmentShiftY_horizontalSlider.setEnabled(enabled)
        self.cloudsProcessing_manualAdjustmentShiftZ_horizontalSlider.setEnabled(enabled)
        self.cloudsProcessing_manualAdjustmentRotShiftXY_horizontalSlider.setEnabled(enabled)
        self.cloudsProcessing_manualAdjustmentRotShiftXZ_horizontalSlider.setEnabled(enabled)
        self.cloudsProcessing_manualAdjustmentRotShiftYZ_horizontalSlider.setEnabled(enabled)

    def applyNProcess(self):
        if len(self.td_clouds_info) == 0:
            return

        dir_path = os.path.dirname(self.td_clouds_info[0]["file"]) + "/"
        base_name = os.path.basename(self.td_clouds_info[0]["file"]).split('_')[0]

        table = self.cloudsProcessing_cloudsList_tableWidget

        merge_rows = []
        angles_row = ""
        for row in range(table.rowCount()):
            widget = table.cellWidget(row, 1)  # столбец Merge
            checkbox = widget.layout().itemAt(0).widget()
            
            if checkbox.isChecked():
                # merge_rows.append(self.td_cloud_rows[row])

                before, sep, after = table.item(row, 2).text().partition('_')
                angles_row +=  sep + after
                
                merge_rows.append(row)
                checkbox.setChecked(False)

        if len(merge_rows) == 0:
            QMessageBox.information(self, "Merge", "Choose clouds")
            return

        base_name += angles_row
        f_n = dir_path + base_name + ".ply"

        self.add_cloud_row(base_name)

        s = self.cloudsProcessing_featureRadiusMultipliers_lineEdit.text()
        numbers = [float(part.replace('x', '').strip()) for part in s.split(',')]
        print(numbers)  # [3.0, 5.0, 8.0]

        msg = {
            "cmd": "process_clouds",
            "indexes": merge_rows,
            "file": f_n,

            "voxel_size": self.cloudsProcessing_downsamplingMinPointsPerVoxel_doubleSpinBox.value(),
            "calculate_normals": self.cloudsProcessing_normalsCalculation_checkBox.isChecked(),
            "normals_radius": self.cloudsProcessing_normalEstimationRadius_doubleSpinBox.value(),

            "do_global": self.cloudsProcessing_globalRegistrationIsEnabled_checkBox.isChecked(),
            "feature_multipliers": numbers,
            "align_method": self.cloudsProcessing_aligmentMethod_comboBox.currentText(),

            # "ransac_iter": self.cloudsProcessing_ransacMaxIterations_doubleSpinBox.value(),
            # "ransac_conf": self.cloudsProcessing_ransacConfidenceLevel_doubleSpinBox.value(),
            # "ransac_samples": self.cloudsProcessing_ransacRandomSamples_doubleSpinBox.value(),
            # "fgr_iter": self.cloudsProcessing_fgrMaxIterations_doubleSpinBox.value(),
            # "fgr_div": self.cloudsProcessing_fgrDivisionFactor_doubleSpinBox.value(),
            # "check_edges": self.cloudsProcessing_edgeLengthVerificationCheck_checkBox.isChecked(),
            # "check_dist": self.cloudsProcessing_distanceVerificationCheck_checkBox.isChecked(),
            # "check_normals": self.cloudsProcessing_normalsVerificationCheck_checkBox.isChecked(),

            # --- ICP ---
            "icp_method": self.cloudsProcessing_icpMethod_comboBox.currentText(),
            "use_coarse_icp": self.cloudsProcessing_useCoarseIcpRegistration_checkBox.isChecked(),
            "coarse_icp_iter": self.cloudsProcessing_coarseIcpMaxIterations_doubleSpinBox.value(),
            "coarse_abs": self.cloudsProcessing_useAbsoluteValues_checkBox_2.isChecked(),
            "coarse_abs_thresh": self.cloudsProcessing_coarseThresholdAbsolute_doubleSpinBox.value(),
            "coarse_mult_thresh": self.cloudsProcessing_coarseThresholdMultiplier_doubleSpinBox.value(),
            "coarse_rel_fit": self.cloudsProcessing_coarseIcpRelativeFitness_doubleSpinBox.value(),
            "coarse_rel_rmse": self.cloudsProcessing_coarseIcpRelativeRmse_doubleSpinBox.value(),

            "use_fine_icp": self.cloudsProcessing_useFineIcpRegistration_checkBox.isChecked(),
            "fine_icp_iter": self.cloudsProcessing_fineIcpMaxIterations_doubleSpinBox.value(),
            "fine_abs": self.cloudsProcessing_useAbsoluteValues_checkBox.isChecked(),
            "fine_abs_thresh": self.cloudsProcessing_coarseThresholdAbsolute_doubleSpinBox_2.value(),
            "fine_mult_thresh": self.cloudsProcessing_fineThreshold_doubleSpinBox.value(),
            "fine_rel_fit": self.cloudsProcessing_fineIcpRelativeFitness_doubleSpinBox.value(),
            "fine_rel_rmse": self.cloudsProcessing_fineIcpRelativeRmse_doubleSpinBox.value(),

            "use_color_icp": self.cloudsProcessing_useColorIcp_checkBox.isChecked(),
            "shape_importance": self.cloudsProcessing_shapeImportance_doubleSpinBox.value(),
            "color_importance": self.cloudsProcessing_colorImportance_doubleSpinBox.value(),

            # # --- Постобработка ---
            # "post_voxel_size": self.cloudsProcessing_postprocessingVoxelSize_doubleSpinBox.value(),
            # "fill_holes": self.cloudsProcessing_fillHoles_checkBox.isChecked(),
            # "hole_size": self.cloudsProcessing_postprocessingHoleSize_doubleSpinBox.value(),
            # "filtering": self.cloudsProcessing_filtering_checkBox.isChecked(),
            # "min_artifact_size": self.cloudsProcessing_minArtifactSize_doubleSpinBox.value(),
            # "artifact_sensitivity": self.cloudsProcessing_filteringArtifactSensitivity_doubleSpinBox.value(),
            # "keep_largest": self.cloudsProcessing_keepLargest_checkBox.isChecked(),

            # # --- Сглаживание ---
            # "smoothing_method": self.cloudsProcessing_smoothingMethod_comboBox.currentText(),
            # "smoothing_radius": self.cloudsProcessing_smoothingRadius_doubleSpinBox.value(),
            # "smoothing_iter": self.cloudsProcessing_smoothingIterations_doubleSpinBox.value(),
            # "edge_preserve": self.cloudsProcessing_edgePreserve_checkBox.isChecked(),
            # "taubin_lambda": self.cloudsProcessing_smoothingTaubinLambda_doubleSpinBox.value(),
            # "taubin_mu": self.cloudsProcessing_smoothingTaubinMu_doubleSpinBox.value(),

            # # --- Оптимизация ---
            # "optimize": self.cloudsProcessing_optimization_checkBox.isChecked(),
            # "opt_algo": self.cloudsProcessing_optimizationAlgorithm_comboBox.currentText(),
            # "loop_closure": self.cloudsProcessing_useLoopClosure_checkBox.isChecked(),
            # "min_overlap": self.cloudsProcessing_minOverlapPoints_doubleSpinBox.value(),
            # "edge_prune": self.cloudsProcessing_edgePruneThreshold_doubleSpinBox.value(),
            # "geodesic": self.cloudsProcessing_useGeodesic_checkBox.isChecked(),

            # # --- Экспорт ---
            # "export_method": self.cloudsProcessing_exportReconstructionMethod_comboBox.currentText(),
            # "export_resolution": self.cloudsProcessing_exportMeshResolution_comboBox.currentText(),
            # "export_smoothing": self.cloudsProcessing_exportFinalSmoothing_comboBox.currentText(),
        }
        self.td_sock.sendall((json.dumps(msg) + "\n").encode())

        self.td_clouds_info.append({
            # "cloud": cloud,
            "file": f_n,
            "angle": 0,
            "position": {
                "shift_x": 0,  
                "shift_y": 0,
                "shift_z": 0,
                "rot_xy": 0,
                "rot_xz": 0,
                "rot_yz": 0,
            }
            # "show_flag": False,
            # "merge_flag": False
        })

    # # def get_clouds_for_merge(self):
    # #     table = self.cloudsProcessing_cloudsList_tableWidget
    # #     merge_rows = []
    # #     for row in range(table.rowCount()):
    # #         widget = table.cellWidget(row, 1)  # столбец Merge
    # #         checkbox = widget.layout().itemAt(0).widget()
    # #         if checkbox.isChecked():
    # #             merge_rows.append(self.td_cloud_rows[row])
    # #     return merge_rows

    # def get_selected_row(self):
    #     table = self.cloudsProcessing_cloudsList_tableWidget
    #     selected_ranges = table.selectedRanges()
    #     if selected_ranges:
    #         return selected_ranges[0].topRow()
    #     return None

    # # def process_selected_row(self):
    # #     row = self.get_selected_row()
    # #     if row is not None:
    # #         cloud_info = self.td_cloud_rows[row]
    # #         print("Выбранный cloud:", cloud_info["file"])

    def save_cloud(self):
        # if self.cloudsProcessing_saveFormat_comboBox.currentIndex() == 1:
        #     # msg = {
        #     #     "cmd": "move_cloud",
        #     #     "index": selected_row,
        #     #     "delta": delta
        #     # } 

        print("Enter Save")
        
            
        merged_pcd = o3d.io.read_point_cloud("0_final_reconstruction_final.ply")

        # Фильтрация выбросов (обязательный шаг!)
        merged_pcd, _ = merged_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)

        # Оценка нормалей
        merged_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )

        # Poisson-реконструкция
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            merged_pcd, depth=8, linear_fit=True
        )

        # Обрезка по плотности
        density_threshold = np.percentile(densities, 2)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        # Постобработка сетки
        mesh.remove_non_manifold_edges()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()

        # Альтернатива fill_holes - вручную заделываем дыры через вокселизацию
        voxel_size = 0.010  # Размер вокселя (подбирается под ваш объект)
        mesh = mesh.filter_smooth_taubin(number_of_iterations=5)
        mesh = mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average)

        # Пересчет нормалей
        mesh.compute_vertex_normals()

        # Проверка водонепроницаемости
        print("Watertight status:", mesh.is_watertight())

        # Экспорт
        o3d.io.write_triangle_mesh("final_mesh.stl", mesh)
        o3d.io.write_triangle_mesh("final_mesh.obj", mesh)

        # Визуализация
        o3d.visualization.draw_geometries(
            [mesh],
            mesh_show_back_face=True,
            window_name="Final Mesh"
        )


        return


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MyForm()
    window.resize(630, 680)
    window.show()
    sys.exit(app.exec())
