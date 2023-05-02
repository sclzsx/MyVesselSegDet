# -*- coding: utf-8 -*-

import sys
import cv2
from PIL import Image
import os
import numpy as np
from matplotlib import pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from ui.detect_ui import Ui_MainWindow  # 导入detect_ui的界面

from detection import load_cls_model, pre_process_for_cls, do_classify, do_detect
from segmentation import load_seg_model
from pre_processing import my_PreProc
import torch


class UI_Logic_Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(UI_Logic_Window, self).__init__(parent)
        self.timer_video = QtCore.QTimer()  # 创建定时器
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.init_slots()
        self.output_folder = 'output/'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    # 控件绑定相关操作
    def init_slots(self):
        self.ui.pushButton_do_segment.clicked.connect(self.do_segment)
        self.ui.pushButton_cal_heatmap.clicked.connect(self.cal_heatmap)
        self.ui.pushButton_detect.clicked.connect(self.detect)
        self.ui.pushButton_load_img_and_model.clicked.connect(self.load_img_and_model)
        self.ui.pushButton_pre_processing.clicked.connect(self.pre_processing)
        self.ui.pushButton_save_intermediate_result.clicked.connect(self.save_intermediate_result)
        self.ui.pushButton_save_ultimate_result.clicked.connect(self.save_ultimate_result)

    # 选择权重
    def load_img_and_model(self):
        self.seg_model_path = 'weights/segmentation.pth'
        self.cls_model_path = 'weights/classification.pt'
        self.cls_model = load_cls_model(self.cls_model_path)
        self.seg_model = load_seg_model(self.seg_model_path)

        self.input_size = 512
        self.mean = [0.425753653049469, 0.29737451672554016, 0.21293757855892181]
        self.std = [0.27670302987098694, 0.20240527391433716, 0.1686241775751114]
        self.classify_threshold = 0.5
        self.anomaly_threshold = 0.5

        try:
            img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开图片", "images", "*.png;;All Files(*)")
        except OSError as reason:
            print('文件打开出错啦！核对路径是否正确' + str(reason))
        else:
            # 判断图片是否为空
            if not img_name:
                QtWidgets.QMessageBox.warning(self, u"Warning", u"打开图片失败", buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.img_name = str(img_name)

                img = Image.open(self.img_name)
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                info_show = 'path is: {}, shape is: {}'.format(self.img_name, img.shape)
                print(info_show)

                # 检测信息显示在界面
                self.ui.textBrowser.setText(info_show)

                # 检测结果显示在界面
                self.result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
                self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                          QtGui.QImage.Format_RGB32)
                self.ui.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                self.ui.label.setScaledContents(True)  # 设置图像自适应界面大小

    # 选择图像
    def pre_processing(self):
        print('图像预处理')

        data = Image.open(self.img_name).convert('RGB')
        rgb = np.array(data)
        data = rgb.transpose((2, 0, 1))
        data = np.expand_dims(data, axis=0)
        data = my_PreProc(data)
        self.pre_process_np = data
        print('pre process done')
        print(data.shape)
        data_vis = data[0, 0, :, :]
        data_vis = np.clip(data_vis * 255, 0, 255).astype('uint8')
        data_vis = cv2.cvtColor(data_vis, cv2.COLOR_GRAY2BGR)

        info_show = 'mean:{}, std:{}'.format(np.mean(data_vis), np.std(data_vis))

        # 检测信息显示在界面
        self.ui.textBrowser.setText(info_show)

        # 检测结果显示在界面
        self.result = cv2.cvtColor(data_vis, cv2.COLOR_BGR2BGRA)
        self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
        self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                  QtGui.QImage.Format_RGB32)
        self.ui.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        self.ui.label.setScaledContents(True)  # 设置图像自适应界面大小

    # 分割
    def do_segment(self):
        pre_process_tensor = torch.from_numpy(self.pre_process_np).float()
        with torch.no_grad():
            pre_process_tensor = pre_process_tensor.cpu()
            seg_out_tensor = self.seg_model(pre_process_tensor)
        seg_out = np.array(torch.max(seg_out_tensor.data, 1)[1].squeeze().cpu())
        seg_out = np.where(seg_out > 0, 255, 0).astype('uint8')
        seg_out = cv2.cvtColor(seg_out, cv2.COLOR_GRAY2BGR)
        self.seg_out = seg_out

        info_show = 'segment'

        # 检测信息显示在界面
        self.ui.textBrowser.setText(info_show)

        # 检测结果显示在界面
        self.result = cv2.cvtColor(seg_out, cv2.COLOR_BGR2BGRA)
        self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
        self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                  QtGui.QImage.Format_RGB32)
        self.ui.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        self.ui.label.setScaledContents(True)  # 设置图像自适应界面大小

    # 计算热力图
    def cal_heatmap(self):
        print('计算热力图')
        img_tensor, img_np, label = pre_process_for_cls(self.img_name, self.input_size, self.mean, self.std)
        heatmap, pred = do_classify(img_tensor, self.cls_model, self.classify_threshold)
        self.heatmap = heatmap
        self.pred = pred
        self.img_np = img_np
        self.label = label

        plt.imshow(heatmap)
        plt.savefig(self.output_folder + '/tmp.png')
        heatmap_vis = cv2.imread(self.output_folder + '/tmp.png')
        self.heatmap_vis = heatmap_vis
        if os.path.exists(self.output_folder + '/tmp.png'):
            os.remove(self.output_folder + '/tmp.png')
        info_show = 'prediction is:{}'.format(pred)

        # 检测信息显示在界面
        self.ui.textBrowser.setText(info_show)

        # 检测结果显示在界面
        self.result = cv2.cvtColor(heatmap_vis, cv2.COLOR_BGR2BGRA)
        self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
        self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                  QtGui.QImage.Format_RGB32)
        self.ui.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        self.ui.label.setScaledContents(True)  # 设置图像自适应界面大小

    # 病变检测
    def detect(self):
        print('病变检测')
        det_np, max_bbox = do_detect(self.img_np, self.heatmap, self.pred, self.anomaly_threshold)
        self.det_np = det_np
        info_show = 'label is: {}, pred is: {}, bbox is:{}'.format(self.label, self.pred, max_bbox)

        # 检测信息显示在界面
        self.ui.textBrowser.setText(info_show)

        # 检测结果显示在界面
        self.result = cv2.cvtColor(det_np, cv2.COLOR_BGR2BGRA)
        self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
        self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                  QtGui.QImage.Format_RGB32)
        self.ui.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        self.ui.label.setScaledContents(True)  # 设置图像自适应界面大小

    # 保存热力图
    def save_intermediate_result(self):
        print('保存中间结果')

        file_path_seg = str(self.img_name).replace('images', 'output').replace('.png', '_seg.png')
        print(file_path_seg)
        tmp = cv2.cvtColor(self.seg_out, cv2.COLOR_BGR2RGB)
        tmp = Image.fromarray(tmp)
        tmp.save(file_path_seg)

        file_path_heatmap = str(self.img_name).replace('images', 'output').replace('.png', '_heatmap.png')
        print(file_path_heatmap)
        tmp = cv2.cvtColor(self.heatmap_vis, cv2.COLOR_BGR2RGB)
        tmp = Image.fromarray(tmp)
        tmp.save(file_path_heatmap)

        QtWidgets.QMessageBox.information(self, u"Notice", u"中间结果已保存", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)

    # 保存检测结果
    def save_ultimate_result(self):
        print('保存检测结果')
        file_path = str(self.img_name).replace('images', 'output').replace('.png', '_detect.png')
        print(file_path)
        det = cv2.cvtColor(self.det_np, cv2.COLOR_BGR2RGB)
        det = Image.fromarray(det)
        det.save(file_path)
        QtWidgets.QMessageBox.information(self, u"Notice", u"检测结果已保存", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    current_ui = UI_Logic_Window()
    current_ui.show()
    sys.exit(app.exec_())
