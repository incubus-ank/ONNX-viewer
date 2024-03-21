# -*- coding: utf-8 -*-
#   Developed by Alexander Kraynikov krajnikov.a@edu.narfu.ru
import os
import sys

from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, \
    QPushButton, QHBoxLayout, QComboBox, QLineEdit
from PyQt5.QtGui import QPixmap, QDoubleValidator
from PyQt5.QtCore import pyqtSignal, Qt, QThread

import numpy as np
import onnxruntime as ort
import cv2 as cv
import time

from YOLOv8 import YOLOv8

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, model_name, conf, iuo, record=False):
        self.model_name = model_name
        self.conf = conf
        self.iuo = iuo
        self.record = record
        if self.record:
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            self.out = cv.VideoWriter("./videos/" + \
                                    time.strftime("%Y-%m-%d_%H-%M-%S") + \
                                    ".avi",  
                            fourcc,
                            10.0, (640, 480)) 

        super().__init__()
        self.is_run = True

    def run(self):
        cap = cv.VideoCapture(0)

        self.detector = YOLOv8("models/" + self.model_name,
                               self.conf,
                               self.iuo)

        while self.is_run:
            ret, frame = cap.read()
            if ret:

                boxes, scores, class_ids = self.detector(frame)

                combined_img = self.detector.draw_detections(frame)
                if self.record:
                    print("record")
                    self.out.write(cv.resize(frame, (640, 480)))

                self.change_pixmap_signal.emit(combined_img)
        cap.release()

    def stop(self):
        self.is_run = False
        self.wait()



class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ONNX viewer")

        self.select_model = QComboBox()
        self.select_model.addItems(os.listdir("models"))
        self.select_model.currentIndexChanged.connect(self.changemodel)

        self.thread = VideoThread(os.listdir("models")[0],
                                  0.7,
                                  0.5)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        self.image_width = 1280
        self.image_height = 720

        self.record = False

        self.image_label = QLabel(self)
        self.image_label.resize(self.image_width, self.image_height)

        self.button_record = QPushButton("Запись")
        self.button_record.clicked.connect(self.start_record)
        self.button_change = QPushButton("Перезапустить")
        self.button_change.clicked.connect(self.changemodel)

        self.label_confidence = QLabel()
        self.label_confidence.setText("confidence:")
        self.line_confidence = QLineEdit()
        self.line_confidence.setValidator(QDoubleValidator(0.01,1.00,2))
        self.line_confidence.setText("0.70")

        self.label_iuo = QLabel()
        self.label_iuo.setText("iuo:")
        self.line_iuo = QLineEdit()
        self.line_iuo.setValidator(QDoubleValidator(0.01,1.00,2))
        self.line_iuo.setText("0.50")

        self.vbox = QVBoxLayout()
        self.hbox = QHBoxLayout()

        self.hbox.addWidget(self.image_label)
        self.vbox.addWidget(self.button_record)
        self.vbox.addWidget(self.select_model)

        self.vbox.addWidget(self.label_confidence)
        self.vbox.addWidget(self.line_confidence)
        self.vbox.addWidget(self.label_iuo)
        self.vbox.addWidget(self.line_iuo)
        self.vbox.addWidget(self.button_change)
        self.vbox.addStretch()

        self.hbox.addLayout(self.vbox)
        
        self.setLayout(self.hbox)

    def start_record(self):
        self.record = not self.record
        self.changemodel()

    def changemodel(self):
        self.line_iuo.text

        self.thread.stop()
        self.thread = VideoThread(self.select_model.currentText(),
                                  float(self.line_confidence.text()),
                                  float(self.line_iuo.text()),
                                  self.record)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()


    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, 
                                            w, 
                                            h, 
                                            bytes_per_line, 
                                            QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_width, 
                                        self.image_height, 
                                        Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())