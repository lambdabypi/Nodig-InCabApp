from cmath import asin
from operator import itemgetter
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image, ImageTk
from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import time
import plotly.graph_objects as go
from math import atan2, cos
from math import sin
import math
from skimage.io import imread
from skimage.color import rgb2gray, rgb2hsv
from skimage.measure import label, regionprops_table
from scipy.ndimage import median_filter
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO; sns.set()
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import psycopg2
import tkinter as tk
import unittest
from PyQt5.QtTest import QTest
import pytest
import math
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import time
from cmath import asin
from datainsertion import CreateTable,insertmarkerdata,FetchData
from gps_utils import CartesianToGPS,ConvertGPStocartesian
# get_bearing,DistanceLatLong
from ThreeDcoord import project_disparity_to_3d
from distances import DistanceBetweenMarkingAndBucket,DistanceBetweenMarkings
from nodig_region import LinerRegressionOnTPoint, RoiForBoundary,drawMajorandMinorAxis,createOverlay,get_mask, UpdateOverlay


# Variable Declarations
#  ***************************************************************************************************************
Bucket3Dcoordinates=[]
BucketPoints=[]
pointsP=[]
Marker3DCoordinate=[]
lat_long_dig = []
predictx=28
predicty=36
skipframe=600
value=4
Flag=False
alpha=0.3

def yolov8_prediction(image_file):
    model_file_path = r'C:\Users\shrey\Desktop\IncabApp\Nodig\best9623 (1).pt'
    name = 'yolov8n_custom9'
    model = YOLO(model_file_path)
    results = model.predict(image_file, save=True, imgsz=640, conf=0.5)
    print("model_results",results[0])
    # print(results.xyxy[0])
    # print(len(results.boxes))
    annotated_frame = results[0].plot()
        # Display the annotated frame
    # cv2.imshow("YOLOv8 Inference", annotated_frame)
    # cv2.waitKey(0)
    for result in results:
        boxes = result.boxes
        
    return boxes

# Class_name : CodeInputDialog
# Description : A QDialog Widget created to allow the user to input a predefined 4 digit code.
# Functions : initUI, print_digit_inputs, update_code, ok_button_clicked, create_digit_input, digit_input_layout, clear_input
# return : None

class CodeInputDialog(QDialog):
    enable_button_draw_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
    
    #Initialise the dialog window
    def initUI(self):
        self.setWindowTitle("Enter Admin Code")
        self.setModal(True)
        self.code = ""
        
        self.code_label = QLabel("Enter 4-digit code:")
        self.code_input = QLineEdit()
        self.code_input.setMaxLength(4)
        self.code_input.textEdited.connect(self.update_code)
        self.digit_widgets = []
        self.ok_button = QPushButton("OK")
        self.ok_button.setFixedSize(100, 50)
        self.ok_button.clicked.connect(self.ok_button_clicked)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setFixedSize(100, 50)
        self.clear_button.clicked.connect(self.clear_input)

        self.number_panel = NumberPanel()

        self.create_digit_input()

        layout = QVBoxLayout()
        layout.addWidget(self.code_label)
        
        layout.addLayout(self.digit_input_layout())
        layout.addWidget(self.number_panel)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)

        layout.addWidget(line)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.clear_button)
        button_layout.setAlignment(Qt.AlignCenter)  # Center-align the button
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        

    #print the inputed digits on the boxes
    def print_digit_inputs(self):
        for digit_widget in self.digit_widgets:
            print("Digit Input:", digit_widget.text())

    #Update the digits inputed by the user on the number panel to the input boxes
    def update_code(self):
        self.code = ""
        for digit_input in self.digit_widgets:
            self.code += digit_input.text()
            
    #When the ok button is clicked, check if the entered code is the same as the predefined code and allow the user to edit.
    def ok_button_clicked(self):
        entered_code = ""
        for digit_widget in self.digit_widgets:
            entered_code += digit_widget.text()
        predefined_code = "1234"  # Replace with your predefined code

        print("Entered Code:", entered_code)
        print("Predefined Code:", predefined_code)
        entered_code = entered_code.strip()
        predefined_code = predefined_code.strip()
        if entered_code == predefined_code:
            # The entered code is correct
            self.accept()
            self.parent().enable_button_draw()
        else:
            # The entered code is incorrect
            QMessageBox.warning(self, "Invalid Code", "Please enter a valid code.")
    
    #Digits are created here
    def create_digit_input(self):
        for _ in range(4):  # Create 4 digit inputs
            digit_input = DigitInput(self)
            digit_input.setFixedHeight(50)
            digit_input.setFixedWidth(50)
            digit_input.setMaxLength(1)
            digit_input.setAlignment(Qt.AlignCenter)

            validator = QRegExpValidator(QRegExp(r"\d{0,1}"))  # Restrict input to a single digit
            digit_input.setValidator(validator)

            digit_input.textEdited.connect(self.update_code)  # Connect to update_code function

            self.digit_widgets.append(digit_input)

    #The input boxes are created here
    def digit_input_layout(self):
        layout = QHBoxLayout()
        for digit_widget in self.digit_widgets:
            layout.addWidget(digit_widget)
        return layout
    
    #function to clear the input
    def clear_input(self):
        for digit_widget in self.digit_widgets:
            digit_widget.clear()

# Class_name : DigitInput
# Description : A QLineEdit Widget made to create the input boxes.
# Functions : __init__
# return : None

#Initialise the input boxes with the size, restrictions and other restrictions
class DigitInput(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.setFixedWidth(50)
        self.setMaxLength(1)
        self.setAlignment(Qt.AlignCenter)

        validator = QRegExpValidator(QRegExp(r"\d{0,1}"))  # Restrict input to a single digit
        self.setValidator(validator)

# Class_name : NumberPanel
# Description : A Widget created to display a number panel to allow the user to interactively select the number to be inputed.
# Functions : __init__, number_clicked
# return : None

class NumberPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.button_layout = QVBoxLayout()
        self.button_grid_layout = QtWidgets.QGridLayout()

        button_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        row, col = 0, 0
        for label in button_labels:
            self.button = QPushButton(str(label))
            self.button.setFixedSize(100, 50)
            self.button.clicked.connect(self.number_clicked)
            self.button_grid_layout.addWidget(self.button, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1

        self.button0 = QPushButton('0')
        self.button0.setFixedSize(100, 50)
        self.button0.clicked.connect(self.number_clicked)
        self.button_grid_layout.addWidget(self.button0, 3, 1)

        self.button_layout.addLayout(self.button_grid_layout)

        layout = QVBoxLayout()
        layout.addLayout(self.button_layout)
        self.setLayout(layout)

    #A function to assign the number clicked to a variable and push to the input boxes
    def number_clicked(self):
        clicked_button = self.sender()
        code_input = self.parent().digit_widgets[0]
        for self.digit_input in self.parent().digit_widgets:
            if self.digit_input.text() == "":
                code_input = self.digit_input
                break
        code_input.setText(clicked_button.text())

class DeleteNumberPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.button_layout = QVBoxLayout()
        self.button_grid_layout = QtWidgets.QGridLayout()

        button_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        row, col = 0, 0
        for label in button_labels:
            self.button = QPushButton(str(label))
            self.button.setFixedSize(100, 50)
            self.button.clicked.connect(self.number_clicked)
            self.button_grid_layout.addWidget(self.button, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1

        self.button0 = QPushButton('0')
        self.button0.setFixedSize(100, 50)
        self.button0.clicked.connect(self.number_clicked)
        self.button_grid_layout.addWidget(self.button0, 3, 1)

        self.button_space = QPushButton('Space')
        self.button_space.setFixedSize(100, 50)
        self.button_space.clicked.connect(self.space_clicked)
        self.button_grid_layout.addWidget(self.button_space, 3, 0)  # Added space button

        self.button_clear = QPushButton('Clear')
        self.button_clear.setFixedSize(100, 50)
        self.button_clear.clicked.connect(self.clear_clicked)
        self.button_grid_layout.addWidget(self.button_clear, 3, 2)  # Added clear button

        self.button_layout.addLayout(self.button_grid_layout)

        layout = QVBoxLayout()
        layout.addLayout(self.button_layout)
        self.setLayout(layout)

    #A function to assign the number clicked to a variable and push to the input boxes
    def number_clicked(self):
        clicked_button = self.sender()
        number = clicked_button.text()
        self.parent().line_edit.insert(number)

    def space_clicked(self):  # Space button click handler
        self.parent().line_edit.insert(" ")  # Insert a space character

    def clear_clicked(self):
        self.parent().line_edit.clear()  # Clear the line_edit text

# Class_name : ImageButton
# Description : A QPushButton to display a settings icon.
# Functions : __init__
# return : None

class ImageButton(QPushButton):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 50)
        self.setStyleSheet(f"QPushButton {{ border-image: url({image_path}); }}")
        icon_size = QtCore.QSize(100, 100)
        self.setIconSize(icon_size)

class NumberInputDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Nodig Zones to be deleted")

        self.label = QLabel("Enter the label of your desired Tolerence mark:")
        self.line_edit = QLineEdit()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)

        self.numberpanel = DeleteNumberPanel()
        self.numberpanel.line_edit = self.line_edit  # Assign line_edit widget

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.numberpanel)
        layout.addWidget(self.ok_button)
        
        self.setLayout(layout)

    def get_numbers(self):
        if self.exec_() == QDialog.Accepted:
            numbers = [int(num) for num in self.line_edit.text().split()]
            return numbers
        else:
            return None

# Class_name : IncabMonitor
# Description : The main widget that holds evrything in place
# Functions : __init__, initUI, eventFilter, toggle_drawing, enable_button_draw, button_clicked, stop_and_exit, display_frame, update_position
# return : None

class IncabMonitor(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        self.playing = True  # Flag to track the playing state
        self.points = []
        self.start_point = None
        self.mid_point = None
        self.minor_point= None
        self.line_items = []
        self.slope= None
        self.constant= None
        self.minorvalue= None
        self.drawing_enabled=False
        self.add_enabled=False
        self.deletion_var=[]
        self.showlabel=False
        #self.pause_button = None  # Reference to the pause button

    def initUI(self):
        self.setWindowTitle("Cobravision - In Cab Monitor")

        # Set the window dimensions
        window_width, window_height = 800, 600  # Set your desired dimensions

        # Center the window on the screen
        screen_geometry = QtWidgets.QApplication.primaryScreen().geometry()
        screen_width, screen_height = screen_geometry.width(), screen_geometry.height()
        window_x = (screen_width - window_width) // 2
        window_y = (screen_height - window_height) // 2

        # Set the window geometry
        self.setGeometry(window_x, window_y, window_width, window_height)

        # Show the window in fullscreen
        self.showMaximized()

        # Create a qgraphicsview to display the video frame
        self.view = QGraphicsView(self)
        self.view.setSceneRect(QtCore.QRectF())
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setStyleSheet("background-color: transparent;")
        self.view.viewport().installEventFilter(self)
        self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.label = QtWidgets.QLabel(self.view.viewport())
        self.label.setStyleSheet("background-color: transparent;")

        # Create a button to allow editing
        #self.button_edit = ImageButton("icons8-settings-48.png")
        self.button_edit = QToolButton()
        self.button_edit.setIcon(QIcon(r'C:\Users\shrey\Desktop\IncabApp\Nodig\settings.png'))
        self.button_edit.clicked.connect(self.button_clicked)
        self.button_edit.setFixedSize(100, 50)

        self.heading_label = QtWidgets.QLabel("Cobravision - In Cab Monitor", self)
        # Apply style to the heading_label
        self.heading_label.setStyleSheet("""
            background-color: #1e1e1e;
            color: #f5f5f5;
            font-size: 18px;
            font-weight: bold;
            padding: 10px;
        """)

        # Create a button to start the video playback
        self.button_start = QtWidgets.QPushButton(self)
        self.button_start.setText("Start")
        self.button_start.clicked.connect(self.display_frame)
        self.button_start.setFixedSize(100, 50)

        # Create a button to stop and exit the process
        self.button_stop = QtWidgets.QPushButton(self)
        self.button_stop.setText("Exit")
        self.button_stop.clicked.connect(self.stop_and_exit)
        self.button_stop.setEnabled(False)  # Disable initially
        self.button_stop.setFixedSize(100, 50)

        # Create the GUI elements, including a pause button
        self.pause_button = QtWidgets.QPushButton(self)
        self.pause_button.setText("Pause")
        self.pause_button.setEnabled(False)  # Disable initially
        self.pause_button.setFixedSize(100, 50)
        self.pause_button.setCheckable(True)  # Make the button toggleable
        self.pause_button.clicked.connect(self.toggle_pause)

        # Create a button to enable drawing mode
        self.button_draw = QtWidgets.QPushButton(self)
        self.button_draw.setText("Edit")
        self.button_draw.setEnabled(False)
        self.button_draw.setFixedSize(100, 50)
        self.button_draw.setCheckable(True)  # Make the button toggleable
        self.button_draw.clicked.connect(self.toggle_drawing)

        self.button_delete = QtWidgets.QPushButton(self)
        self.button_delete.setText("Delete")
        self.button_delete.setEnabled(False)
        self.button_delete.setFixedSize(100, 50)
        self.button_delete.setCheckable(True)  # Make the button toggleable
        self.button_delete.clicked.connect(self.delete_func)

        self.button_add = QtWidgets.QPushButton(self)
        self.button_add.setText("Add")
        self.button_add.setEnabled(False)
        self.button_add.setFixedSize(100, 50)
        self.button_add.setCheckable(True)  # Make the button toggleable
        self.button_add.clicked.connect(self.toggle_editing)
        

        # Create a button
        self.button_style = QtWidgets.QPushButton(self)
        self.button_style.setText("Light Mode")
        self.button_style.setFixedSize(100, 50)
        self.button_style.setCheckable(True)  # Make the button toggleable
        self.button_style.clicked.connect(self.toggle_style)

        self.button_start.setAttribute(QtCore.Qt.WA_AcceptTouchEvents)
        self.button_stop.setAttribute(QtCore.Qt.WA_AcceptTouchEvents)
        self.view.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.view.grabGesture(Qt.PinchGesture)
        self.view.grabGesture(Qt.SwipeGesture)

        # Create a horizontal line
        Hline = QFrame()
        Hline.setFrameShape(QFrame.HLine)
        Hline.setFrameShadow(QFrame.Sunken)

        # Create a Vertical line
        Vline = QFrame()
        Vline.setFrameShape(QFrame.VLine)
        Vline.setFrameShadow(QFrame.Sunken)

        # Set up the layout
        main_layout = QtWidgets.QGridLayout(self)  # Use QGridLayout for more control

        # Create a layout for self.view
        view_layout = QtWidgets.QHBoxLayout()
        view_layout.addWidget(self.view)
        

        # Set the alignment of the layout to center
        

        edit_layout = QVBoxLayout()
        edit_layout.addWidget(self.button_edit)
        edit_layout.addStretch()
        edit_layout.addWidget(self.button_add)
        edit_layout.addWidget(self.button_delete)
        edit_layout.addStretch()
        edit_layout.addWidget(self.button_style)
        
        main_layout.addLayout(edit_layout, 0, 0, 3, 1)
        main_layout.addWidget(Vline, 0, 1, 3, 1)
        main_layout.addWidget(self.heading_label, 0, 0, 1, 3)
        main_layout.addLayout(view_layout, 1, 0, 1, 3)  # Add label widget below the buttons, span it across three columns
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.heading_label.setAlignment(QtCore.Qt.AlignCenter)
        button_layout = QtWidgets.QHBoxLayout()  # Create a horizontal layout for the start and stop buttons
        button_layout.addWidget(self.button_start)
        button_layout.addWidget(self.button_stop)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.button_draw)

        # Other variables
        
        main_layout.addWidget(Hline, 2, 0, 1, 3)
        main_layout.addLayout(button_layout, 3, 0, 1, 3)  # Add button layout below the label, span it across three columns
        self.setLayout(main_layout)
        self.hide()  # Hide the player initially
        
        
        # Apply modern styling
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
            }
            
            QGraphicsView {
                background-color: #1e1e1e;
            }

            QLabel {
                background-color: rgba(255, 255, 255, 150);
                color: #000000;
            }

            QToolButton {
                background-color: transparent;
                border: none;
            }

            QPushButton {
                color: #ffffff;
                border: none;
                font-weight: bold;
                padding: 8px;
                border-radius: 10px;
            }

            QPushButton:enabled {
                background-color: #3498db;
            }

            QPushButton:disabled {
                background-color: transparent;
                color: #888;
            }

            QPushButton:hover:enabled {
                background-color: #2980b9;
            }

            QPushButton:checked:enabled {
                background-color: #16a085;
            }
                                
            QPushButton:checked:disabled {
                background-color: transparent;
                color: #888;
            }
                                
            QPushButton:pressed:enabled {
                background-color: #1f618d;
            }
                                
            QPushButton:pressed:disabled {
                background-color: transparent;
                color: #888;
            }

            ...
        """)
        self.dark_style = True

    def toggle_style(self):
        # Toggle between two styles
        self.dark_style = not self.dark_style

        if self.dark_style:
            self.button_edit.setIcon(QIcon(r'C:\Users\shrey\Desktop\IncabApp\Nodig\settings.png'))
            self.button_style.setText("Light Mode")
            self.heading_label.setStyleSheet("""
                background-color: #1e1e1e;
                color: #f5f5f5;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
            """)
            self.setStyleSheet("""
                QWidget {
                    background-color: #1e1e1e;
                }
                
                QGraphicsView {
                    background-color: #1e1e1e;
                }

                QLabel {
                    background-color: rgba(255, 255, 255, 150);
                    color: #000000;
                }

                QToolButton {
                    background-color: transparent;
                    border: none;
                }

                QPushButton {
                    color: #ffffff;
                    border: none;
                    font-weight: bold;
                    padding: 8px;
                    border-radius: 6px;
                }

                QPushButton:enabled {
                    background-color: #3498db;
                }

                QPushButton:disabled {
                    background-color: transparent;
                    color: #888;
                }

                QPushButton:hover:enabled {
                    background-color: #2980b9;
                }

                QPushButton:checked:enabled {
                    background-color: #16a085;
                }
                                    
                QPushButton:checked:disabled {
                    background-color: transparent;
                    color: #888;
                }
                                    
                QPushButton:pressed:enabled {
                    background-color: #1f618d;
                }
                                    
                QPushButton:pressed:disabled {
                    background-color: transparent;
                    color: #888;
                }
            """)
        else:
            self.button_edit.setIcon(QIcon(r'C:\Users\shrey\Desktop\IncabApp\Nodig\cogwheel.png'))
            self.button_style.setText("Dark Mode")
            self.heading_label.setStyleSheet("""
                background-color: #f2f2f2;
                color: #333333;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
            """)
            self.setStyleSheet("""                               
                QWidget {
                    background-color: #f5f5f5;
                }
                
                
                QSlider {
                    background-color: #f5f5f5;
                }
                
                QSlider::handle:horizontal {
                    background-color: #555;
                }
                
                QSlider::handle:vertical {
                    background-color: #555;
                    
                }

                QToolButton {
                    background-color: transparent;
                    border: none;
                }
                
                QPushButton {
                    background-color: #4CAF50;
                    border: none;
                    color: #fff;
                    font-weight: bold;
                    padding: 8px;
                    border-radius: 6px;
                }
                
                QPushButton:hover:hover {
                    background-color: #45a049;
                }
                
                QPushButton:pressed, QToolButton:pressed {
                    background-color: #3d8b40;
                }
                
                QPushButton:disabled, QToolButton:disabled {
                    background-color: #ccc;
                    color: #888;
                }
            """)

    def eventFilter(self, obj, event):
        if hasattr(self, 'add_enabled') and self.add_enabled:
            if obj == self.view.viewport() and event.type() == QtCore.QEvent.MouseButtonPress:
                if event.button() == QtCore.Qt.LeftButton:
                    position = event.pos()
                    if self.mid_point is None:
                        self.mid_point = self.view.mapToScene(position)
                        text = "Mid point: ({}, {})".format(self.mid_point.x(), self.mid_point.y())
                        print("Mid point coordinates: ({}, {})".format(self.mid_point.x(), self.mid_point.y()))
                    elif self.start_point is None:
                        viewport_rect = self.view.viewport().rect()
                        edge_margin = 10  # Adjust the margin value as per your requirement
                        if (position.x() <= viewport_rect.left() + edge_margin or
                                position.x() >= viewport_rect.right() - edge_margin or
                                position.y() <= viewport_rect.top() + edge_margin or
                                position.y() >= viewport_rect.bottom() - edge_margin):
                            self.start_point = self.view.mapToScene(position)
                            text = "Start point: ({}, {})".format(self.start_point.x(), self.start_point.y())
                            print("Start point coordinates: ({}, {})".format(self.start_point.x(), self.start_point.y()))
                        else:
                            # Start point tapped inside the frame, do nothing
                            return True
                    elif self.minor_point is None:
                        self.minor_point = self.view.mapToScene(position)
                        text = "Minor point: ({}, {})".format(self.minor_point.x(), self.minor_point.y())
                        print("Minor point coordinates: ({}, {})".format(self.minor_point.x(), self.minor_point.y()))
                        
                    else:
                        # Clear previous points and lines
                        self.mid_point = None
                        self.start_point = None
                        self.minor_point = None
                        self.slope = None
                        self.constant = None
                        self.minorvalue = None
                        text = ""

                    self.view.setToolTip(text)
                    self.label.setText(text)
                    self.label.adjustSize()
                    self.label.move(position)
                    self.label.show()
                    return True

        return super().eventFilter(obj, event)
    
    #Toggles delete mode
    def delete_func(self):
        
        number_input_dialog = NumberInputDialog()
        self.deletion_var = number_input_dialog.get_numbers()

    #function to toggle the edit button to show the sliders

    def toggle_drawing(self):
        self.showlabel=True
        self.button_add.setEnabled(True)
        self.button_delete.setEnabled(True)

    def toggle_editing(self):
        self.drawing_enabled = not self.drawing_enabled
        if self.drawing_enabled:
            self.button_add.setText("Update")
            self.add_enabled=True
            self.mid_point = None
            self.start_point = None
            self.minor_point = None
            self.slope = None
            self.constant = None
            self.minorvalue = None
            self.label.setMouseTracking(True)
            self.label.installEventFilter(self)
        else:
            self.button_add.setText("Add")
            self.add_enabled=False
            self.label.setMouseTracking(False)
            self.label.removeEventFilter(self)

    #function to enable the edit button(initially disabled)
    def enable_button_draw(self):
        self.button_draw.setEnabled(True)

    #function to call the Number panel widget when the settings button is pressed
    def button_clicked(self):
        code_input_dialog = CodeInputDialog(self)
        if code_input_dialog.exec_() == QDialog.Accepted:
            code = code_input_dialog.code_input.text()
            if len(code) == 4:
                number_panel = NumberPanel(self)
                number_panel.show()
    
    def toggle_pause(self):
        if self.playing:
            self.pause()
        else:
            self.resume()

    def pause(self):
        self.playing = False
        self.pause_button.setText("Resume")

    def resume(self):
        self.playing = True
        self.pause_button.setText("Pause")

    #function to exit the process
    def stop_and_exit(self):
        self.button_start.setEnabled(True)
        self.button_stop.setEnabled(False)        
        sys.exit()
    
    #Function that displays the processed frames
    def display_frame(self):
        self.pause_button.setEnabled(True)
        global pointsP
        global Bucket3Dcoordinates
        global Marker3DCoordinate
        global BucketPoints
        global lat_long_dig
        global alpha
        global predictx
        global predicty
        global Flag

        # load model to detect T marking in frame
        rf = Roboflow(api_key="khzPtVBExR4BSFqdeXjT")
        project = rf.workspace().project("no-dig-segmentation")
        model = project.version(6).model

        # Load model to detect bucket (assumed)  in frame
        model1 = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        f = cv2.VideoCapture('color.mp4')
    
        depth = cv2.VideoCapture('disparity.mp4')
    
        success = 1
        count = 0
            
        while success:
            if self.playing:

                count+=1

                ret, frame = f.read()
                ret1, frame1 = depth.read()
                
                height= frame1.shape[0]
                width = frame1.shape[1]
                
                frame = cv2.resize(frame, (width,height))
                
                
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                
                if(count==skipframe):
                    CreateTable()
                if(count<skipframe):
                    continue
                # break
                if ret == True:
                    
                    cv2.imwrite('temp.jpg', frame)
                    
                
                    points=[]
                    pointsB=[]
                    
                    boxes = yolov8_prediction(frame)
                        # continue
                    results = model1(frame)
                
                
                    points=[]
                    pointsB=[]
                    pointsP=[]
                    
                    detections_dict = {}
                    for i in range (len(boxes.xyxy)):
                            # Boxes object for bbox outputs
                            print(boxes)
                            clas = boxes.cls[i].tolist()
                            print(clas)
                            print("bbbbbbbbbbbbbbbbbbb",boxes.xyxy)
                            if clas==6.0:
                                
                                continue
                            cords = boxes.xyxy[i].tolist()
                            x1, y1, x2, y2 = cords
                            print(x1,y1,x2,y2)
                        
                        
                            

                            start_point = (int(x1), int(y1))
                            end_point = (int(x2), int(y2))
                            mid_point = (int((start_point[0]+end_point[0])/2),int((start_point[1]+end_point[1])/2))



                            dist= project_disparity_to_3d(frame1, mid_point)
                            print(dist)

                        
                            lat1 = 12.917816972222222441666666666667
                            lon1 = 77.633101972222227971083333333333
                            
                            lat2 = 12.917817872222222441666666666667
                            lon2 = 77.633102872222227971083333333333
                            
                            
                            
                            
                            
                            x = dist[0][0]
                            y=dist[0][1]
                            z= dist[0][2]
                            
                            # after rotating z and y axis by 90 degrees clockwise
                            X=x 
                            Y= z
                            Z = -y
                            
                            # after rotating y and x by 90 degree in anticlockwise
                            
                            x = -Y
                            y= X
                            z= Z
                            
                            print("coordinates after rotation :", x,y,z)
                            X,Y,Z=ConvertGPStocartesian(lat1,lon1) # camera old coordinates
                            X1,Y1,Z1=ConvertGPStocartesian(lat2,lon2) # camera new coordinates
                            
                        
                            
                            # Adding marking coordinates to camera coordinates
                            x=X+x 
                            y=Y+y 
                            z= Z+z 
                            
                        
                            
                            lat,lon=CartesianToGPS(x,y,z)
                            
                            print("marking lat long by approach 1",lat,lon)
                            lat_long_dig.append((lat,lon))
                            
                            if(dist and len(dist)):
                                print(dist[1])
                                points.append(dist[0])
                                Marker3DCoordinate.append(dist[0][1])
                                
                                pointsP.append(mid_point)
                    
                            
                    print("marking pixel coordinate:",pointsP)
                    
                
                    # Angle=get_bearing(lat1,lon1,lat2,lon2)
                    # print("bearing angle:",Angle)
                    # d=DistanceLatLong(lat1,lon1,lat2,lon2)
                    # print("Distance between two lat long:",d)
                    
                # To Detect the bucket in the frame coordinates   
                        
                    for i in range (len(boxes.xyxy)):
                            # Boxes object for bbox outputs
                            print(boxes)
                            clas = boxes.cls[i].tolist()
                            print(clas)
                            print("bbbbbbbbbbbbbbbbbbb",boxes.xyxy)
                            if clas!=6.0:
                                continue
                                
                            
                            cords = boxes.xyxy[i].tolist()
                            x1, y1, x2, y2 = cords
                            start_point = (int(x1), int(y1))
                            end_point = (int(x2), int(y2))
                            mid_point = (int((start_point[0]+end_point[0])/2),int((start_point[1]+end_point[1])/2))
                            dist= project_disparity_to_3d(frame1, mid_point)
                            
                            dist1=dist[0]
                            x=dist1[0]
                            y=dist1[1]
                            
                            Bucket3Dcoordinates.append(dist1)
                            BucketPoints.append(mid_point)
                
                            if(dist and len(dist)):
                                
                                
                                print(dist[1])
                                
                                pointsB.append(dist[0])
                                
                                print("bucket mid point:",mid_point)
                                
                    
                    overlaylist=[]
                    overlaylistnew = []
                    mapping={}
            
                    for idx, point in enumerate(pointsP):
                                               
                        pointsp=[]
                        #mid_point = QPointF(mid_point[0], mid_point[1])
                        print("Points are:",point)
                        mask,n=get_mask(frame,point)
                        DF = pd.DataFrame(n)
                        a = "data00"+str(idx+1)+".csv"
                        DF.to_csv(a,index = False)

                        #linearregression was created to plot the lines. This is not used anymore and we use PCA to draw the lines.
                        # LinerRegressionOnTPoint(a)
                        slope,constant,minorvalue,minor_y,minor_x=drawMajorandMinorAxis(n,frame)

                        global X11, Y2, X2, Y11
                        # Calculate the end points of the slope
                        X11 = 0  # Start x-coordinate (adjust as needed)
                        Y11 = int(slope * X11 + constant)  # Calculate the y-coordinate using the slope-intercept form
                        global pt1  # First end point
                        pt1 = (X11, Y11)
                        X2 = frame.shape[1]  # End x-coordinate (adjust as needed)
                        Y2 = int(slope * X2 + constant)  # Calculate the y-coordinate using the slope-intercept form
                        global pt2
                        pt2 = (X2, Y2)  # Second end point
                        self.x1=Y11
                        self.y2=X2
                        print("pt1:", pt1)
                        print("pt2:", pt2)
                        print("self.x1:",self.x1)
                        print("self.y2:",self.y2)
                        
                        
                        color = (0,0,255)
                        distance_threshold = 25
                        for mid_point in pointsP:
                            mid_point = QPointF(mid_point[0], mid_point[1])
                            print(mid_point)
                            # Convert QPointF coordinates to integers
                            x = int(mid_point.x())
                            y = int(mid_point.y())
                            print(x)
                            print(y)
                            pointsp.append(mid_point)
                            if self.mid_point is not None:
                                print("mid_point before:",self.mid_point)
                                distance = math.dist((self.mid_point.x(), self.mid_point.y()), (mid_point.x(), mid_point.y()))
                                print(distance)
                                if distance <= distance_threshold:
                                    self.mid_point=mid_point
                                print("mid_point after:",self.mid_point)
                            if self.start_point is not None and self.minor_point is not None and self.mid_point==mid_point and self.mid_point is not None:
                                self.slope = (self.start_point.y() - self.mid_point.y()) / (self.start_point.x() - self.mid_point.x())
                                print("New Slope is:", self.slope)
                                self.constant = self.mid_point.y() - self.slope * self.mid_point.x()
                                print("New Constant is:", self.constant)
                                self.minorvalue = self.minor_point.y() - (self.slope*self.minor_point.x()) - self.constant
                                slope1=self.slope
                                constant1= self.constant
                                minorvalue1=self.minorvalue
                                if slope1 is not None and constant1 is not None and minorvalue1 is not None:
                                    new_overlay,flag,pt1,pt2=UpdateOverlay(frame,color,Marker3DCoordinate,BucketPoints,Bucket3Dcoordinates,slope1,constant1,minorvalue1, pt1, pt2)
                                    new_overlay1=new_overlay
                                    overlaylistnew.append(new_overlay1)
                                    print("Overlaylistnew:",overlaylistnew)
                        
                        overlay,flag,pt1, pt2=createOverlay(frame,color,Marker3DCoordinate,BucketPoints,Bucket3Dcoordinates,slope,constant,minorvalue, pt1, pt2)
                        RoiForBoundary(slope,constant,point,frame1)# insert gps location into db for line persistence

                        overlaylist.append(overlay)                        

                        Flag=flag
                        slope=round(slope,4)
                        constant=round(constant,4)
                        minorvalue=round(minorvalue,4)
                        
                        
                        insertmarkerdata(slope,constant,minorvalue,minor_y,minor_x,point[0],point[1],count,lat_long_dig[idx][0],lat_long_dig[idx][1],lat1,lon1)

                    pointsP.append(mid_point)
                    
                    
                    # DistanceBetweenMarkingAndBucket(frame,points,pointsB)
                    # DistanceBetweenMarkings(frame,points,pointsP)
                    
                    result=FetchData()
                    print("database results:",result)
                    
                    pointsP=[]
                    Bucket3Dcoordinates=[]
                    Marker3DCoordinate=[]
                    BucketPoints=[]
                    lat_long_dig=[]
                    overlay=frame.copy()
                    new_overlay=frame.copy()
                    
                    label_text = "{}"

                    if self.deletion_var is not None:
                        for index in sorted(self.deletion_var, reverse=True):
                            if len(pointsp) > 0 and index < len(pointsp):
                                pointsp.pop(index)  # Remove the item at the specified index from pointsp

                            if len(overlaylist) > index:
                                overlaylist.pop(index)  # Remove the overlay from overlaylist

                            # Remove the corresponding key from the mapping dictionary
                            key = f"key_{index}"
                            if key in mapping:
                                del mapping[key]
                    
                    paired_items = []
                    # Map the midpoint and overlay with the key in the dictionary
                    for idx, (overlay, mid_point) in enumerate(zip(overlaylist, pointsp)):
                        # Generate a unique key using the index
                        key = f"key_{idx}"
                        
                        mapping[key] = {
                            'mid_point': mid_point,
                            'overlay': overlay
                        }

                        # Append the midpoint and overlay as a tuple to the paired_items list
                        paired_items.append((mid_point, overlay))

                    # Sort the paired_items list based on the first midpoint
                    paired_items.sort(key=lambda pair: pair[0].x(), reverse=True)
                    print("Sorted list:", paired_items)
                    overlaylistnew.append(new_overlay)
                    for new_overlay in overlaylistnew:
                        print("This is the new_overlay in overlaylist", new_overlay)
                        cv2.addWeighted(new_overlay, alphanew, frame, 1 - alphanew, 0, frame)
                    
                    # Iterate over the sorted mapping list
                    for idx, (mid_point, overlay) in enumerate(paired_items):
                        key = f"key_{idx}"
                        # Update the mapping dictionary with the sorted overlay
                        mapping[key]['overlay'] = overlay
                        #print(mid_point)      
                        x = int(mid_point.x())
                        y = int(mid_point.y())
                        print(mid_point)
                        # Add label text to the overlay image at the calculated position
                        label = label_text.format(idx)
                        cv2.putText(overlay, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=2)

                        # Find the index of the midpoint in pointsp
                        if self.mid_point is not None:
                            print("mid_point before:",self.mid_point)
                            distance = math.dist((self.mid_point.x(), self.mid_point.y()), (mid_point.x(), mid_point.y()))
                            print(distance)
                            if distance <= distance_threshold:
                                self.mid_point=mid_point
                                index = pointsp.index(self.mid_point)
                                print(index)
                                # Delete the overlay at the given index from overlaylist
                                if index < len(overlaylist):
                                    paired_items[idx] = (mid_point, None)
                                    overlaylist[index] = None
                                print("mid_point after:", self.mid_point)
    
                    paired_items = [(mid_point, overlay) for (mid_point, overlay) in paired_items if overlay is not None]
                    overlaylist = [overlay for overlay in overlaylist if overlay is not None]

                        # Iterate over the sorted mapping list
                    for idx, (mid_point, overlay) in enumerate(paired_items):
                        key = f"key_{idx}"
                        # Update the mapping dictionary with the sorted overlay
                        mapping[key]['overlay'] = overlay
                        #print(mid_point)      
                        x = int(mid_point.x())
                        y = int(mid_point.y())
                        print(mid_point)
                        # Add label text to the overlay image at the calculated position
                        label = label_text.format(idx)
                        cv2.putText(overlay, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=2)

                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                    print(frame)
                # if (Flag==True):
                #     freq=2000
                #     dur=2000
                #     winsound.Beep(freq,dur)
                
                # Convert the frame to RGB format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Create a Qt image from the frame
                qt_image = QtGui.QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0],
                                            QtGui.QImage.Format_RGB888)

                
                frame_height, frame_width, _ = frame.shape

                self.view.setFixedSize(frame_width, frame_height)

                # Create a pixmap from the Qt image
                pixmap = QtGui.QPixmap.fromImage(qt_image)

                # Set the pixmap on the QGraphicsScene
                self.scene.clear()
                self.scene.addPixmap(pixmap)

                self.button_start.setEnabled(False)
                self.button_stop.setEnabled(True)

                # Update the GUI
                QtWidgets.QApplication.processEvents()        

        return None

# Class_name : LoadingThread
# Description : A loading thread to allow the connections to occur seamlessly
# Functions : run
# return : None

class LoadingThread(QThread):
    loadingFinished = pyqtSignal()  # Signal emitted when loading is finished
    progressChanged = pyqtSignal(int)  # Signal emitted to update the progress bar value

    def run(self):
        # Simulate loading process
        for i in range(101):
            time.sleep(0.0)  # Adjust the delay as needed
            self.progressChanged.emit(i)
        self.loadingFinished.emit()

# Class_name : CustomProgressBar
# Description : A QProgressBar widget, that centres itself to the screen to show a loading bar with custom animation
# Functions : __init__, paintEvent
# return : None

class CustomProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.background_image = QPixmap(r"C:\Users\BigBoa\TeamCity_Nodig\Nodig\cobra.jpg")

    #Function to animate the progress bar fill with a label to show its percentage, and show the background as the cobravision logo
    def paintEvent(self, event):
        painter = QPainter(self)
        # Draw the background image
        painter.drawPixmap(self.rect(), self.background_image)
        # Draw the progress bar value
        progress_rect = self.rect()
        progress_rect.setWidth(int(progress_rect.width() * self.value() / (self.maximum() - self.minimum())))
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QColor(0, 0, 0))
        painter.drawRect(progress_rect)
        # Draw the progress bar animation
        progress_anim_rect = self.rect()
        progress_anim_rect.setWidth(int(progress_anim_rect.width() * self.value() / (self.maximum() - self.minimum())))
        opacity = 0.5  # Set the desired opacity value (0.0 - fully transparent, 1.0 - fully opaque)
        painter.setOpacity(opacity)
        painter.fillRect(progress_anim_rect, QColor(0, 128, 0))

        # Draw the progress bar value text
        font = QFont()
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        text = f"{self.value()}%"
        text_width = painter.fontMetrics().width(text)
        text_height = painter.fontMetrics().height()
        text_x = (self.width() - text_width) // 2
        text_y = (self.height() + text_height) // 2
        painter.drawText(text_x, text_y, text)

# Class_name : LoadingApplication
# Description : The main loading widget that holds things in place
# Functions : __init__, initUI, update_progress, loading_finished, centre_window
# return : None

class LoadingApplication(QApplication):
    def __init__(self, args):
        super().__init__(args)
        self.loading_window = QMainWindow()
        self.initUI()

    #Function to initialise the window for the loading bar
    def initUI(self):
        self.loading_window.setWindowTitle("Loading")
        self.loading_window.setWindowFlags(self.loading_window.windowFlags() | Qt.FramelessWindowHint)
        
        # Create a custom progress bar
        self.loading_bar = CustomProgressBar(self.loading_window)
        self.loading_bar.setRange(0, 100)
        self.loading_bar.setMinimumSize(250, 25)  # Set the minimum size of the loading bar

        self.loading_window.setCentralWidget(self.loading_bar)

        # Start the loading thread
        self.loading_thread = LoadingThread()
        self.loading_thread.loadingFinished.connect(self.loading_finished)
        self.loading_thread.progressChanged.connect(self.update_progress)
        self.loading_thread.start()

        self.loading_window.resize(300, 100)  # Set the size of the loading window
        self.center_window(self.loading_window)  # Center the loading window on the screen

        self.loading_window.show()

    #Function to update the loading progress value
    def update_progress(self, value):
        self.loading_bar.setValue(value)

    #Function to start the main application once the loading is completed
    def loading_finished(self):
        # Open the main application here
        self.main_application = IncabMonitor()
        self.main_application.showMaximized()
        self.loading_window.close()

    #Function to centre the loading bar to the window
    def center_window(self, window):
        # Center the window on the screen
        frame_geometry = window.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        frame_geometry.moveCenter(center_point)
        window.move(frame_geometry.topLeft())

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Configure the serial port settings
port = "COM5"  # Replace with the appropriate serial port on your system
baudrate = 9600
alpha=0.3
alphanew=0.3
Marker3DCoordinate=[]
Bucket3Dcoordinates=[]
BucketPoints=[]
lat_long_dig = []

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    loading_dialog = LoadingApplication(sys.argv)
    if loading_dialog.exec_() == QDialog.Accepted:
        sys.exit(app.exec_())