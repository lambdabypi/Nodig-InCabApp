from cmath import asin
from operator import itemgetter
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image, ImageTk
from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import time
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
from pytestqt.qtbot import QtBot
import unittest
from PyQt5.QtTest import QTest
import pytest
import math
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QPropertyAnimation
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QDialog, QLineEdit, QMessageBox, QFrame, QToolButton
from PyQt5.QtGui import QRegExpValidator, QIcon
from PyQt5.QtCore import QRegExp
from PyQt5.QtWidgets import QMainWindow

#from miantest import test_video_player_open_and_responds_correctly

# function to create a table in DB
def CreateTable():
    conn = psycopg2.connect(
    database="No_dig", user='postgres', password='12345', host='127.0.0.1', port= '5432'
    )
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS MarkerData")

    #Creating table as per requirement
    sql ='''CREATE TABLE MarkerData(
    FRAME_ID INT ,
    OBJECT CHAR(50),
    SLOPE CHAR(50) ,
    CONSTANT CHAR(50) ,
    MINOR_VALUE CHAR(50) ,
    MIDPOINTX CHAR(20) ,
    MIDPOINTY CHAR(20) ,
    LAT CHAR(50),
    LONG CHAR(50),
    CAMERA_LAT CHAR(50),
    CAMERA_LONG CHAR(50)
    
    )'''
    cursor.execute(sql)
    
    conn.commit()
    conn.close()
    
    return None

# finction to insert data in DB
def insertmarkerdata(slope,constant,minorValue,pointx,pointy,count,lat,long,camera_lat,camera_long):
    print("data to insert:",pointx,pointy)
    conn = psycopg2.connect(
    database="No_dig", user='postgres', password='12345', host='127.0.0.1', port= '5432'
    )
    
    cursor = conn.cursor()
    
    
    cursor.execute("INSERT INTO MarkerData VALUES('%d','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s')" % (count,"marking",slope, constant, minorValue,pointx,pointy,lat,long,camera_lat,camera_long))
    print("data inserted successfully")

    conn.commit()
    conn.close()
    
    return None
    
    

# function to extract the T point from image
def get_mask(img,point):
    y,x=point
    neighbors = []
    area=50
    img = cv2.bilateralFilter(img, 11, 75, 75)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_color_range1 = (0, 50, 50)
    upper_color_range1 = (5, 255, 255)

    lower_color_range2 = (160, 50, 50)
    upper_color_range2 = (180, 255, 255)

    mask1 = cv2.inRange(hsv, lower_color_range1, upper_color_range1)
    mask2 = cv2.inRange(hsv, lower_color_range2, upper_color_range2)

    mask = cv2.bitwise_or(mask1, mask2)
    

    show=mask[x-area:x+area+1,y-area:y+area+1]
    # cv2.imshow("mask",show)
    # cv2.waitKey(0)
   
    for i in range(x-area,x+area+1):
        for j in range(y-area,y+area+1):
                
                try:
                    if(mask[i,j]):
                        neighbors.append((j,720-i))
    
                except:
                     pass
            
            
    
    return mask,neighbors







# Now, Lets import the data set
def LinerRegressionOnTPoint(file):
    plt.clf()
    dataset = pd.read_csv(file)
   
    print(dataset.shape)
    

    distance = dataset.iloc[:, :-1].values
    
    fare = dataset.iloc[:, -1].values


    # Splitting the dataset into the Training set and Test set

    distance_points_train, distance_points_test, fare_train, fare_test = \
    train_test_split(distance, fare, test_size = 0.10, random_state = 0)



    # Fitting Simple Linear Regression Model to the taxi fare training set

    regressor = LinearRegression()
    regressor.fit(distance_points_train, fare_train)


    regressor.coef_
    regressor.intercept_

    # Predicting the Test set results
    fare_pred = regressor.predict(distance_points_test)
    print("Sum of squared error is: " , np.sum(np.square(fare_test-fare_pred)))

    # SSE is not a good measure for measuring the error in prediction
    #it depends on no of predictions!!
    #MSE - Squared error per observation
    print("Mean of squared error is: " , np.sum(np.square(fare_test-fare_pred))/len(fare_test))

    #RMSE - Estimated absolute error in the prediction per observation
    print("Root Mean squared error is : " , np.sqrt(np.sum(np.square(fare_test-fare_pred))/len(fare_test)))

    from sklearn.metrics import r2_score
    print("R squared coeff is: ",r2_score(fare_test, fare_pred) )

    # Visualising the Training set results
    plt.scatter(distance_points_train, fare_train, color = 'red')
    plt.plot(distance_points_train, regressor.predict(distance_points_train), color = 'blue')
    plt.title('Distance Vs fare (Training set)')
    plt.xlabel('Dsitance')
    plt.ylabel('Fare')
    plt.show()

    # Visualising the Test set results
    plt.scatter(distance_points_test, fare_test, color = 'red')
    plt.plot(distance_points_test, fare_pred, color = 'blue')
    plt.title('Distance Vs Fare(Test set)')
    plt.xlabel('Distance')
    plt.ylabel('Fare')
    plt.show()
    
    
    
    # return None
        

# function to dram Major and Minor axis on T point
def drawMajorandMinorAxis(n,frame,index,clsss):
    
    rng = np.random.RandomState(1)
    X=np.array(n)

    plt.scatter(X[:, 0], X[:, 1])
    plt.axis('equal');

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    print(X)
    print(X.shape)

    print(pca.components_)

    print(pca.explained_variance_)

    # def draw_vector(v0, v1, ax=None):
    #     ax = ax or plt.gca()
    #     arrowprops=dict(arrowstyle='->',
    #                     linewidth=2,
    #                     color='red',
    #                     shrinkA=0, shrinkB=0)
    #     ax.annotate('', v1, v0, arrowprops=arrowprops)


    # plot data
    plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
    print("+++++++++++++++++++++++++++")
    print("hurrayyyyyyyyyyyyy:",pca.explained_variance_,pca.components_)
    
    maxlength=max(pca.explained_variance_)
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector  * np.sqrt(abs(length)*5)
        # v=-v
        print("vector:",vector)
        
            
        print("output vector",v)
        # draw_vector(pca.mean_, pca.mean_ + v)
        print("maxlength:",maxlength)
        if(length!=maxlength):
            print("minor axis length:",length)
            m1= int(pca.mean_[0]+v[0])
            m2 = (720-(int(pca.mean_[1]+v[1])))
            minor_axis_vector = (m1,m2)
        else:
            
            print("major axis length:",length)
            p1=(int(pca.mean_[0]),720-int(pca.mean_[1]))
            major_axis_vector = (int(pca.mean_[0]+v[0]),720-(int(pca.mean_[1]+v[1])))
            # cv2.arrowedLine(frame,p1,major_axis_vector,color=(0,0,0),thickness=2,line_type=5)
            
        p1=(int(pca.mean_[0]),720-int(pca.mean_[1]))
        p2 = (int(pca.mean_[0]+v[0]),720-(int(pca.mean_[1]+v[1])))
       
        
        # slope = (p1[1]-p2[1])/(p1[0]-p2[0])
        # constant = p2[1]-(slope*p2[0])
        
        # print("points",p1,p2)
        # cv2.line(frame, (p1[0],int((slope*p1[0])+constant)),(p2[0],int((slope*p2[0])+constant)),(0,255,0))
    if clsss==0.0:
        direction="D"
    elif clsss==1.0:
        direction="U"
    elif clsss==2.0:
        direction="R"
    else:
        direction="L"
    
    
        

    p1=(int(pca.mean_[0]),720-int(pca.mean_[1]))
    major_slope = (p1[1]-major_axis_vector[1])/(p1[0]-major_axis_vector[0])
    major_constant = major_axis_vector[1]-(major_slope*major_axis_vector[0])
    value = m2-(major_slope*m1)-major_constant
    plt.show()
    
    
        

    return (major_slope,major_constant,direction)


def drawMinoraxis(n,frame):
     
    X=np.array(n)
    
    # X=np.

    # X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
    plt.scatter(X[:, 0], X[:, 1])
    plt.axis('equal');

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
    
    p1=(int(pca.mean_[0]),720-int(pca.mean_[1]))
    
    
    return p1
    
            
        

# function to convert GPS coordinate into cartesian coordinate
def ConvertGPStocartesian(lat,lon):
    R= 6371000
    lon =  lon * math.pi / 180
    lat = lat * math.pi / 180
    x = R * cos(lat) * cos(lon)

    y = R * cos(lat) * sin(lon)

    z = R *sin(lat)
    
    return x,y,z

# function to convert cartesian coordinate to GPS coordinate
def CartesianToGPS(x,y,z):
   R= 6371000
   lat = asin(z / R)
   lon = atan2(y, x)
   lat = lat*180/math.pi
   lon = lon*180/math.pi
   
   return lat,lon


# function to campute distance between marking and bucket
def DistanceBetweenMarkingAndBucket(points,pointsB):
       
    for i in range(len(points)):
        for j in range(len(pointsB)):
    
        # print(pts)
            print(points)
            print(pointsP)
            
            distance = (((points[i][0]-pointsB[j][0])**2+(points[i][1]-pointsB[j][1])**2+(points[i][2]-pointsB[j][2])**2)**0.5)
                    
                
                
    return None
# function to compute distance between markings                   
def DistanceBetweenMarkings(points,pointsP):
    
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            
            # print(pts)
            print(points)
            
            distance = (((points[i][0]-points[j][0])**2+(points[i][1]-points[j][1])**2+(points[i][2]-points[j][2])**2)**0.5)
            
    return None
            

# function to calculate 3-d coordinates of marking and bucket
def project_disparity_to_3d(disparity, boundingbox, rgb=[]):

    points = []

    f = 798.4660034179688
    B = .075

    height, width = 720,1280

    print(boundingbox)
    y= boundingbox[1]-1
    x=boundingbox[0]-1
    X=0
    Y=0
    Z=0
    
    print("disparity image resolution:",disparity.shape)
    print("disparity value:",disparity[y,x])
    
    if (disparity[y,x] > 0):

        
        Z = (f * B) / disparity[y,x]
        
        X = ((x) * Z) / f
        Y = ((y ) * Z) / f
        
    points.append([X,Y,Z])
    
    points.append( (X*2+Y2+Z*2)**0.5)
    return points
    

# function to compute the No dig region
def createOverlay(frame,colour,Marker3DCoordinate,Bucket3Dcoordinates,slope,constant,minorvalue):
    print("bucket 3d coordinates",Bucket3Dcoordinates)
    print("marker y coordinates:",Marker3DCoordinate)
    minYvalue = min(Marker3DCoordinate)
    
    if (Bucket3Dcoordinates!=[] and Marker3DCoordinate!=[]):
        bucketHeight = abs(minYvalue-Bucket3Dcoordinates[0][1])
        bucketHeightinfeet = bucketHeight*3.28084
        
    print("maximum value:",minYvalue)
    if (Bucket3Dcoordinates!=[]):
        x=Bucket3Dcoordinates[0][0]*3.28084
        y= Bucket3Dcoordinates[0][1]*3.28084
        z= Bucket3Dcoordinates[0][2]*3.28084
    overlay1=frame.copy()
    for w in range(1280):
      for h in range(720):
        pixelvalue=int(h-(slope*w)-constant)
        if(pixelvalue*minorvalue>0):
    #         # overlay=frame.copy()
            cv2.circle(overlay1,(w,h),2,colour)
    #         # print("1")
            
    
    if (BucketPoints!=[]):
        print("Bucket pixel coordinate",BucketPoints)
        print(BucketPoints[0][0])
    
        Bucket_value=int(BucketPoints[0][1]-(slope*BucketPoints[0][0])-constant)
        print("bucket value:",Bucket_value)
        if (Bucket_value*minorvalue>0):
            print("+++++++++++++++++++++++++Alert : Bucket in No dig Zone")
            
            # cv2.putText(
            #             frame,
            #             str(round(bucketHeightinfeet,2))+" ft.",
            #             BucketPoints[0],
            #             fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale = 0.6,
            #             color = (0, 0, 0),
            #             thickness=2
            #         )
            
            if ((abs(bucketHeightinfeet))<=2.5):
                print("+++++++++++++++ALERT:Bucket is below 1 feet of the ground. Bucket is at a height",(abs(y)*3.28084),"feet")
                
                cv2.putText(
                            frame,
                            ("Alert: Bucket is in NO dig Zone below 1 foot"),
                            (600,300),
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale = 0.6,
                            color = (0, 0, 255),
                            thickness=2
                        )
                
    return overlay1
@pytest.fixture
def image_file():
    # Provide the path to a sample image file for testing
    return r"C:\Users\BigBoa\TeamCity_Nodig\Nodig\runs\detect\predict2535"

def test_yolov8_prediction(image_file):
    # Import necessary libraries and modules here
    from ultralytics import YOLO

    model_file_path = r'C:\Users\BigBoa\Downloads\best5623.pt'
    name = 'yolov8n_custom9'
    
    # Perform the prediction
    model = YOLO(model_file_path)
    boxes = model.predict(image_file, save=True, imgsz=640, conf=0.5)
    
    # Perform the assertion to validate the result
    assert len(boxes) > 0

def yolov8_prediction(image_file):
    model_file_path = r'C:\Users\BigBoa\Downloads\best5623.pt'
    name = 'yolov8n_custom9'
    model = YOLO(model_file_path)
    results = model.predict(image_file, save=True, imgsz=640, conf=0.5)
    print("model_results",results[0])
    # print(results.xyxy[0])
    # print(len(results.boxes))
    # annotated_frame = results[0].plot()
        # Display the annotated frame
    # cv2.imshow("YOLOv8 Inference", annotated_frame)
    # cv2.waitKey(1)
    # Resize the frame to fit the windo
    for result in results:
        boxes = result.boxes
        
    return boxes

def overlaycreate(frame,lineEquation):
    overlay=frame.copy()
    for w in range(1280):
        for h in range(720):
            flag = False
            for (slope,constant,direction) in lineEquation:
                dire=1
                if((slope<0 and direction=="L")):
                    dire=-1
                if((slope>0 and direction=="D")):
                    dire=-1
                if ((slope>0 and direction=="R")):
                    dire=-1
                if((dire<0 and h<((slope*w)+constant))or (dire>0 and h>((slope*w)+constant))):
                    #  if((d[direction]<0 and h>((slope*w)+constant))or (d[direction]>0 and h>((slope*w)+constant))):
                # if( h<((equation[0]*w)+equation[1])):    
                    flag=True
                    break
            if(flag):
                # print("fuhldjdsljldsk;hf;khk;kdsahfksadhfsakdhfk")
                # overlay=frame.copy()
                cv2.circle(overlay,(w,h),2,(0,0,255))
                
    return overlay

def Crop_and_resize(img, x1,y1,x2,y2):
    cropped_img = img[y1:y2,x1:x2]
    resized_img = cv2.resize(cropped_img, (180,180))
    return resized_img

def coordinates(lat1,lon1,x,y):
    x=x/(6378160)
    y=y/(6378160)
    lon1 =  lon1 * math.pi / 180
    lat1 = lat1 * math.pi / 180
    lat2 = lat1+y
    lon2 = (x/cos(0.5 * (lat2 + lat1)))+lon1
    lat2 = lat2*180/math.pi
    lon2 = lon2*180/math.pi

    return lat2,lon2

def test_coordinates():
    # Test case 1
    lat1 = 12.917816972222222441666666666667
    lon1 = 77.633101972222227971083333333333
    x = 100
    y = 200
    expected_lat2 = 12.919080077661158
    expected_lon2 = 77.63319788030119
    actual_lat2, actual_lon2 = coordinates(lat1, lon1, x, y)
    print("Actual Output:", actual_lat2, actual_lon2)
    assert (actual_lat2, actual_lon2) == (expected_lat2, expected_lon2)

    # Test case 2
    lat1 = 0
    lon1 = 0
    x = 500
    y = 1000
    expected_lat2 = 0.014214749830793785
    expected_lon2 = 0.023204621051355292
    actual_lat2, actual_lon2 = coordinates(lat1, lon1, x, y)
    print("Actual Output:", actual_lat2, actual_lon2)
    assert (actual_lat2, actual_lon2) == (expected_lat2, expected_lon2)


@pytest.fixture

def video_player(qtbot):
    player = VideoPlayer()
    qtbot.addWidget(player)
    return player


def test_video_player_open_and_responds_correctly(qtbot, video_player):
    # Assert that the player is initially not visible
    assert not video_player.isVisible()

    # Simulate showing the player
    video_player.show()

    # Wait for the player to be exposed
    qtbot.waitUntil(video_player.isVisible)

    # Assert that the player is now visible
    assert video_player.isVisible()
    print("Application opens correctly")

class CodeInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enter Code")
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
        
    def update_code(self):
        self.code = ""
        for digit_input in self.digit_widgets:
            self.code += digit_input.text()

    def ok_button_clicked(self):
        predefined_code = "1234"  # Replace with your predefined code

        entered_code = self.code_input.text()
        print("Entered Code:", entered_code)
        print("Predefined Code:", predefined_code)

        if entered_code == predefined_code:
            # The entered code is correct
            self.accept()
        else:
            # The entered code is incorrect
            QMessageBox.warning(self, "Invalid Code", "Please enter a valid code.")
            
    def create_digit_input(self):
        for _ in range(4):
            digit_input = DigitInput()
            self.digit_widgets.append(digit_input)

    def digit_input_layout(self):
        layout = QHBoxLayout()
        for digit_widget in self.digit_widgets:
            layout.addWidget(digit_widget)
        return layout
    
    def clear_input(self):
        for digit_widget in self.digit_widgets:
            digit_widget.clear()

class DigitInput(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.setFixedWidth(50)
        self.setMaxLength(1)
        self.setAlignment(Qt.AlignCenter)

        validator = QRegExpValidator(QRegExp(r"\d{0,1}"))  # Restrict input to a single digit
        self.setValidator(validator)

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

    def number_clicked(self):
        clicked_button = self.sender()
        code_input = self.parent().digit_widgets[0]
        for self.digit_input in self.parent().digit_widgets:
            if self.digit_input.text() == "":
                code_input = self.digit_input
                break
        code_input.setText(clicked_button.text())

class ImageButton(QPushButton):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 50)
        self.setStyleSheet(f"QPushButton {{ border-image: url({image_path}); }}")
        self.setIconSize(self.size())
        
class IncabMonitor(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("In Cab Monitor")

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

        # Create a label to display the video frame
        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        # Create a button to allow editing
        #self.button_edit = ImageButton("icons8-settings-48.png")
        self.button_edit = QToolButton()
        self.button_edit.setIcon(QIcon('icons8-settings-48.png'))
        self.button_edit.clicked.connect(self.button_clicked)
        self.button_edit.setFixedSize(100, 50)

        # Create a button to start the video playback
        self.button_start = QtWidgets.QPushButton(self)
        self.button_start.setText("Start")
        self.button_start.clicked.connect(self.display_frame)
        self.button_start.setFixedSize(100, 50)

        # Create a button to stop and exit the process
        self.button_stop = QtWidgets.QPushButton(self)
        self.button_stop.setText("Stop and Exit")
        self.button_stop.clicked.connect(self.stop_and_exit)
        self.button_stop.setEnabled(False)  # Disable initially
        self.button_stop.setFixedSize(100, 50)

        # Create extra buttons on the left and right sides
        self.button_1 = QtWidgets.QPushButton(self)
        self.button_1.setText("Extra 1")
        self.button_1.setFixedSize(100, 50)

        self.button_2 = QtWidgets.QPushButton(self)
        self.button_2.setText("Extra 2")
        self.button_2.setFixedSize(100, 50)

        self.button_3 = QtWidgets.QPushButton(self)
        self.button_3.setText("Extra 3")
        self.button_3.setFixedSize(100, 50)

        self.button_start.setAttribute(QtCore.Qt.WA_AcceptTouchEvents)
        self.button_stop.setAttribute(QtCore.Qt.WA_AcceptTouchEvents)
        self.label.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.label.grabGesture(Qt.PinchGesture)
        self.label.grabGesture(Qt.SwipeGesture)

        # Set up the layout
        main_layout = QtWidgets.QGridLayout(self)  # Use QGridLayout for more control
        #main_layout.addWidget(self.button_1, 0, 0)  # Add button_1 to top-left corner
        #main_layout.addWidget(self.button_2, 0, 2)  # Add button_2 to top-right corner
        #main_layout.addWidget(self.button_3, 0, 1)  # Add button_3 to the middle
        edit_layout = QHBoxLayout()
        edit_layout.addWidget(self.button_edit)
        edit_layout.addStretch()
        main_layout.addLayout(edit_layout, 1, 0, 1, 3)
        main_layout.addWidget(self.label, 2, 0, 1, 3)  # Add label widget below the buttons, span it across three columns

        button_layout = QtWidgets.QHBoxLayout()  # Create a horizontal layout for the start and stop buttons
        button_layout.addWidget(self.button_start)
        button_layout.addWidget(self.button_stop)

        main_layout.addLayout(button_layout, 3, 0, 1, 3)  # Add button layout below the label, span it across three columns
        self.setLayout(main_layout)
        self.hide()  # Hide the player initially

    def button_clicked(self):
        code_input_dialog = CodeInputDialog(self)
        if code_input_dialog.exec_() == QDialog.Accepted:
            code = code_input_dialog.code_input.text()
            if len(code) == 4:
                number_panel = NumberPanel(self)
                number_panel.show()
            else:
                QMessageBox.warning(self, "Invalid Code", "Please enter a 4-digit code.")

    def touchEvent(self, event):
        # Handle touch events here
        if event.type() == QtCore.QEvent.TouchBegin:
            # Handle touch begin event
            pass
        elif event.type() == QtCore.QEvent.TouchUpdate:
            # Handle touch update event
            pass
        elif event.type() == QtCore.QEvent.TouchEnd:
            # Handle touch end event
            pass

    def event(self, event):
        if event.type() == QtCore.QEvent.TouchBegin or \
                event.type() == QtCore.QEvent.TouchUpdate or \
                event.type() == QtCore.QEvent.TouchEnd:
            self.touchEvent(event)

        return super().event(event)
    
    def stop_and_exit(self):
        self.button_start.setEnabled(True)
        self.button_stop.setEnabled(False)        
        sys.exit()
    
    
    def display_frame(self):
        f = cv2.VideoCapture('color.mp4')
        
        depth = cv2.VideoCapture('disparity.mp4')
    
        success = 1
        count = 0
            
        while success:
            count+=1

            ret, frame = f.read()
            frame = cv2.resize(frame, (1280,720))
            cv2.imwrite('temp.png', frame)
            # frame = cv2.resize(frame, (1280,720))
            
            ret1, frame1 = depth.read()
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            
            if(count==550):
                CreateTable()
            if(count<100):
                continue
            # break
            if ret == True:
                                    # frame = cv2.resize(frame, (1280,720))
                    cv2.imwrite('temp.png', frame)
                    # if (count==482):
                    #  cv2.imwrite("distance//frame%d.png" % count, frame)  
                    # frame = cv2.imread('temp.jpg')
                    frame = cv2.imread('temp.png')
                    # frame = cv2.resize(frame, (1280,720))
                    boxes = yolov8_prediction(frame)
                    # continue
                    results=model(frame)
                    points=[]
                    pointsB=[]
                    pointsP=[]
                    classes=[]
                    overlaylist=[]
                    
                    for i in range (len(boxes.xyxy)):
                        # Boxes object for bbox outputs
                        print(boxes)
                        clas = boxes.cls[i].tolist()
                        print(clas)
                        print("bbbbbbbbbbbbbbbbbbb",boxes.xyxy)
                        if clas==4.0:
                            
                            continue
                        cords = boxes.xyxy[i].tolist()
                        x1, y1, x2, y2 = cords
                        print(x1,y1,x2,y2)
                        slope= (y2-y1/x2-x1)
                        start_point = (int(x1), int(y1))
                        height=frame.shape[0]
                        width=frame.shape[1]
                        lat_long_dig=[]
                        lat1 = 12.917816972222222441666666666667
                        lon1 = 77.633101972222227971083333333333
                        lat2,lon2 = coordinates(lat1,lon1,x1,y1)
                        lat_long_dig.append((lat2,lon2))
                        #cv2.putText(
                            #frame,
                            #str(lat_long_dig),
                            #start_point,
                            #fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                            #fontScale = 0.6,
                            #color = (0, 0, 0),
                            #thickness=2
                            #)
                        # frame = cv2.imread('frame_0175.jpg')
                        if clas==0.0:
                            print("inside down")
                            contours = np.array([ [0,int(y1)], [width,int(y1)], [width,height],[0,height]])
                            # image = np.zeros((200,200))
                            overlay1=frame.copy()
                            cv2.fillPoly(overlay1, pts = [contours], color =(0,0,255))
                            overlaylist.append(overlay1)
                            
                        
                        #     danger_zone = (int(x1), int(y1), int(x2), frame.shape[0])
                        #     safe_zone = (int(x1), 0, int(x2), int(y1))
                        # elif clas==1.0:
                        #     # direction="U"
                        #     overlay2=frame.copy()
                        #     contours = np.array([ [0,int(y2)], [width,int(y2)], [width,0],[0,0]])
                        #     # image = np.zeros((200,200))
                        #     cv2.fillPoly(overlay2, pts = [contours], color =(0,0,255))
                        #     overlaylist.append(overlay2)
                            # danger_zone = (int(x1), 0, int(x2), int(y1))
                            # safe_zone = (int(x1), int(y2), int(x2), frame.shape[0])
                        elif clas==2.0:
                            # direction="R"
                            print("inside right")
                            overlay3=frame.copy()
                            contours = np.array([ [int(x1),height], [int(x1),0], [width,0],[width,height]])
                        #     # image = np.zeros((200,200))
                            cv2.fillPoly(overlay3, pts = [contours], color =(0,0,255))
                            overlaylist.append(overlay3)
                        #     # danger_zone = (int(x2), int(y1), frame.shape[1], int(y2))
                        #     # safe_zone = (0, int(y1), int(x1), int(y2))
                        if clas==3.0:
                            # direction="L"
                            overlay4=frame.copy()
                            contours = np.array([ [int(x2),height], [int(x2),0], [0,0],[0,height]])
                            # image = np.zeros((200,200))
                            cv2.fillPoly(overlay4, pts = [contours], color =(0,0,255))
                            overlaylist.append(overlay4)
                            # danger_zone = (0, int(y1), int(x1), int(y2))
                            # safe_zone = (int(x2), int(y1), frame.shape[1], int(y2))
                        # image=Crop_and_resize(frame,int(x1),int(y1),int(x2),int(y2))
                        # danger_zone = (int(x1), int(y1), int(x2), frame.shape[0])
                        # safe_zone = (int(x1), 0, int(x2), int(y1))
                        # cv2.rectangle(frame, (safe_zone[0], safe_zone[1]), (safe_zone[2], safe_zone[3]),  (10, 245, 10), 2)
                        # cv2.putText(frame, "Safe Region", (safe_zone[0] + 10, safe_zone[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        # cv2.rectangle(frame, (danger_zone[0], danger_zone[1]), (danger_zone[2], danger_zone[3]), (0, 0, 255, 128), -1)
                        # cv2.putText(frame, "Do Dig Region", (danger_zone[0] + 10, danger_zone[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        # cv2.rectangle(frame,(int(x1),int(x2)),(int(x2),int(y2)),(255, 0, 0),2)
                        
                    for overlay in overlaylist:
            
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                    
                    # cv2.imwrite("data//frame%d.jpg" % count, frame)
                    # print("frame number",count)
                    #window.mainloop()
                    #cv2.imshow("show",frame)
                    # cv2.waitKey(0)
                    
                    # Convert the frame to RGB format
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Create a Qt image from the frame
                    qt_image = QtGui.QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0],
                                            QtGui.QImage.Format_RGB888)

                    # Create a pixmap from the Qt image
                    pixmap = QtGui.QPixmap.fromImage(qt_image)

                    # Set the pixmap on the label
                    self.label.setPixmap(pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio))

                    self.button_start.setEnabled(False)
                    self.button_stop.setEnabled(True)
                    
                    # Update the GUI
                    QtWidgets.QApplication.processEvents()
                    continue
                    #print(f'boxes : {boxes} masks: {masks} probs: {probs}')
                    print("++++++++++++++++++++===")
                    
            
                    print(":::::::::::::::::::",x1,y1,x2,y2)
                    
                    
                    
                    start_point = (int(x1), int(y1))
                    end_point = (int(x2), int(y2))
                    mid_point = (int((start_point[0]+end_point[0])/2),int((start_point[1]+end_point[1])/2))
                    dist= project_disparity_to_3d(frame1, mid_point)
                    
                        
                    # predictions_json = predictions.json()
                    # print(predictions)
                    
                    # predictions_json = predictions.json()
                
                    
                    if(dist and len(dist)):
                        print(dist[1])
                        points.append(dist[0])
                        Marker3DCoordinate.append(dist[0][1])
                        
                        pointsP.append(mid_point)
                        classes.append(clas)
                
                    

                    
                            # # Read a line from the serial port
                            # line = ser.readline().decode().strip()

                            # # Check if the line contains valid GPS data
                            # if line.startswith('$GNGGA'):
                            #     # Split the line into individual fields
                            #     data = line.split(',')

                            #     # Extract relevant GPS information
                            #     time = data[1]
                            #     latitude = data[2]
                            #     longitude = data[4]
                            #     altitude = data[9]

                            #     # Print the GPS data
                            #     print("Time:", time)
                            #     print("Latitude:", latitude)
                            #     print("Longitude:", longitude)
                            #     print("Altitude:", altitude)

                            # # Close the serial connection
                            # ser.close()
                        
                            # lat1 = 12.917816972222222441666666666667
                            # lon1 = 77.633101972222227971083333333333
                            
                            # x = dist[0][0]
                            # y=dist[0][1]
                            # z= dist[0][2]
                            
                            # # after rotating z and y axis by 90 degrees clockwise
                            # X=x 
                            # Y= z
                            # Z = -y
                            
                            # # after rotating y and x by 90 degree in anticlockwise
                            
                            # x = -Y
                            # y= X
                            # z= Z
                            
                            # print("coordinates after rotation :", x,y,z)
                            # X,Y,Z=ConvertGPStocartesian(lat1,lon1)
                            
                        
                            
                            # # Adding marking coordinates to camera coordinates
                            # x=X+x 
                            # y=Y+y 
                            # z= Z+z 
                            
                            # lat,lon=CartesianToGPS(x,y,z)
                            
                            # print("marking lat long by approach 1",lat,lon)
                            # lat_long_dig.append((lat,lon))
                            
                            
                            
                    print("marking pixel coordinate:",pointsP)
                    print("class coordinates",classes)
                        
                    for result in results.xyxy[0]:
                        print(result)
                        if int(result[-1]) in [28,36]:
                            print("suitcase",results)
                            x1, y1, x2, y2 = result[:4]
                            box = [int(x1),int(y1),int(x2),int(y2)]
                            score = result[4]
                            label = int(result[-1])
                            print(label)
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
                                
                                
                    # print("mid point list",mid_point)            
                    lineEquation=[]
                    overlaylist=[]
                    # pointsP =[(311,555),(330,183),(673,412),(674,81)]
                    print("enumerate points:",pointsP)
                    for idx, point in enumerate(pointsP):
                        lineEquation=[]
                        
                        
                        mask,n=get_mask(frame,point)
                        n=sorted(n,key=itemgetter(1))
                        # print(n)
                        print("length list:",len(n))
                        
                                
                        
                        # print(n)n[]
                    
                        DF = pd.DataFrame(n)
                        a = "data00"+str(idx+1)+".csv"
                        DF.to_csv(a,index = False)
                        # LinerRegressionOnTPoint(a)
                        
                        lineEquation.append(drawMajorandMinorAxis(n,frame,idx,classes[idx]))
                    
                        
                        # slope,constant,minorvalue,mean1=drawMajorandMinorAxis(n,frame,idx)
                        
                        # color = (0,0,255)
                        # n1=[]
                        # n2=[]
                        # for idx, point in enumerate(n):
                        #     value=(720-point[1])-(slope*point[0])-constant
                        #     if value>0:
                        #         n1.append(point)
                        #     elif value<0:
                        #         n2.append(point)
                        #     else:
                        #         continue
                        # # print("n1111111111111111111111",n1)
                        # # print("n2222222222222222222222",n2)
                                
                        # if (len(n1)>len(n2)):
                            
                        #     # print("mean of shorter arm",b1,b2)
                        
                        #     mean2=drawMinoraxis(n2,frame) 
                        #     # cv2.arrowedLine(frame,mean1,mean2,color=(0,0,0),thickness=2,line_type=5)  
                        #     print("means value",mean1,mean2)
                            
                        # else:
                            
                        #     mean2=drawMinoraxis(n1,frame) 
                            # cv2.arrowedLine(frame,mean1,mean2,color=(0,0,0),thickness=2,line_type=5)   
                                
                        # mean2=drawMinoraxis(n1,frame)
                        # mean3=drawMinoraxis(n2,frame)
                        # # cv2.arrowedLine(frame,mean2,mean3,color=(0,0,255),thickness=2,line_type=5)  
                        
                        
                        
                            
                        # minorvalue=int(mean2[1])-(slope*int(mean2[0]))-constant 
                        overlaylist.append(overlaycreate(frame,lineEquation))
            
                        overlaylist.append( createOverlay(frame,color,Marker3DCoordinate,Bucket3Dcoordinates,slope,constant,minorvalue))
                        # slope=round(slope,4)
                        # constant=round(constant,4)
                        # minorvalue=round(minorvalue,4)
                        
                        # insertmarkerdata(slope,constant,minorvalue,point[0],point[1],count,lat_long_dig[idx][0],lat_long_dig[idx][1],lat1,lon1)
                        
                    # pointsP.append(mid_point)
                    
                    # DistanceBetweenMarkingAndBucket(points,pointsB)
                    # DistanceBetweenMarkings(points,pointsP)
                    
                    # pointsP=[]
                    Bucket3Dcoordinates=[]
                    Marker3DCoordinate=[]
                    BucketPoints=[]
                    lat_long_dig=[]
                    
                                        
                    for overlay in overlaylist:
            
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                    
                    # cv2.putText(
                    #         frame,
                    #         "frame number"+str(count),
                    #         (100,100),
                    #         fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    #         fontScale = 0.6,
                    #         color = (0, 0, 0),
                    #         thickness=2
                    #     )
                    
            
                
                    print("frame number:",count)
                    print("enumerate points:",pointsP)
                    
                    # if (count==558):
                    # cv2.imwrite("dataresult//frame%d.jpg" % count, frame)  
                    cv2.imshow("frame",frame)
                    
                    # except:
                    #  print("no detection")
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break


# rf = Roboflow(api_key="khzPtVBExR4BSFqdeXjT")
# project = rf.workspace().project("no-dig-segmentation")
# model = project.version(6).model

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# file = "best.pt"
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# print(torch.load(file))
# model.load_state_dict(torch.load(file))

Bucket3Dcoordinates=[]
BucketPoints=[]

# Configure the serial port settings
port = "COM5"  # Replace with the appropriate serial port on your system
baudrate = 9600




alpha=0.3


Marker3DCoordinate=[]
lat_long_dig = []
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    player = IncabMonitor()
    player.show()
    sys.exit(app.exec_())