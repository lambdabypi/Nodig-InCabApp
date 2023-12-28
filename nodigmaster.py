# ******************************************************************************************************************
#  * FileName:     nodigmaster.py
#  * Project:      No Dig
#  * Developed by: Chipmonk Technologies Private Limited
#  * Description : This script reads the frame from video and passes into the model for the object detection.
#  * Copyright and Disclaimer Notice Software:
#  *****************************************************************************************************************


# Dependencies
#  ******************************************************************************************************************

from cmath import asin
import pandas as pd
import winsound
import argparse
from roboflow import Roboflow
import cv2
import numpy as np
import time
from math import atan2, cos
from math import sin
import math
from skimage.measure import label, regionprops_table
from scipy.ndimage import median_filter
import torch
import seaborn as sns; sns.set()
import sklearn
from datainsertion import CreateTable,insertmarkerdata,FetchData
from gps_utils import CartesianToGPS,ConvertGPStocartesian
# get_bearing,DistanceLatLong
from ThreeDcoord import project_disparity_to_3d
from distances import DistanceBetweenMarkingAndBucket,DistanceBetweenMarkings
from nodig_region import LinerRegressionOnTPoint,drawMajorandMinorAxis,createOverlay,get_mask


# Variable Declarations
#  ***************************************************************************************************************
Bucket3Dcoordinates=[]
BucketPoints=[]
pointsP=[]
Marker3DCoordinate=[]
lat_long_dig = []
predictx=28
predicty=36
skipframe=200
value=4
Flag=False
alpha=0.3


# Function Details-
# **************************************************************************************************************************
# Function name : main
# Description : It reads the frames and passes into the model for object detection. 
# parameters to be passed : Color video file(colorfile) and disparity video file(disparityfile)
# return : None
# **************************************************************************************************************************

def main(colorfile,disparityfile):
    global pointsP
    global Bucket3Dcoordinates
    global Marker3DCoordinate
    global BucketPoints
    global lat_long_dig
    global alpha
    global predictx
    global predicty
    global Flag
    
    f = cv2.VideoCapture(colorfile)
   
    depth = cv2.VideoCapture(disparityfile)
   
    success = 1
    count = 0
        
    while success:
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
            predictions = model.predict("temp.jpg")
            predictions_json = predictions.json()
            results = model1(frame)
            predictions_json = predictions.json()
           
            points=[]
            pointsB=[]
            
            for bounding_box in predictions_json['predictions']:
                
                if (bounding_box["class"] == "Tolerance"):
                    x0 = bounding_box['x'] - bounding_box['width'] / 2
                    x1 = bounding_box['x'] + bounding_box['width'] / 2
                    y0 = bounding_box['y'] - bounding_box['height'] / 2
                    y1 = bounding_box['y'] + bounding_box['height'] / 2
                    

                    start_point = (int(x0), int(y0))
                    end_point = (int(x1), int(y1))
                    mid_point = (int((start_point[0]+end_point[0])/2),int((start_point[1]+end_point[1])/2))
                             
                    dist= project_disparity_to_3d(frame1, mid_point)
                    print(dist)
                    
# Temporary hard coded (in future we will fetch it from the GPS device)                   
                    lat1 = 12.917816972222222441666666666667
                    lon1 = 77.633101972222227971083333333333
                    
                    lat2 = 12.917817872222222441666666666667
                    lon2 = 77.633102872222227971083333333333                 
                    
                    
# Aligning the image axis with cartesian system axis       
                 
                    x = dist[0][0]
                    y=dist[0][1]
                    z= dist[0][2]
                    
            # After rotating z and y axis by 90 degrees clockwise
                    X=x 
                    Y= z
                    Z = -y
                    
            # After rotating y and x by 90 degree in anticlockwise                    
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
                
            for result in results.xyxy[0]:
                if int(result[-1]) in [predictx,predicty]:
                   
                   
                    x1, y1, x2, y2 = result[:value]
                    box = [int(x1),int(y1),int(x2),int(y2)]
                
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
                        
                        
            print("mid point list",mid_point)            
            overlaylist=[]
            
            if(pointsP==[]):
                result=FetchData()
                for i in range (len(result)):
                    slope=result[i][2]
                    constant=result[i][3]
                    minorvalue=result[i][4]
                
                print("slope region:",slope)
                print("constant region:",constant)
                print("minor value region:",minorvalue)
                
                overlaylist.append( createOverlay(frame,color,Marker3DCoordinate,BucketPoints,Bucket3Dcoordinates,slope,constant,minorvalue))
        
        
    
                
            for idx, point in enumerate(pointsP):
                
                
                mask,n=get_mask(frame,point)
                DF = pd.DataFrame(n)
                a = "data00"+str(idx+1)+".csv"
                DF.to_csv(a,index = False)
                
                #linearregression was created to plot the lines. This is not used anymore and we use PCA to draw the lines.
                # LinerRegressionOnTPoint(a)
                
                slope,constant,minorvalue,minor_y,minor_x=drawMajorandMinorAxis(n,frame)
                
                color = (0,0,255)
                overlay,flag=createOverlay(frame,color,Marker3DCoordinate,BucketPoints,Bucket3Dcoordinates,slope,constant,minorvalue)
    
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
                                   
            for overlay in overlaylist:
    
             cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
             
            cv2.imshow("frame",frame)
            
            # if (Flag==True):
            #     freq=2000
            #     dur=2000
            #     winsound.Beep(freq,dur)
            
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    

    return None
    
    


if __name__ == '__main__':
    
    # load model to detect T marking in frame
    rf = Roboflow(api_key="khzPtVBExR4BSFqdeXjT")
    project = rf.workspace().project("no-dig-segmentation")
    model = project.version(6).model

    # Load model to detect bucket (assumed)  in frame
    model1 = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument("-c", "--color",required=True)
    parser.add_argument("-d", "--disparity",required=True)    
    
    
    args = vars(parser.parse_args())    
    
    colorfile= args['color']
    disparityfile = args['disparity']

# Calling the main function
    main(colorfile,disparityfile)

    
    
    
    
        