# filename : nodig_region.py
# author : Chipmonk techlnologies.
# Date : 11th April 2023
# Description : this script creates the no dig region in frame




import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ThreeDcoord import project_disparity_to_3d
from gps_utils import CartesianToGPS,ConvertGPStocartesian
from datainsertion import CreateTable,insertmarkerdata,FetchData
import winsound



BucketHeightFromGround=2.5
feet=3.28084

# function_name : get_mask
# Description : extract all the red pixels in T point. 
# parameters : image(ndarray) and mid point(tuple) of T point
# return : red pixel coordinate of T point
def get_mask(img,point):
    height=img.shape[0]
    
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
   
    for i in range(x-area,x+area+1):
        for j in range(y-area,y+area+1):
                
                try:
                    if(mask[i,j]):
                        neighbors.append((j,height-i))
    
                except:
                     pass
            
            
    
    return mask,neighbors


# function_name : LinerRegressionOnTPoint
# Description : Draw the line on T point and gives the slope and constant of line. 
# parameters : file(csv)
# return : None
def LinerRegressionOnTPoint(file):
    dataset = pd.read_csv(file)
   
    print(dataset.shape)
    

    X_axis_points = dataset.iloc[:, :-1].values
    
    Y_axis_points = dataset.iloc[:, -1].values


    # Splitting the dataset into the Training set and Test set

    X_points_train, X_points_test, Y_train, Y_test = \
    train_test_split(X_axis_points, Y_axis_points, test_size = 0.10, random_state = 0)



    # Fitting Simple Linear Regression Model to the taxi fare training set

    regressor = LinearRegression()
    regressor.fit(X_points_train, Y_train)


    regressor.coef_
    regressor.intercept_

    # Predicting the Test set results
    Y_axis_pred = regressor.predict(X_points_test)
    print("Sum of squared error is: " , np.sum(np.square(Y_test-Y_axis_pred)))

    # SSE is not a good measure for measuring the error in prediction
    #it depends on no of predictions!!
    #MSE - Squared error per observation
    print("Mean of squared error is: " , np.sum(np.square(Y_test-Y_axis_pred))/len(Y_test))

    #RMSE - Estimated absolute error in the prediction per observation
    print("Root Mean squared error is : " , np.sqrt(np.sum(np.square(Y_test-Y_axis_pred))/len(Y_test)))

    from sklearn.metrics import r2_score
    print("R squared coeff is: ",r2_score(Y_test, Y_axis_pred) )

    # Visualising the Training set results
    plt.scatter(X_points_train, Y_train, color = 'red')
    plt.plot(X_points_train, regressor.predict(X_points_train), color = 'blue')
    plt.title('Distance Vs fare (Training set)')
    plt.xlabel('Dsitance')
    plt.ylabel('Fare')
    plt.show()

    # Visualising the Test set results
    plt.scatter(X_points_test, Y_test, color = 'red')
    plt.plot(X_points_test, Y_axis_pred, color = 'blue')
    plt.title('Distance Vs Fare(Test set)')
    plt.xlabel('Distance')
    plt.ylabel('Fare')
    # plt.show()
    
    
    return None

# function_name : drawMajorandMinorAxis
# Description : Draw the major and minor axis on T point. 
# parameters :  red pixel coordinates of T point(list) and frame(ndarray)
# return : slope,constant and minor value
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def drawMajorandMinorAxis(n, frame):
    height = frame.shape[0]

    rng = np.random.RandomState(1)
    X = np.array(n)

    plt.scatter(X[:, 0], X[:, 1])
    plt.axis('equal')

    pca = PCA(n_components=2)
    pca.fit(X)

    maxlength = np.max(pca.explained_variance_)

    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * np.sqrt(abs(length) * 20)
        if length < maxlength:
            m1 = int(pca.mean_[0] + v[0])
            m2 = height - int(pca.mean_[1] + v[1])
            minor_axis_vector = (m1, m2)
        else:
            major_axis_vector = (int(pca.mean_[0] + v[0]), height - int(pca.mean_[1] + v[1]))

    p1 = (int(pca.mean_[0]), height - int(pca.mean_[1]))
    major_slope = (p1[1] - major_axis_vector[1]) / (p1[0] - major_axis_vector[0])

    major_constant = major_axis_vector[1] - (major_slope * major_axis_vector[0])
    value = m2 - (major_slope * m1) - major_constant

    # Calculate the end points of the slope
    x1 = 0  # Start x-coordinate (adjust as needed)
    y1 = int(major_slope * x1 + major_constant)  # Calculate the y-coordinate using the slope-intercept form
    pt1 = (x1, y1)  # First end point

    x2 = frame.shape[1]  # End x-coordinate (adjust as needed)
    y2 = int(major_slope * x2 + major_constant)  # Calculate the y-coordinate using the slope-intercept form
    pt2 = (x2, y2)  # Second end point

    # Draw the end points on the frame
    cv2.circle(frame, pt1, radius=5, color=(0, 255, 0), thickness=-1)  # Green circle at the first end point
    cv2.circle(frame, pt2, radius=5, color=(0, 255, 0), thickness=-1)  # Green circle at the second end point

    plt.text(p1[0], p1[1], f"Major Slope: {major_slope:.2f}", ha='center', va='bottom')
    plt.text(p1[0], p1[1] - 10, f"Major Constant: {major_constant:.2f}", ha='center', va='top')
    plt.text(p1[0], p1[1] - 20, f"Value: {value:.2f}", ha='center', va='top')

    return (major_slope, major_constant, int(value), m2, m1)

# function_name : RoiForBoundary
# Description : Insert the gps locarion of the boundary between dig and nodig zones.. 
# parameters : slope,constant,midpoint,disparityframe
# return : None

def RoiForBoundary(slope,constant,midpoint,depthframe):
    i,j=midpoint
    # print(y,x)
    horizon=25
    vertical=25
    #camera lat long
    lat1 = 12.917816972222222441666666666667
    lon1 = 77.633101972222227971083333333333
    
    count=0
    for w in range(int(i-horizon),int(i+horizon)):
        for h in range(int(j-vertical),int(j+vertical)):
             pixelvalue=int(h-(slope*w)-constant)
             if (pixelvalue==0):
                    # print("111111111")
                    count+=1
                    # cv2.circle(frame,(w,h),2,(0,0,0))
                    dist= project_disparity_to_3d(depthframe, (w,h))
                
                    x = dist[0][0]
                    y=dist[0][1]
                    z= dist[0][2]
                    
                    X=x 
                    Y= -z
                    Z = y
                        
                        
                    x = -Y
                    y= -X
                    z= Z
                
                    X,Y,Z=ConvertGPStocartesian(lat1,lon1) 
                   
                    x=X+x 
                    y=Y+y 
                    z= Z+z 
                    
                    # print("marker earth centered coordinates:",x,y,z)   
                    lat,lon=CartesianToGPS(x,y,z)
                    # InsertPixelLatLong(w,h,round(lat,8),round(lon,8),lat1,lon1)
                
    print("count of no of points on line",count)
                    
            
            

# function_name : createOverlay
# Description : colour the no dig region in red color. 
# parameters : frame(ndarray),colour(tuple),Marker3DCoordinate(list),BucketPoints(list),Bucket3Dcoordinates(list),slope(float),constant(float),minorvalue(float)
# return : None
def createOverlay(frame,colour,Marker3DCoordinate,BucketPoints,Bucket3Dcoordinates,slope,constant,minorvalue, pt1, pt2):
    height = frame.shape[0]
    width =frame.shape[1]
    flag=False

    print("bucket 3d coordinates",Bucket3Dcoordinates)
    print("marker y coordinates:",Marker3DCoordinate)
    minYvalue = min(Marker3DCoordinate)
        
    
    if (Bucket3Dcoordinates!=[] and Marker3DCoordinate!=[]):
        bucketHeight = abs(minYvalue-Bucket3Dcoordinates[0][1])
        bucketHeightinfeet = bucketHeight*feet
        # cv2.putText(
        #                 frame,
        #                 str(round(bucketHeightinfeet,2))+ " ft.",
        #                 BucketPoints[0],
        #                 fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        #                 fontScale = 0.6,
        #                 color = (0, 255, 0),
        #                 thickness=2
        #             )
        
    print("maximum value:",minYvalue)
    if (Bucket3Dcoordinates!=[]):
        x=Bucket3Dcoordinates[0][0]*feet
        y= Bucket3Dcoordinates[0][1]*feet
        z= Bucket3Dcoordinates[0][2]*feet
    overlay1=frame.copy()
    for w in range(width):
      for h in range(height):
        pixelvalue=int(h-(slope*w)-constant)
        if(pixelvalue*minorvalue>0):            
            cv2.circle(overlay1,(w,h),2,colour)    

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
            
            if ((abs(bucketHeightinfeet))<=BucketHeightFromGround):
                print("+++++++++++++++ALERT:Bucket is below 1 feet of the ground. Bucket is at a height",(abs(y)*feet),"feet")
                
                
                cv2.putText(
                            frame,
                            ("Alert: Bucket is in NO dig Zone below 2.5 foot"),
                            (600,300),
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale = 0.6,
                            color = (0, 0, 255),
                            thickness=2
                        )
                flag=True
                
    return overlay1,flag, pt1, pt2
            

# function_name : UpdateOverlay
# Description : colour the no dig region in red color. 
# parameters : frame(ndarray),colour(tuple),Marker3DCoordinate(list),BucketPoints(list),Bucket3Dcoordinates(list),slope(float),constant(float),minorvalue(float)
# return : None
def UpdateOverlay(frame,colour,Marker3DCoordinate,BucketPoints,Bucket3Dcoordinates,slope,constant,minorvalue, pt1, pt2):
    height = frame.shape[0]
    width =frame.shape[1]
    flag=False

    print("bucket 3d coordinates",Bucket3Dcoordinates)
    print("marker y coordinates:",Marker3DCoordinate)
    minYvalue = min(Marker3DCoordinate)
        
    
    if (Bucket3Dcoordinates!=[] and Marker3DCoordinate!=[]):
        bucketHeight = abs(minYvalue-Bucket3Dcoordinates[0][1])
        bucketHeightinfeet = bucketHeight*feet
        # cv2.putText(
        #                 frame,
        #                 str(round(bucketHeightinfeet,2))+ " ft.",
        #                 BucketPoints[0],
        #                 fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        #                 fontScale = 0.6,
        #                 color = (0, 255, 0),
        #                 thickness=2
        #             )
        
    print("maximum value:",minYvalue)
    if (Bucket3Dcoordinates!=[]):
        x=Bucket3Dcoordinates[0][0]*feet
        y= Bucket3Dcoordinates[0][1]*feet
        z= Bucket3Dcoordinates[0][2]*feet
    overlay1=frame.copy()
    for w in range(width):
      for h in range(height):
        pixelvalue=int(h-(slope*w)-constant)
        if(pixelvalue*minorvalue>0):            
            cv2.circle(overlay1,(w,h),2,colour)
    

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
            
            if ((abs(bucketHeightinfeet))<=BucketHeightFromGround):
                print("+++++++++++++++ALERT:Bucket is below 1 feet of the ground. Bucket is at a height",(abs(y)*feet),"feet")
                
                
                cv2.putText(
                            frame,
                            ("Alert: Bucket is in NO dig Zone below 2.5 foot"),
                            (600,300),
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale = 0.6,
                            color = (0, 0, 255),
                            thickness=2
                        )
                flag=True
                
    return overlay1,flag, pt1, pt2
        
