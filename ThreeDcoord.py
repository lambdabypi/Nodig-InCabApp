# ******************************************************************************************************************
#  * FileName:     ThreeDcoord.py
#  * Project:      No Dig
#  * Developed by: Chipmonk Technologies Private Limited
#  * Description : This script compute the 3 D coordinates of marking and bucket.
#  * Copyright and Disclaimer Notice Software:
#  *****************************************************************************************************************

# Variable Declarations
#  ***************************************************************************************************************
f= 798.4660034179688 #focal length of camera
B = .075  


# Function Details-
# **************************************************************************************************************************
# Function name : project_disparity_to_3d
# Description : It computes the 3-d coordinates of red T markings on the ground.
# parameters to be passed : Disparity frame of type np.array and coordinates of marking of type tuple
# return : 3d coordinates along with distance of marking from camera
# **************************************************************************************************************************

def project_disparity_to_3d(disparity, boundingbox):
    print(type(disparity))
    print(type(boundingbox))
    

    points = []


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
    
    points.append( (X**2+Y**2+Z**2)**0.5)
    return points
