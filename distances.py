# ******************************************************************************************************************
#  * FileName:     distances.py
#  * Project:      No Dig
#  * Developed by: Chipmonk Technologies Private Limited
#  * Description : This script calculates the distance between ground markings and bucket.
#  * Copyright and Disclaimer Notice Software:
#  *****************************************************************************************************************



# Function Details-
# **************************************************************************************************************************
# Function name : DistanceBetweenMarkingAndBucket
# Description : It compute the distance between ground marking and bucket. 
# parameters to be passed : Frame in np.array type and 3d coordinates of marking and bucket in the list type
# return : None
# **************************************************************************************************************************


def DistanceBetweenMarkingAndBucket(frame,points,pointsB):
       
    for i in range(len(points)):
        for j in range(len(pointsB)):
    
        # print(pts)
            print(points)
            # print(pointsP)
            
            distance = (((points[i][0]-pointsB[j][0])**2+(points[i][1]-pointsB[j][1])**2+(points[i][2]-pointsB[j][2])**2)**0.5)
                    
                
                
    return None


# Function Details-
# **************************************************************************************************************************
# Function name : DistanceBetweenMarkings
# Description : It compute the distances between ground markings. 
# parameters to be passed : Frame in np.array type and 3d coordinates of marking in the list type.
# return : None
# **************************************************************************************************************************

def DistanceBetweenMarkings(frame,points,pointsP):
    
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            
            # print(pts)
            print(points)
            
            distance = (((points[i][0]-points[j][0])**2+(points[i][1]-points[j][1])**2+(points[i][2]-points[j][2])**2)**0.5)
            
    return None
            