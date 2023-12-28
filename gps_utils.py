
# ******************************************************************************************************************
#  * FileName:     gps_location.py
#  * Project:      No Dig
#  * Developed by: Chipmonk Technologies Private Limited
#  * Description : This script compute the GPS location of the ground markings.
#  * Copyright and Disclaimer Notice Software:
#  *****************************************************************************************************************


# Dependencies
#  ******************************************************************************************************************

from math import atan2, cos,sin,asin
import math
import numpy as np

# Function_name : ConvertGPStocartesian
# **************************************************************************************************************************
# Description   : It Convert latitude and longitude of the object to 3D coordinates. 
# parameters    : It uses the latitude and longitude of the object
# return        : 3d coordinates
# **************************************************************************************************************************

def ConvertGPStocartesian(lat,lon):
    R= 6371000
    lon =  lon * math.pi / 180
    lat = lat * math.pi / 180
    x = R * cos(lat) * cos(lon)

    y = R * cos(lat) * sin(lon)

    z = R *sin(lat)
    
    return x,y,z

# Function_name : CartesianToGPS
# **************************************************************************************************************************
# Description   : It Convert 3D coordinates of the object into latitude and longitude of that object.
# parameters    : x,y,z coordinates
# return        : latitude and longitude
# **************************************************************************************************************************


def CartesianToGPS(x,y,z):
   R= 6371000
   lat = asin(z / R)
   lon = atan2(y, x)
   lat = lat*180/math.pi
   lon = lon*180/math.pi
   
   return lat,lon

# code commented. It may be used in persistence calculation

# from geographiclib.geodesic import Geodesic
# ...
# def get_bearing(lat1, lat2, long1, long2):
#     brng = Geodesic.WGS84.Inverse(lat1, long1, lat2, long2)['azi1']
#     return brng

# def DistanceLatLong(lat1,lon1,lat2,lon2):    
#     radius = 6371  # km

#     dlat = math.radians(lat2 - lat1)
#     dlon = math.radians(lon2 - lon1)
#     a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
#          math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
#          math.sin(dlon / 2) * math.sin(dlon / 2))
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#     d = ((radius * c)/1000)

#     return d