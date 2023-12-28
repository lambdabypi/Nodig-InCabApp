# **************************************************************************************************************************
#  * FileName:     datainsertion.py
#  * Project:      No Dig
#  * Developed by: Chipmonk Technologies Private Limited
#  * Description : This script creates the table -"MarkerData" in Postgres "nodig" database and insert the GPS data of the marked lines in that table.
#  * Copyright and Disclaimer Notice Software:
# **************************************************************************************************************************

# Dependencies
# **************************************************************************************************************************
import psycopg2



# Function Name : CreateTable
# **************************************************************************************************************************
# Description : Create the table with name MarkerData in database. 
# parameters : None
# return : None
# **************************************************************************************************************************

def CreateTable():
    conn = psycopg2.connect(
    database="nodig", user='postgres', password='teamcity', host='127.0.0.1', port= '5432'
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
    MINOR_VALUE_y CHAR(50),
    MINOR_VALUE_x CHAR(50),
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

# Function_name : insertmarkerdata
# **************************************************************************************************************************
# Description   : It inserts the slope of the marked line,constant of the line,minorValue of the line,pointx (pixel x ordinate),pointy(pixel y ordinate),count(frame number),latitude of the marked T,longitude of of the marked T,camera_lat (camera latitude position),camera_long(camera longitude position) data into MarkerData table in "nodig" database. 
# parameters    : slope,constant,minorValue,pointx,pointy,count,lat,long,camera_lat,camera_long
# return        : None
# **************************************************************************************************************************

def insertmarkerdata(slope,constant,minorValue,minor_y,minor_x,pointx,pointy,count,lat,long,camera_lat,camera_long):
    print("data to insert:",pointx,pointy)
    conn = psycopg2.connect(
    database="nodig", user='postgres', password='teamcity', host='127.0.0.1', port= '5432'
    )
    
    cursor = conn.cursor()
    
    
    cursor.execute("INSERT INTO MarkerData VALUES('%d','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s')" % (count,"marking",slope, constant, minorValue,minor_y,minor_x,pointx,pointy,lat,long,camera_lat,camera_long))
    print("data inserted successfully")

    conn.commit()
    conn.close()
    
    return None




# Function_name : FetchData
# **************************************************************************************************************************
# Description   : It fetched the slope of the marked line,constant of the line,minorValue of the line,pointx (pixel x ordinate),pointy(pixel y ordinate),count(frame number),latitude of the marked T,longitude of of the marked T,camera_lat (camera latitude position),camera_long(camera longitude position) data from the "MarkerData" table in "nodig" database. 
# parameters    : slope,constant,minorValue,pointx,pointy,count,lat,long,camera_lat,camera_long
# return        : None
# **************************************************************************************************************************



def FetchData():
    
    conn = psycopg2.connect(
    database="nodig", user='postgres', password='teamcity', host='127.0.0.1', port= '5432'
    )
    
    cursor = conn.cursor()
    
    cursor.execute("select * from markerdata ORDER BY frame_id DESC limit 3")
    result=cursor.fetchall()
    
    conn.commit()
    conn.close()
    
    return result