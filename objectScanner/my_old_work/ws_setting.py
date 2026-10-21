#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('yaskawa_bin_picking')
import sys
import rospy
import rospkg
import cv2
import fiducial_msgs.msg as f_msgs

from utils import *

import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from image_geometry import PinholeCameraModel
import tf
# import geometry_msgs.msg
import math
from geometry_msgs.msg import Transform


class workspace_creator:
  def __init__(self,ws_path=None):
    #check if ws is existed
    self.draw_ws=False
    self.set_new_workspace=True
    self.ws_path=ws_path
    self.is_ws_exist = isFileExist(self.ws_path)
    self.uv_pos_list=[]
    self.pos_list=[]
    if(self.is_ws_exist):
        self.set_new_workspace=False
        ws_data = read_data(ws_path)
        b0=convert_text2position(ws_data['b0'])
        b1=convert_text2position(ws_data['b1'])
        b2=convert_text2position(ws_data['b2'])
        b3=convert_text2position(ws_data['b3'])
        b4=convert_text2position(ws_data['b4'])
        b5=convert_text2position(ws_data['b5'])
        b6=convert_text2position(ws_data['b6'])
        b7=convert_text2position(ws_data['b7'])
        self.pos_list.append(b0)
        self.pos_list.append(b1)
        self.pos_list.append(b2)
        self.pos_list.append(b3)
        self.pos_list.append(b4)
        self.pos_list.append(b5)
        self.pos_list.append(b6)
        self.pos_list.append(b7)


    self.listener= tf.TransformListener()
    self.image_pub = rospy.Publisher("ws_image",Image,queue_size=2)
    self.keypress_res=rospy.Publisher('KeypressResponse', String, queue_size=2)
    self.bridge = CvBridge()
    self.windowName="workspace window"
    self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback)
    self.caminfo_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_data)
    self.fiducialTF = rospy.Subscriber('/fiducial_transforms',f_msgs.FiducialTransformArray,self.callback_fid)
    rospy.Subscriber('keypress', String, self.keyboard_callback)

    self.camInfo=None
    self.save_Data=False
    self.image=None

    #draw
    self.leftMousePress=False
    self.edit_ws=False
    self.corners2D=None
    self.clickPosition=[-1,-3]
    self.markerClicked=8
    self.num_baseMarker=4




  def CallBackFunc(self,event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        self.leftMousePress=True
        self.clickPosition[0]=x
        self.clickPosition[1]=y
        self.markerClicked=getMarkerClicked(self.corners2D,self.clickPosition)
        if self.markerClicked < 4:
            self.edit_ws=True
        else:
            self.edit_ws=False
        # print("Left button of the mouse is clicked - position (", x, ", ",y, ")")
    elif event == cv2.EVENT_LBUTTONUP:
        self.leftMousePress=False
        #self.edit_ws=False

    elif event == cv2.EVENT_MOUSEMOVE:
        if(self.leftMousePress==True and self.edit_ws==True):
            self.corners2D[self.markerClicked][0]=x
            self.corners2D[self.markerClicked][1]=y
        # print("Mouse move over the window - position (", x, ", ", y, ")")


  def keyboard_callback(self,msgs):
      self.keypress_res.publish("Please press (p) for saving or (d) for deleting workspace")
      if(msgs.data=='p' or msgs.data=='P'):
          if(isFileExist(self.ws_path)):
              self.keypress_res.publish("workspace is found")
          else:
              if(self.draw_ws):
                  f=open(self.ws_path, "a+")
                  i=0
                  for p in self.pos_list:
                      pos_text=convert_position2text(p)
                      text_data='b'+str(i)+'='+pos_text
                      f.write(text_data)
                      f.write('\n')
                      i=i+1
                  f.close()
                  self.is_ws_exist = isFileExist(self.ws_path)
                  self.set_new_workspace=False
                  self.keypress_res.publish("workspace is saved")
              else:
                  self.keypress_res.publish("No marker found")

      elif(msgs.data=='n' or msgs.data=='N'):
          if(isFileExist(self.ws_path)):
              os.remove(self.ws_path)
              self.is_ws_exist = isFileExist(self.ws_path)
              self.set_new_workspace=True
              self.uv_pos_list=[]
              self.pos_list=[]
              self.keypress_res.publish("workspace is deleted")
          else:
              self.keypress_res.publish('WS File does not exists')
          # self.keypress_res.publish(str(count+1)+" data is SAVED!!")

      else:
          self.keypress_res.publish("Please press (p) for saving or (d) for deleting workspace")

  def camera_data(self,msgs):
      self.camInfo=msgs
  def callback_fid(self,msgs):
      if(len(msgs.transforms)>3):
          self.draw_ws=True
      else:
          self.draw_ws=False


  def callback(self,data):
    try:
         cv_image = self.bridge.imgmsg_to_cv2(data,"bgr8")
    except CvBridgeError as e:
         print(e)
    cam_model=PinholeCameraModel()
    cam_model.fromCameraInfo(self.camInfo)

    if(self.set_new_workspace==False):
        self.draw_ws=True
    else:
        try:
            (vertice0,rot0) = self.listener.lookupTransform("camera_color_frame", 'ws_base0', rospy.Time(0))
            (vertice1,rot1) = self.listener.lookupTransform("camera_color_frame", 'ws_base1', rospy.Time(0))
            (vertice2,rot2) = self.listener.lookupTransform("camera_color_frame", 'ws_base2', rospy.Time(0))
            (vertice3,rot3) = self.listener.lookupTransform("camera_color_frame", 'ws_base3', rospy.Time(0))

            (vertice4,rot4) = self.listener.lookupTransform("camera_color_frame", 'ws_lid0', rospy.Time(0))
            (vertice5,rot5) = self.listener.lookupTransform("camera_color_frame", 'ws_lid1', rospy.Time(0))
            (vertice6,rot6) = self.listener.lookupTransform("camera_color_frame", 'ws_lid2', rospy.Time(0))
            (vertice7,rot7) = self.listener.lookupTransform("camera_color_frame", 'ws_lid3', rospy.Time(0))

            self.draw_ws=True
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

    if(self.draw_ws):
        self.uv_pos_list=[]
        if(self.set_new_workspace==True):
            self.pos_list=[]
                #base
            base_pos_list=[]
            pos_list2D=[]
            base_pos_list.append([vertice0,vertice4])
            base_pos_list.append([vertice1,vertice5])
            base_pos_list.append([vertice2,vertice6])
            base_pos_list.append([vertice3,vertice7])
            #map3Dpos2Pixel
            for p in base_pos_list:
                proj_base=cam_model.project3dToPixel(p[0])
                proj_lid=cam_model.project3dToPixel(p[1])
                px_base=int(round(proj_base[0]))
                py_base=int(round(proj_base[1]))
                px_lid=int(round(proj_lid[0]))
                py_lid=int(round(proj_lid[1]))
                pos_list2D.append([(px_base,py_base),(px_lid,py_lid),p])

            #base_pos_array= np.array(pos_list2D)
            sorted_x=sorted(pos_list2D, key=lambda tup: tup[0][0])
            #sorted_x=np.sort(base_pos_array.view('i8,i8'), order=['f0'], axis=0).view(np.int)
            v1=np.array(sorted_x[1][0])-np.array(sorted_x[0][0])
            v2=np.array(sorted_x[2][0])-np.array(sorted_x[0][0])
            v3=np.array(sorted_x[3][0])-np.array(sorted_x[0][0])
            ang1=get_angle(v1)
            ang2=get_angle(v2)
            ang3=get_angle(v3)

            paired_angle=[]
            paired_angle.append([sorted_x[0],0])
            paired_angle.append([sorted_x[1],np.rad2deg(ang1)])
            paired_angle.append([sorted_x[2],np.rad2deg(ang2)])
            paired_angle.append([sorted_x[3],np.rad2deg(ang3)])

            sorted_base=sorted(paired_angle, key=lambda tup: tup[1])
            for base in sorted_base:
                self.pos_list.append(base[0][2][0]) #0
                self.pos_list.append(base[0][2][1]) #4

                self.uv_pos_list.append(base[0][0])
                self.uv_pos_list.append(base[0][1])
            #display
        else:
            for p in self.pos_list:
                proj_pos=cam_model.project3dToPixel(p)
                rpx=int(round(proj_pos[0]))
                rpy=int(round(proj_pos[1]))
                self.uv_pos_list.append((rpx,rpy))

        self.corners2D = np.array(np.reshape(self.uv_pos_list[:16], [8, 2]), dtype='float32')

        lineThickness = 1
        line_color=(0,255,255)
        edges_corners = [[0, 1], [2, 3], [4, 5], [6, 7],
         [1, 3], [3, 5], [5, 7], [7, 1],
         [0, 2], [2, 4], [4, 6], [6, 0]]

        i=0
        for corner in self.corners2D:
            marker_colors=(0,0,255)
            if(self.set_new_workspace==False):
                marker_colors=(255,0,0)
            cv2.circle(cv_image, (corner[0],corner[1]), 5, (0,0,255),-1)
            cv2.putText(cv_image, str(i), (corner[0],corner[1]), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,255),1, cv2.LINE_AA)
            i=i+1

        for edge in edges_corners:
             cv2.line(cv_image,(self.corners2D[edge[0], 0], self.corners2D[edge[0], 1]), (self.corners2D[edge[1], 0], self.corners2D[edge[1], 1]), line_color, 1)
    else:
        i=0
        for corner in self.corners2D:
            cv2.circle(cv_image, (corner[0],corner[1]), 3, (255,0,0),-1)
            i=i+1

    cv2.namedWindow(self.windowName)
    cv2.setMouseCallback(self.windowName,self.CallBackFunc)
    cv2.imshow(self.windowName, cv_image)
    cv2.waitKey(1)

    try:
       self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
       print(e)



def main(args):
  rospy.init_node('ws_processing', anonymous=True)

  ws_cfg    = rospy.get_param('~workspace_path')
  wsc = workspace_creator(ws_cfg)

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
