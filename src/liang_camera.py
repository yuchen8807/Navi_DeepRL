#!/usr/bin/env python
# Copy right: Hongzhuo Liang Zhen Deng
# All rights reserved.

import cv2
import rospy
import scipy.misc

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage

import sys, time
import numpy as np
from scipy.ndimage import filters
import cv2
import roslib


def callback(msg):
    #### 1. direct conversion to CV2 ####
    np_arr = np.fromstring(msg.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
    global tmp_data
    tmp_data = image_np
    # 2. convert np image to grayscale
    '''
    featPoints = feat_det.detect( cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY))
    time2 = time.time()
    '''
    #cv2.imshow('cv_img', image_np)
    #cv2.waitKey(2)
    #### 3. Create CompressedIamge####
    #image_array = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()

def callback2(msg2):
    print msg2



def data_collector():
    rospy.init_node('data_collector_tb', anonymous=True)
    rospy.Subscriber("/camera/rgb/image_color/compressed",CompressedImage, callback)
    rospy.Subscriber("/cmd_vel_mux/input/navi",Twist,callback2)
    rospy.sleep(2)
    print(tmp_data[0])

    rospy.spin()



#camera/rgb/image_color

#/cmd_vel_mux/input/navi

if __name__ == '__main__':
    data_collector()
