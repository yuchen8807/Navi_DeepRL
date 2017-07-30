#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
record turlebot state: image , current state.

yuchen 2017.02.15
deng@informatik.uni-hamburg.de
'''
import ros
import rospy
import numpy as np
import cv2
import cv_bridge
import sensor_msgs.msg
from agent_ros_mobile.msg import DataRequest
from preprocessors import AtariPreprocessor

global tmp_image
def process_image(arg):
    '''compress the raw image for RL'''
    pass

def image_callback(msg):
    w = msg.width
    h = msg.height
    global tmp_image
    tmp_image = msg.data
    #image_compressed = process_image(image_pub)
    #print 'width : ' + str(w) + '\theight : ' + str(h)
#end of image_callback method.

if __name__ == '__main__':
    rospy.init_node('record_robot')

    image_pub = rospy.Publisher("/yuchen_controller_report", DataRequest, queue_size=1)
    image_sub = rospy.Subscriber("/camera/rgb/image_raw",sensor_msgs.msg.Image, image_callback)
    rospy.loginfo('initialized')
    rospy.sleep(0.5) # wait for topic
    #rospy.spinOnce()
    imageProcess_obj = AtariPreprocessor()
    while (True):
        #1. get img and publish to RL
        #print('tmp_image: ', tmp_image)
        tmp_DataRequest = DataRequest()
        #compress image:
        compressed_img = imageProcess_obj.process_state_for_network(tmp_image)
        tmp_DataRequest.imgState = np.ones(21168) # for testing
        tmp_DataRequest.rState = np.ones(2)
        image_pub.publish(tmp_DataRequest)

        # 2. get robot state and publish to RL(now, don't consider!!)

        rospy.sleep(0.1) # record loop: 0.1 sec (10Hz)
        #rospy.spinOnce()
        #global tmp_image = np.zeros(2)
    #end of while.
    rospy.spin()
