#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
mobile robot for a navigation senario.
    action: linear(0.1, 0.2, 0.3) + angle(-pi/6, -pi/12, 0, pi/12, pi/6)
    state: image
Hamburg, 2016.12.21
'''
import sys
sys.path.append('../')
import rospy
import copy
import numpy as np
import math
from math import radians
import threading
# tip: from package_name.msg  import ; and package_name != file_name
from agent_ros_mobile.msg import Command, DataRequest, collisionState
from utility import DataFiles
import sensor_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
# global variable
#curr_sample_imgState = np.zeros(21168)*np.nan #global variables: for testing
#curr_sample_rState = np.zeros(6)*np.nan #global variables: Temporary save Sample
class AgentMobile():
    """docstring for a AgentMobile (turtlebot)."""
    def __init__(self):
        self.start_vw = np.zeros(2)
        rospy.init_node('agent_mobile_node')
        self.thread_pubsub = threading.Thread(target=self.init_pubs_subs())
        self.thread_pubsub.start()
        self.bridge = CvBridge()
    #end of init method.

    def init_pubs_subs(self):
        # publisher
        self.action_pub = rospy.Publisher('/yuchen_controller_command', Command, queue_size=1)
        #self.Gripper_pub = rospy.Publisher('/yuchen_controller_angle_command', angleCommand, queue_size=1)
        #self.data_request_pub = rospy.Publisher('/yuchen_controller_data_request', DataRequest, queue_size=1000)
        #subscriber
        self.sample_result_sub = rospy.Subscriber('/yuchen_controller_report', DataRequest, self.sample_callback)
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw",sensor_msgs.msg.Image, self.sample_callback)
        self.collision_sub = rospy.Subscriber('yuchen_collision', collisionState, self.collisionState_callback)
    #end of init_pubs_subs method
    def sample_callback(self, msg):
        '''get sample under data-request'''
        global curr_sample_imgState
        curr_sample_imgState = msg # imgState
    # end of sample_callback method
    def collisionState_callback(self, msg):
        global curr_collisionState
        curr_collisionState = msg.collision
    #end of collisionState_callback method

    def get_data(self):
        global curr_sample_imgState
        tmp_image = self.bridge.imgmsg_to_cv2(curr_sample_imgState, "rgb8")
        imgState = np.asarray(tmp_image)
        return imgState
    #end of get_data method

    def reset(self, reset_vw= None):
        '''reset robot and sensor'''
        # command control.
        reset_command = Command()
        reset_command.linearSpeed = 0.0
        reset_command.angle = 0

        self.action_pub.publish(reset_command)
        #print('--------RL_agent: send reset_arm command')
        rospy.sleep(0.2)
        tmp_imgState = self.get_data()
        return tmp_imgState
    #end of reset_arm method

    def step(self, action):
        '''one step: execute action , observe next_state and reward'''
        # <<<<<<<<<<<<<<<<<<<<<<<<<
        #action: 0-go (linear = 0.3, angle = 0), 1-turn left (0, -1), 2-turn right (0,-1), 3-slow(0.1, 0)
        # <<<<<<<<<<<<<<<<<<<<<<<<<
        if action == 0: #go
            linear = 0.1
            angle = 0
        elif action == 1:# turn left
            linear = 0.05
            angle = 1
        elif action == 2:#turn right
            linear = 0.05
            angle = -1
        else:
            linear = 0.05#slow
            angle = 0

        # command control
        reset_command = Command()
        reset_command.linearSpeed = linear
        reset_command.angle = angle
        self.action_pub.publish(reset_command)

        # get next_state
        rospy.sleep(0.1)
        next_imgState = self.get_data()

        # compute reward.
        reward = 0.01
        '''
        collision_flag = collisionDetection()
        target_flag = False
        if collision_flag == True:
            reward = -0.99
        elif target_flag == True:
            reward = 5
        else:
            reward = 0.005

        '''
        done = False
        info = ''
        return next_imgState, reward, done, info
    #end of robot_step method

''' test'''
if __name__ == '__main__':
    '''
    for i in xrange(10):
        num_actions = 4
        #action = np.random.randint(0, num_actions)
        action = 0
        if i==3:
            action = 1
        elif i==5:
            action = 2
        elif i == 7:
            action = 3
    AgentMobile_obj = AgentMobile()
    AgentMobile_obj.reset()
    rospy.sleep(1)
    tmp_imgState, tmp_rState= AgentMobile_obj.get_data()
    print('test-data:',tmp_rState )
    num_actions = 4
    for i in xrange(10):
        action = np.random.randint(0, num_actions)
        AgentMobile_obj.step(action)
        print(i, 'control command:', action)
        rospy.sleep(0.5)
    '''
    rospy.spin()
