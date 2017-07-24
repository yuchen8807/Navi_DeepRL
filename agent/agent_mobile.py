
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
from agent_ros_mobile.msg import Command, DataRequest
from utility import DataFiles

# global variable
curr_sample_imgState = np.zeros(21168)*np.nan #global variables: for testing
curr_sample_rState = np.zeros(6)*np.nan #global variables: Temporary save Sample
class AgentMobile():
    """docstring for a AgentMobile (turtlebot)."""
    def __init__(self):
        self.start_vw = np.zeros(2)
        rospy.init_node('agent_mobile_node')
        self.thread_pubsub = threading.Thread(target=self.init_pubs_subs())
        self.thread_pubsub.start()

    #end of init method.

    def init_pubs_subs(self):
        # publisher
        self.action_pub = rospy.Publisher('/yuchen_controller_command', Command, queue_size=1)
        #self.Gripper_pub = rospy.Publisher('/yuchen_controller_angle_command', angleCommand, queue_size=1)
        #self.data_request_pub = rospy.Publisher('/yuchen_controller_data_request', DataRequest, queue_size=1000)
        #subscriber
        self.sample_result_sub = rospy.Subscriber('/yuchen_controller_report', DataRequest, self.sample_callback)
    #end of init_pubs_subs method
    def sample_callback(self, msg):
        '''get sample under data-request'''
        global curr_sample_imgState, curr_sample_rState
        curr_sample_imgState = msg.imgState # imgState
        curr_sample_rState = msg.rState # robot state: linear + angle velocity
    # end of sample_callback method

    def get_data(self):
        global curr_sample_imgState, curr_sample_rState
        robot_linearVelocity = curr_sample_rState[0]
        robot_angleVelocity = curr_sample_rState[1]
        rState = np.array([robot_linearVelocity, robot_angleVelocity])
        imgState = curr_sample_imgState
        return imgState, rState
    #end of get_data method

    def reset_robot(self, reset_vw= None):
        '''reset robot and sensor'''
        reset_vw = self.start_vw if reset_vw is None else reset_vw

        # command control.
        reset_command = Command()
        reset_command.linearSpeed = 0.0
        reset_command.angle = 0
        self.action_pub.publish(reset_command)
        print('--------RL_agent: send reset_arm command', 'reset_vw=',reset_vw)
        rospy.sleep(0.2)
        tmp_imgState, tmp_rState = self.get_data()
        return tmp_imgState
    #end of reset_arm method

    def robot_step(self, action):
        '''one step: execute action , observe next_state and reward'''

        # command control
        reset_command = Command()
        reset_command.linearSpeed = action[0]
        reset_command.angle = action[1]
        self.action_pub.publish(reset_command)

        # get next_state
        next_imgState, next_rState = self.get_data()

        # compute reward.
        reward = 0.0
        '''
        collision_flag = collisionDetection()
        if collision_flag == True:
            reward = -1
        else:
            reward = 0.01
        '''
        return next_imgState, reward
    #end of robot_step method


''' test'''
if __name__ == '__main__':
    AgentMobile_obj = AgentMobile()
    AgentMobile_obj.reset_robot()
    rospy.sleep(1)
    tmp_imgState, tmp_rState= AgentMobile_obj.get_data()
    print('test-data:',tmp_rState )

    for i in xrange(10):
        action = np.array([0.2, 0])
        AgentMobile_obj.robot_step(action)
        print(i, 'control command:', action)
        rospy.sleep(0.5)

    rospy.spin()
