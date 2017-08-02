<<<<<<< HEAD
import rospy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import *

class GoForwardAvoid():
    def __init__(self):
        rospy.init_node('nav_test', anonymous=False)

	#what to do if shut down (e.g. ctrl + C or failure)
	rospy.on_shutdown(self.shutdown)


	#tell the action client that we want to spin a thread by default
	self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
	rospy.loginfo("wait for the action server to come up")
	#allow up to 5 seconds for the action server to come up
	self.move_base.wait_for_server(rospy.Duration(5))

	#we'll send a goal to the robot to move 3 meters forward
	goal = MoveBaseGoal()
	goal.target_pose.header.frame_id = 'base_link'
	goal.target_pose.header.stamp = rospy.Time.now()
	goal.target_pose.pose.position.x = 3.0 #3 meters
	goal.target_pose.pose.orientation.w = 1.0 #go forward

	#start moving
        self.move_base.send_goal(goal)

	#allow TurtleBot up to 60 seconds to complete task
	success = self.move_base.wait_for_result(rospy.Duration(60))

	if not success:
                self.move_base.cancel_goal()
                rospy.loginfo("The base failed to move forward 3 meters for some reason")
    	else:
		# We made it!
		state = self.move_base.get_state()
		if state == GoalStatus.SUCCEEDED:
		    rospy.loginfo("Hooray, the base moved 3 meters forward")

    def shutdown(self):
        rospy.loginfo("Stop")


if __name__ == '__main__':
    try:
        GoForwardAvoid()
    except rospy.ROSInterruptException:
        rospy.loginfo("Exception thrown")
=======
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
run turlebot.

yuchen 2017.02.15
deng@informatik.uni-hamburg.de
'''
import ros
import rospy
import numpy as np
#from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from agent_ros_mobile.msg import Command
import actionlib
from actionlib_msgs.msg import *

# global variable.
speed = .2
turn = 1
robotCommand = np.zeros(2)*np.nan
MobileControl_flag = False

def control_callback(msg):
    global MobileControl_flag
    MobileControl_flag = True

    linear_speed = msg.linearSpeed
    angle = msg.angle
    global robotCommand
    robotCommand = np.array([linear_speed, angle])
    print('....recieves command, then control!')
#end of control_callback method.

if __name__=="__main__":
    rospy.init_node('turtlebot_teleop', anonymous=False)
    pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=5)
    rospy.Subscriber('/yuchen_controller_command', Command, control_callback)

    target_speed = 0
    target_turn = 0
    control_speed = 0
    control_turn = 0
    rospy.sleep(1) # wait for topic
    #ros.spinOnce()
    while(True):
        if MobileControl_flag:
            print('test2')
            #1. recieves control command.
            target_speed = robotCommand[0] #linear velocity: 0.2, 0.4
            target_turn = robotCommand[1] # angle : left=-1, keep=0, right=1

            '''
            if target_speed > control_speed:
                control_speed = min( target_speed, control_speed + 0.02 )
            elif target_speed < control_speed:
                control_speed = max( target_speed, control_speed - 0.02 )
            else:
                control_speed = target_speed

            if target_turn > control_turn:
                control_turn = min( target_turn, control_turn + 0.1 )
            elif target_turn < control_turn:
                control_turn = max( target_turn, control_turn - 0.1 )
            else:
                control_turn = target_turn
            '''
            control_speed = target_speed
            control_turn = target_turn

            twist = Twist()
            twist.linear.x = control_speed
            twist.linear.y = 0
            twist.linear.z = 0

            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = control_turn #
            pub.publish(twist)
            print('command(linear and angle):', twist.linear.x, twist.angular.z)

            rospy.sleep(0.1) # control loop: 0.1 sec (10Hz)
            #ros.spinOnce()
            global MobileControl_flag
            MobileControl_flag = False
    rospy.spin()
>>>>>>> 132a2008d85b4a1cb187f30a3d3123db970f3829
