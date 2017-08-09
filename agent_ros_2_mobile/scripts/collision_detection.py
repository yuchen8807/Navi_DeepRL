'''liang: collision detection'''
from agent_ros_mobile.msg import  collisionState
from sensor_msgs.msg import Image
from kobuki_msgs.msg import BumperEvent
import rospy
def cal_bumper_callback(data):
    print data

def listener():
    rospy.init_node('depth_info_listener', anonymous=True)
    rospy.Subscriber('/mobile_base/events/bumper',BumperEvent,cal_bumper_callback)

    rospy.spin()

if __name__ == '__main__':
    listener()
