'''
use to collect sample by turtlebot_teleop

yuchen
30.07.2017, Hamburg
'''
import argparse
import os
import rospy
import tensorflow as tf
#from algorithm.DQNAgent_yuchen import DQNAgent
from agent.agent_mobile import AgentMobile
from algorithm.core import ReplayMemory
from algorithm.preprocessors import *
from geometry_msgs.msg import Twist
import sys, select, termios, tty

msg = """
Control Your Turtlebot!
---------------------------
Moving around:
        i
   j    k    l
i : go ahead
j : turn left
l : trun right
k : stop

a : force stop and quit

"""
# args setting.
parser = argparse.ArgumentParser(description='Collect samples by hand!')
parser.add_argument('-o', '--output', default='./log/', help='Directory to save data to')
parser.add_argument('--batch_size', default=32, type=int, help='Minibatch size')
parser.add_argument('--num_frames', default=4, type=int, help='Number of frames to feed to Q-Network')
parser.add_argument('--frame_width', default=84, type=int, help='Resized frame width')
parser.add_argument('--frame_height', default=84, type=int, help='Resized frame height')
parser.add_argument('--replay_memory_size', default=50, type=int, help='Number of replay memory the agent uses for training')
args = parser.parse_args()
print("==== Output saved to: ", args.output)
print("==== Args used:")
print(args)

#define action: action: 0-go , 1-turn left , 2-turn right , 3-stop
actions = {
    'i':0,
    'j':1,
    'l':2,
    'k':3,
       }

def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)
    #instance class object.
    AgentMobile_obj = AgentMobile()
    memory_obj = ReplayMemory(args)
    atari_processor = AtariPreprocessor()
    rospy.sleep(0.5) # must need! to wait the ros topic.

    state = AgentMobile_obj.reset()
    t = 0
    try:
        print msg
        while(True):
            t = t + 1
            key = getKey()
            #print('current key:', key)
            if key in actions.keys():
                action = actions[key]
                #print('current action:', action)
                processed_state = atari_processor.process_state_for_memory(state)
                state, reward, done, info = AgentMobile_obj.step(action)
                rospy.sleep(0.5)
                processed_reward = atari_processor.process_reward(reward)
                memory_obj.append(processed_state, action, processed_reward, done)
            elif key == '' :
                AgentMobile_obj.reset()
                rospy.sleep(0.5)
            elif key == 'a':
                break
    except:
        print 'error'
    finally:
        file_memory = open('results/memory.csv', 'w')
        memoryResults = dict(
                        actions = memory_obj.actions.tolist(),
                        states = memory_obj.screens.tolist(),
                        rewards = memory_obj.rewards.tolist(),
                        terminals = memory_obj.terminals.tolist() )
        file_memory.write(str(memoryResults))
        rospy.sleep(1)
        file_memory.close()
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
