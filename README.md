**in progress!**

## Navi_DeepRL
Goal: use deep reinforcement learning to realize obstacle avoidance.

Platform: turlebot

Deep RL: directly learn robot actions from pixel information.

### 1. basic settig of DRL.
Action: linearSpeed (0.1, 0.2, 0.3) + angle (right=1, still=0, left=-1)

Pixel state: （小梁和唐宋确定下！）  raw image --> processed imgState(vector).

Reward: r = -2 (collision ), r = 0.01 (O.W)



### 2. install some dependences.
1. install [turtlebot source](http://wiki.ros.org/turtlebot/Tutorials/indigo/Turtlebot%20Installation#turtlebot.2BAC8-Tutorials.2BAC8-indigo.2BAC8-Source_Installation). use 'Source_Installation'.

b. other dependences.

**Tensorflow**

### 3. file introduction
**agent** file: define RL agent

**agent_ros_2_mobile**　file: use to communicate RL agent with mobile robot.

**algorithm** file: define DRQN algorithm

**utility** file: some useful function.



### 4. turlebot simulation (tested, ok!)
Rviz

```
a. view in rviz.
roslaunch turtlebot_rviz_launchers view_robot.launch

b. navigation in Rviz.
roslaunch turtlebot_stage turtlebot_in_stage.launch

```
Gazebo

```
a. view in gazebo.
roslaunch turtlebot_gazebo turtlebot_world.launch

```

Keyboard control.

```
roslaunch turtlebot_teleop keyboard_teleop.launch

```
### 5. turtlebot (real-robot)
```
1. 小梁
reference:
a: http://wiki.ros.org/turtlebot/Tutorials
b. http://learn.turtlebot.com/

```

### 6. Deep recurrent Q-learning (DRQN)

```
1. agent_ros_2_mobile (tested, ok!)
roslaunch agent_ros_mobile agent_ros_mobile.launch
    实现： a. python recordMobile.py  # get image from '/camera/rgb/image_raw' topic.
          b. python runMobile.py  #

2. DRQN (tested, ok!)
cd /home/yuchen/catkin_ws/src/Navi_DeepRL
python Navi_deepRL_training.py
    实现： a. RL agent (mobile robot).(tested, ok!)
            cd /home/yuchen/catkin_ws/src/Navi_DeepRL/agent
            python agent_mobile.py
          b. Q-network
            ./algorithm
```


### 7. State learning (???)


### 8. ???
