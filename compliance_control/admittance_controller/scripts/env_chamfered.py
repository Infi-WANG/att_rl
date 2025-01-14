import gym
import rospy
import threading
import random
import numpy as np
import math
from copy import deepcopy
from gym import spaces
from geometry_msgs.msg import Pose
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import SetBool, Trigger
from self_defined_msgs.srv import set_VF_srv
from self_defined_msgs.srv import position_ctrl_srv

pose_state = Pose()
wrench_state = WrenchStamped()
force_scale = 10.0
torque_scale = 0.8

class UREnv_chamfer(gym.Env):
    def __init__(self):
        super(UREnv_chamfer, self).__init__()
        self.count = 0
        self.area = [1,2,3,4]
        self.itimes = []
        self.stimes = []
        thread = StateThread()
        thread.start()
        rospy.sleep(0.2)

        self.activate_admittance_control = rospy.ServiceProxy('/admittance_controller/admittance_controller_activation_service', SetBool)
        self.activate_virtual_force = rospy.ServiceProxy('/admittance_controller/virtual_force_control_activation_service', SetBool)
        self.set_cur_pose_as_equilibrium = rospy.ServiceProxy('/admittance_controller/set_cur_pose_as_equilibrium_service', Trigger)
        self.zero_ft_sensor = rospy.ServiceProxy('/ur_hardware_interface/zero_ftsensor', Trigger)
        self.set_VF = rospy.ServiceProxy('/admittance_controller/set_VF_service', set_VF_srv)
        self.position_ctrl = rospy.ServiceProxy('/UR_position_control', position_ctrl_srv)        
        self.set_refer_pose = rospy.ServiceProxy('/admittance_controller/set_EEF_ref_pose_service', Trigger)
        while not self.zero_ft_sensor():
            rospy.INFO("failed to zero ft sensor")
        while not self.activate_admittance_control(True):
            rospy.INFO("failed to activate admittance control")
        while not self.activate_virtual_force(True):
            rospy.INFO("failed to activate virtual force")

        self.force_upper_x = self.force_upper_y = self.force_upper_z = 1.
        self.force_lower_x = self.force_lower_y = self.force_lower_z = -1.
        self.action_space = spaces.Box(
            low=np.array([self.force_lower_x, self.force_lower_y]),
            high=np.array([self.force_upper_x, self.force_upper_y]),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.finfo(np.float32).max,
            high=np.finfo(np.float32).max,
            shape=(10,),
            dtype=np.float32
        )
        self.insert_space = spaces.Box(
            low=-np.finfo(np.float32).max,
            high=np.finfo(np.float32).max,
            shape=(8,),
            dtype=np.float32
        )

    def step(self, action):
        self.step_count += 1
        self.action[0] = action[0]
        self.action[1] = action[1]
        x_force, y_force = action
        while not self.set_VF([x_force, y_force, 20, 0, 0, 0]):
            rospy.INFO("failed to set virtual force")
        next_state = self.getCurrentState(20)  # average = sum devided by 20, delay 0.01 * 20 = 0.2 s
        reward, done = self.caculateReward(next_state)
        self.state = next_state
        if self.step_count == 199:
            self.count += 1
        return next_state, reward, done
    
    def insert_step(self, action):
        self.insert_step_count += 1
        self.insert_action[0] = action[0]
        self.insert_action[1] = action[1]
        x_torque, y_torque = action
        while not self.set_VF([0, 0, 15, x_torque, y_torque, 0]):
            rospy.INFO("failed to set virtual force")
        next_state = self.insertCurrentState(20)
        reward, done = self.insertReward(next_state)
        self.insert_state = next_state
        return next_state, reward, done

    def caculateReward(self, next_state):
        done_reward = 0
        translation_reward_z = 0.
        zoom_factor_z = 100
        x_p = math.atanh(next_state[0])
        y_p = math.atanh(next_state[1])
        distance_p = np.sqrt(np.square(x_p) + np.square(y_p))
        translation_reward_xy = -1.0
        if distance_p <= 0.02:
            translation_reward_xy = -0.5
        if distance_p < 0.008 and math.atanh(next_state[2]) >= 0.001:
            translation_reward_xy = 0.0
        if distance_p < 0.008 and math.atanh(next_state[2]) >= 0.002:
            translation_reward_z = ((math.atanh(next_state[2])-math.atanh(self.state[2])) * zoom_factor_z + 0.1)*5
            translation_reward_xy = 0.0

        if distance_p >= self.deviate_from_target:
            done = True
            done_reward = -100
            self.count += 1
        elif math.atanh(next_state[2]) >= 0.002 and distance_p < 0.008:
            done = True
            done_reward = 10
            self.count += 1
            self.end_search = rospy.Time.now()
        else:
            done = False

        # if distance_p < 0.003:
        #     done = True
        #     done_reward = 10
        #     self.count += 1
        # elif distance_p > self.deviate_from_target or math.atanh(next_state[2]) >= 0.003:
        #     done = True
        #     done_reward = -10
        #     self.count += 1
        # else:
        #     done = False

        reward = (done_reward + translation_reward_xy + translation_reward_z)
        print("action: {:.4} {:.4}, xy: {:.4}, z: {:.4}, done: {}".format(self.action[0], self.action[1], translation_reward_xy, translation_reward_z, done))
        return reward, done

    def insertReward(self, next_state):
        done_reward = 0
        translation_reward_z = -abs(next_state[3])/10
        if abs(next_state[3]) < 3:
            translation_reward_z = (5 - abs(next_state[3]))/10

        if math.atanh(next_state[0]) >= 0.045:
            done = True
            done_reward = 10.0
            self.end_insert = rospy.Time.now()
            itm = (self.end_insert-self.start_insert).to_sec()
            stm = (self.end_search-self.start_search).to_sec()
            
            self.stimes.append(stm)
            step1 = deepcopy(self.stimes)
            s1 = '\n'
            for i in range(len(step1)):
                step1[i]=str(step1[i])
            f1=open("search_times.txt", "w")
            f1.write(s1.join(step1))
            f1.close()

            self.itimes.append(itm)
            step = deepcopy(self.itimes)
            s2 = '\n'
            for i in range(len(step)):
                step[i]=str(step[i])
            f2=open("insert_times.txt", "w")
            f2.write(s2.join(step))
            f2.close()
        else:
            done = False

        reward = (done_reward + translation_reward_z)
        print("action: {:.4} {:.4}, z: {:.4}, done: {}".format(self.insert_action[0], self.insert_action[1], translation_reward_z, done))
        return reward, done

    def reset(self, origin):
        self.action = [0, 0]
        self.step_count = 0
        self.deviate_from_target = 0.022
        self.origin = deepcopy(origin)
        self.target = deepcopy(origin)
        r = 0.005
        theta = random.uniform(0, math.pi*2)
        self.target.position.x += r*math.cos(theta)
        self.target.position.y += r*math.sin(theta)
        self.target.position.z += -0.005 - 0.247
        while not self.zero_ft_sensor():
            rospy.INFO("failed to zero ft sensor")
        rospy.sleep(0.2)
        while not self.set_cur_pose_as_equilibrium():
            rospy.INFO("failed to set equilibrium")
        rospy.sleep(0.2)
        while not self.set_VF([0, 0, -40, -2, 0, 0]):
            rospy.INFO("failed to set virtual force")
        rospy.sleep(0.1)
        for i in range(50):
            while not self.set_VF([0, 0, -40, 2*math.pow(-1, (i%2)), 0, 0]):
                rospy.INFO("failed to set virtual force")
            if pose_state.position.z <= -0.02:
                break
            rospy.sleep(0.2)
        while not self.set_VF([0, 0, 0, 0, 0, 0]):
            rospy.INFO("failed to set virtual force")
        origin = deepcopy(self.origin)
        target = deepcopy(self.target)
        self.postionMotion(origin, target)
        while not self.set_VF([0, 0, 20, 0, 0, 0]):
            rospy.INFO("failed to set virtual force")
        rospy.sleep(2.5)
        self.state = self.getCurrentState(250)
        self.start_search = rospy.Time.now()
        return self.state
    
    def insert(self):
        self.insert_action = [0, 0]
        self.insert_step_count = 0
        while not self.set_cur_pose_as_equilibrium():
            rospy.INFO("failed to set equilibrium")
        while not self.set_VF([0, 0, 15, 0, 0, 0]):
            rospy.INFO("failed to set virtual force")
        rospy.sleep(0.5)
        while not self.set_cur_pose_as_equilibrium():
            rospy.INFO("failed to set equilibrium")

        while not self.set_VF([0, 0, 0, -2, -2, 0]):
            rospy.INFO("failed to set virtual force")
        rospy.sleep(2)

        while not self.set_cur_pose_as_equilibrium():
            rospy.INFO("failed to set equilibrium")
        while not self.set_VF([0, 0, 15, 0, 0, 0]):
            rospy.INFO("failed to set virtual force")
        rospy.sleep(0.5)
        self.insert_state = self.insertCurrentState(100)
        while self.insert_state[3] > -5:
            self.insert_state = self.insertCurrentState(20)
        self.start_insert = rospy.Time.now()
        return self.insert_state
    
    def postionMotion(self, origin, target):
        while not self.activate_admittance_control(False):
            rospy.INFO("failed to activate admittance control")
        rospy.sleep(0.2)
        while not self.activate_virtual_force(False):
            rospy.INFO("failed to activate virtual force")
        rospy.sleep(0.2)
        if self.count % 4 == 0:
            while not self.position_ctrl(origin):
                rospy.INFO("failed to position control")
            rospy.sleep(0.2)
            random.shuffle(self.area)
        origin = self.randomArea(origin, self.area[self.count%4])

        target.position.z += 0.247 + 0.005
        origin.position.z += 0.005

        while not self.position_ctrl(target):
            rospy.INFO("failed to position control")
        rospy.sleep(0.2)
        while not self.set_refer_pose():
            rospy.INFO("failed to set refer pose")
        rospy.sleep(0.2)
        while not self.position_ctrl(origin):
            rospy.INFO("failed to position control")
        rospy.sleep(0.2)
        while not self.zero_ft_sensor():
            rospy.INFO("failed to zero ft sensor")
        rospy.sleep(0.2)
        while not self.set_cur_pose_as_equilibrium():
            rospy.INFO("failed to set equilibrium")
        rospy.sleep(0.2)
        while not self.activate_admittance_control(True):
            rospy.INFO("failed to activate admittance control")
        rospy.sleep(0.2)
        while not self.activate_virtual_force(True):
            rospy.INFO("failed to activate virtual force")
        rospy.sleep(0.2)

    def finish(self):
        while not self.activate_admittance_control(False):
            rospy.INFO("failed to activate admittance control")
        rospy.sleep(0.2)
        while not self.activate_virtual_force(False):
            rospy.INFO("failed to activate virtual force")
        rospy.sleep(0.2)
        while not self.zero_ft_sensor():
            rospy.INFO("failed to zero ft sensor")
        rospy.sleep(0.2)

    def randomArea(self, origin, area):
        r = 0.005
        theta = random.uniform(0, math.pi/2)
        x = r*math.cos(theta)
        y = r*math.sin(theta)

        if area == 1:
            origin.position.x += x
            origin.position.y += y
        elif area == 2:
            origin.position.x += -x
            origin.position.y += y
        elif area == 3:
            origin.position.x += x
            origin.position.y += -y
        elif area == 4:
            origin.position.x += -x
            origin.position.y += -y
        print("area: {}".format(area))
        return origin
    
    def getCurrentState(self, num):
        torque_x = []
        torque_y = []
        force_x = []
        force_y = []
        force_z = []
        for _ in range(num):
            wrench = wrench_state
            torque_x.append(wrench.wrench.torque.x)
            torque_y.append(wrench.wrench.torque.y)
            force_x.append(wrench.wrench.force.x)
            force_y.append(wrench.wrench.force.y)
            force_z.append(wrench.wrench.force.z)
            rospy.sleep(0.01)
        tx = sum(torque_x)/num
        ty = sum(torque_y)/num
        fx = sum(force_x)/num
        fy = sum(force_y)/num
        fz = sum(force_z)/num

        pose = pose_state
        wrench = wrench_state
        return np.array([math.tanh(pose.position.x), math.tanh(pose.position.y), math.tanh(pose.position.z), 
                        fx, fy, fz, math.tanh(tx), math.tanh(ty), self.action[0]/force_scale, self.action[1]/force_scale])
    
    def insertCurrentState(self, num):
        torque_x = []
        torque_y = []
        force_x = []
        force_y = []
        force_z = []
        for _ in range(num):
            wrench = wrench_state
            torque_x.append(wrench.wrench.torque.x)
            torque_y.append(wrench.wrench.torque.y)
            force_x.append(wrench.wrench.force.x)
            force_y.append(wrench.wrench.force.y)
            force_z.append(wrench.wrench.force.z)
            rospy.sleep(0.01)
        tx = sum(torque_x)/num
        ty = sum(torque_y)/num
        fx = sum(force_x)/num
        fy = sum(force_y)/num
        fz = sum(force_z)/num

        pose = pose_state
        wrench = wrench_state
        return np.array([math.tanh(pose.position.z), fx, fy, fz, math.tanh(tx), math.tanh(ty), 
                         self.insert_action[0]/torque_scale, self.insert_action[1]/torque_scale])

class StateThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        rospy.init_node('env_listener', anonymous=True)
        rospy.Subscriber("/ur_cartesian_pose", Pose, self.poseCallback)
        rospy.Subscriber("/wrench_filter", WrenchStamped, self.wrenchCallback)

    def run(self):
        rospy.spin()

    def poseCallback(self, data):
        global pose_state
        pose_state = data

    def wrenchCallback(self, data):
        global wrench_state
        wrench_state = data
