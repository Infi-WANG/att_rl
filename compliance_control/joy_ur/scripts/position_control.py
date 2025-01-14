#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rospy, sys
import moveit_commander
from moveit_commander import MoveGroupCommander
from self_defined_msgs.srv import position_ctrl_srv, position_ctrl_srvResponse, SetDoubleArray, SetDoubleArrayResponse, SetDoubleArrayRequest
from geometry_msgs.msg import PoseStamped, Pose
from std_srvs.srv import Trigger, TriggerResponse
from copy import deepcopy

class URMove(object):
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.arm = MoveGroupCommander('manipulator')
        self.arm.set_pose_reference_frame('base_link')
        self.end_effector_link = "tool0"
        self.arm.allow_replanning(True)
        self.arm.set_goal_position_tolerance(0.0001)
        self.arm.set_goal_orientation_tolerance(0.0001)
        self.arm.set_end_effector_link(self.end_effector_link)
        
        self.home ={"elbow_joint": 1.6571649808236542, 
                    "shoulder_lift_joint":-1.2880557128294647,
                    "shoulder_pan_joint":-0.6246967740568543,
                    "wrist_1_joint": -1.8882637629599257,
                    "wrist_2_joint": -1.4998743378314696,
                    "wrist_3_joint": 0.34014125603193435      }
        
        self.home_pose = Pose()
        self.home_pose.position.x = 0.520
        self.home_pose.position.y = -0.015 #-0.020
        self.home_pose.position.z = 0.285
        self.home_pose.orientation.x = 0    
        self.home_pose.orientation.y = 1    
        self.home_pose.orientation.z = 0    
        self.home_pose.orientation.w = 0    
        
        # print(self.arm.get_current_pose('peg_link'))
        # print(self.arm.get_current_joint_values())
        rospy.Service('/UR_position_control', position_ctrl_srv, self.line)
        rospy.Service('/UR_joints_control', SetDoubleArray, self.go_to_joint_state)
        rospy.Service('/UR_go_home', Trigger, self.go_home)
        # self.set_ready()
        # for _ in range(3):
        #     self.line()

    def go_home(self, _):
        res = TriggerResponse()
        self.arm.set_joint_value_target(self.home)
        traj = self.arm.plan()
        self.arm.execute(traj)
        
        self.arm.set_start_state_to_current_state()
        waypoints = [] 
        waypoints.append(deepcopy(self.home_pose))
        self.plan_execute(waypoints)
        
        res.success = True
        return res

    def line(self, req):
        self.arm.set_start_state_to_current_state()
        start_pose = self.arm.get_current_pose(self.end_effector_link).pose
        waypoints = []
        self.home_pose = deepcopy(start_pose)
        self.home_pose = req.pose
        waypoints.append(deepcopy(self.home_pose))
        success = self.plan_execute(waypoints)
        return position_ctrl_srvResponse(success)
        
    def plan_execute(self, waypoints): 
        fraction = 0.0   #路径规划覆盖率
        maxtries = 100   #最大尝试规划次数
        attempts = 0     #已经尝试规划次数   
        # 尝试规划一条笛卡尔空间下的路径，依次通过所有路点
        while fraction < 1.0 and attempts < maxtries:
            (plan, fraction) = self.arm.compute_cartesian_path (
                                    waypoints,   # waypoint poses，路点列表
                                    0.01,        # eef_step，终端步进值
                                    0.0,         # jump_threshold，跳跃阈值
                                    True)        # avoid_collisions，避障规划  
            attempts += 1
            if attempts % 10 == 0:
                rospy.loginfo("Still trying after " + str(attempts) + " attempts...")
                         
        if fraction == 1.0:
            # rospy.loginfo("Path computed successfully. Moving the arm.")
            self.arm.execute(plan)
            # rospy.loginfo("Path execution complete.")
            return True
        else:
            rospy.loginfo("Path planning failed with only " + str(fraction) + " success after " + str(maxtries) + " attempts.")
            return False
        
    def go_to_joint_state(self, req):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        res = SetDoubleArrayResponse()
        if len(req.array) != 6:
            rospy.WARN("Wrong input joint state")
            res.success = False
            return res
        target = {}
        # for i in range(6):
        #     joint_goal[i] = req.array[i]
        target["shoulder_pan_joint"] = req.array[0]
        target["shoulder_lift_joint"] = req.array[1]
        target["elbow_joint"] = req.array[2]
        target["wrist_1_joint"] = req.array[3]
        target["wrist_2_joint"] = req.array[4]
        target["wrist_3_joint"] = req.array[5]
        
        self.arm.set_joint_value_target(target)
        traj = self.arm.plan()
        self.arm.execute(traj)
        
        res.success = True
        return res


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    all_equal = True
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is Pose:
        return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

    return True

def pose_to_list(pose_msg):
    pose = []
    pose.append(pose_msg.position.x)
    pose.append(pose_msg.position.y)
    pose.append(pose_msg.position.z)
    pose.append(pose_msg.orientation.x)
    pose.append(pose_msg.orientation.y)
    pose.append(pose_msg.orientation.z)
    pose.append(pose_msg.orientation.w)
    return pose


if __name__ == "__main__":
    try:
        rospy.init_node('URMove')
        server = URMove()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("program interrupted before completion")   