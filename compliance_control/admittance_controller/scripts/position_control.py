#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, sys
import moveit_commander
from moveit_commander import MoveGroupCommander
from self_defined_msgs.srv import position_ctrl_srv, position_ctrl_srvResponse
from copy import deepcopy

class URMove(object):
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.arm = MoveGroupCommander('manipulator')
        self.arm.set_pose_reference_frame('base_link')
        self.end_effector_link = "tool0"
        self.arm.allow_replanning(True)
        self.arm.set_goal_position_tolerance(0.001)
        self.arm.set_goal_orientation_tolerance(0.001)
        self.arm.set_end_effector_link(self.end_effector_link)
        print(self.arm.get_current_pose('tool0'))
        rospy.Service('/UR_position_control', position_ctrl_srv, self.line)
    
    def line(self, req):
        self.arm.set_start_state_to_current_state()
        start_pose = self.arm.get_current_pose(self.end_effector_link).pose
        waypoints = []
        wpose = deepcopy(start_pose)
        wpose = req.pose
        waypoints.append(deepcopy(wpose))
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

if __name__ == "__main__":
    try:
        rospy.init_node('URMove')
        server = URMove()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("program interrupted before completion")   