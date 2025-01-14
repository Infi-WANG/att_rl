#!/usr/bin/env python

import rospy
from admittance_controller.msg import joint_trajectory
from sensor_msgs.msg import JointState
import copy

if __name__ == '__main__':

    test_trajectory = joint_trajectory()

    JointState_temp = JointState()
    JointState_temp.name = ["1", "2", "3", "4", "5", "6"]
    JointState_temp.header.stamp.secs = 0
    JointState_temp.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    JointState_temp.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    test_trajectory.trajectory.append(copy.deepcopy(JointState_temp))

    JointState_temp.header.stamp.secs = 1
    JointState_temp.position = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    JointState_temp.velocity = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    test_trajectory.trajectory.append(copy.deepcopy(JointState_temp))

    JointState_temp.header.stamp.secs = 3
    JointState_temp.position = [0.8, 0.0, 0.0, 0.0, 0.0, 0.0]
    JointState_temp.velocity = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    test_trajectory.trajectory.append(copy.deepcopy(JointState_temp))

    JointState_temp.header.stamp.secs = 4
    JointState_temp.position = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0]
    JointState_temp.velocity = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    test_trajectory.trajectory.append(copy.deepcopy(JointState_temp))

    JointState_temp.header.stamp.secs = 6
    JointState_temp.position = [1.8, 0.0, 0.0, 0.0, 0.0, 0.0]
    JointState_temp.velocity = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    test_trajectory.trajectory.append(copy.deepcopy(JointState_temp))

    JointState_temp.header.stamp.secs = 7
    JointState_temp.position = [1.9, 0.0, 0.0, 0.0, 0.0, 0.0]
    JointState_temp.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    test_trajectory.trajectory.append(copy.deepcopy(JointState_temp))

    test_trajectory.velocity_scaling_percentage = 100

    rospy.init_node('trajectory_test_node', anonymous=True)
    pub = rospy.Publisher('/admittance_controller/trajectory_execution', joint_trajectory, queue_size=1)

    rospy.sleep(0.5)
    pub.publish(test_trajectory)
