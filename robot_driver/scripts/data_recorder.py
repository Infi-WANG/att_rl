#!/usr/bin/env python3
import rospy, rospkg
import numpy as np
import pickle

from sensor_msgs.msg import Image
from sensor_msgs.msg import Joy
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import Pose
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from rospy.numpy_msg import numpy_msg

from copy import deepcopy
from dataclasses import dataclass
import datetime
from os.path import join

@dataclass
class DataRecorded():
    image: np.ndarray
    image_plus1: np.ndarray
    wrench: np.ndarray
    wrench_plus1: np.ndarray
    pose: np.ndarray
    pose_plus1: np.ndarray
    action: list
    
class RosNode:
    def __init__(self, action_list_length):
        rospy.init_node("data_recorder")
        rospy.loginfo("Starting data recorder.")
        
        rospack = rospkg.RosPack()
        self.path = join(rospack.get_path('robot_driver'), "data")
    
        self.data_list = []
        
        self.image_sub = rospy.Subscriber("/global/camera1/image_raw", numpy_msg(Image), self.image_cb)
        self.action_sub = rospy.Subscriber("/joy", Joy, self.action_cb)
        self.wrench_sub = rospy.Subscriber("/wrench_filter", WrenchStamped, self.wrench_cb)
        self.pose_sub = rospy.Subscriber("/ur_cartesian_pose_rel", Pose, self.pose_cb)
        srv_server_handler = rospy.Service("/activate_sap_data_record", SetBool, self.activate_record_cb)
        
        self.action_list_length = action_list_length
        self.cur_action = np.zeros(3)
        self.action_list = []
        self.action_valid = False
        self.action_rec = False
        
        self.image_rec = False
        self.image_data_valid = False
        
        self.wrench_rec = False
        self.wrench_data_valid = False
        
        self.pose_rec = False
        self.pose_data_valid = False
        
        self.begin_to_record = False
        
        self.__i = 0

        rospy.Timer(rospy.Duration(1.0 / 10), self.record_cb)
        
    def activate_record_cb(self, request: SetBoolRequest):
        rospy.loginfo('Received request:' + str(request.data))
        response = SetBoolResponse()
        response.success = True
        if request.data:
            self.begin_to_record = True
            rospy.loginfo("Begin to record SAP data")
        else:
            self.begin_to_record = False
            if len(self.data_list) > 0:
                filename_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M") + "_L" + str(len(self.data_list))
                file_path = join(self.path, f"data_{filename_stamp}.pkl")
                with open(file_path, 'wb') as f:
                    pickle.dump(self.data_list ,f)
                rospy.loginfo("SAP data saved at %s", file_path)
                rospy.loginfo("Data length: %d", len(self.data_list))
                self.data_list.clear()
                response.success = True
            else:
                response.success = False
                rospy.logwarn("No data received")   
        return response
        
    def action_cb(self, msg: Joy):
        self.cur_action[0] = msg.axes[0]  # X
        self.cur_action[1] = msg.axes[1]  # Y
        self.cur_action[2] = (msg.axes[5] - msg.axes[2])/2  # Z
        self.action_rec = True
        # print(self.cur_action)
        
    def image_cb(self, data):
        self.cur_image = deepcopy(np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1))
        # self.__i += 1
        # print("image count: {}".format(self.__i))
        self.image_rec = True
        
    def wrench_cb(self, data: WrenchStamped):
        self.cur_wrench = np.array([data.wrench.force.x,
                                    data.wrench.force.y,
                                    data.wrench.force.z,
                                    data.wrench.torque.x,
                                    data.wrench.torque.y,
                                    data.wrench.torque.z])
        self.wrench_rec = True
        
    def pose_cb(self, data: Pose):
        self.cur_pose = np.array(  [data.position.x,
                                    data.position.y,
                                    data.position.z,
                                    data.orientation.x,
                                    data.orientation.y,
                                    data.orientation.z,
                                    data.orientation.w])
        self.pose_rec = True
        

    def record_cb(self, event):
        if self.begin_to_record and self.image_data_valid and self.wrench_data_valid and self.pose_data_valid and self.action_valid:
            # print("-----------------------")

            self.data_list.append(deepcopy(DataRecorded(image=self.last_image,
                                                        image_plus1=self.cur_image,
                                                        wrench=self.last_wrench,
                                                        wrench_plus1=self.cur_wrench,
                                                        pose=self.last_pose,
                                                        pose_plus1=self.cur_pose,
                                                        action=self.action_list)) 
                                  )
        if self.action_rec:
            self.action_list.append(deepcopy(self.cur_action))
            if len(self.action_list) > self.action_list_length:
                self.action_list.pop(0)
                self.action_valid = True
        
        if self.image_rec:
            self.last_image = deepcopy(self.cur_image)
            self.image_data_valid = True
        
        if self.wrench_rec:
            self.last_wrench = deepcopy(self.cur_wrench)
            self.wrench_data_valid = True
            
        if self.pose_rec:
            self.last_pose = deepcopy(self.cur_pose)
            self.pose_data_valid = True


if __name__ == "__main__":
    ros_node = RosNode(action_list_length=5)
    rospy.spin()