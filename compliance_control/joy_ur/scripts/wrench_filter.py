#!/usr/bin/env python
import rospy
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Trigger, TriggerResponse
import numpy as np
      
class ROSNode:
    def __init__(self):
        rospy.init_node("wrench_filter")
        rospy.Subscriber("/wrench_origin", WrenchStamped, self.wrench_CB, queue_size=10)
        
        self.wrench_pub_ = rospy.Publisher('/wrench', WrenchStamped, queue_size=10)
        srv_server_handler = rospy.Service("/ur_hardware_interface/zero_ftsensor", Trigger, self.zero_ft_CB)
        
        self._filter = Filter(list_len = 100)
        
        self.wrench_zero = WrenchStamped()
        self.wrench_cur = WrenchStamped()
        
        self.wrench_recv = False
        
        rospy.loginfo('Wrench filter starts successfully')

    def wrench_CB(self, msg):
        # rospy.loginfo('Received message')
        # msg = self._filter(msg)
        self.wrench_cur.wrench.force.x = msg.wrench.force.x - self.wrench_zero.wrench.force.x
        self.wrench_cur.wrench.force.y = msg.wrench.force.y - self.wrench_zero.wrench.force.y
        self.wrench_cur.wrench.force.z = msg.wrench.force.z - self.wrench_zero.wrench.force.z
        self.wrench_cur.wrench.torque.x = msg.wrench.torque.x - self.wrench_zero.wrench.torque.x
        self.wrench_cur.wrench.torque.y = msg.wrench.torque.y - self.wrench_zero.wrench.torque.y
        self.wrench_cur.wrench.torque.z = msg.wrench.torque.z - self.wrench_zero.wrench.torque.z
        self.wrench_pub_.publish(self.wrench_cur)
        self.wrench_recv = True
        
    def zero_ft_CB(self, request):
        response = TriggerResponse()
        if not self.wrench_recv:
            response.success = False
            response.message = "Not wrench received yet"
        
        self.wrench_zero.wrench.force.x += self.wrench_cur.wrench.force.x
        self.wrench_zero.wrench.force.y += self.wrench_cur.wrench.force.y
        self.wrench_zero.wrench.force.z += self.wrench_cur.wrench.force.z
        self.wrench_zero.wrench.torque.x += self.wrench_cur.wrench.torque.x
        self.wrench_zero.wrench.torque.y += self.wrench_cur.wrench.torque.y
        self.wrench_zero.wrench.torque.z += self.wrench_cur.wrench.torque.z

        response.success = True
        return response

class Filter():
    def __init__(self, list_len):
        self.x_ = []
        self.y_ = []
        self.z_ = []
        self.wx_ = []
        self.wy_ = []
        self.wz_ = []
        
        self.len = list_len
        self.__cur = WrenchStamped()
        
    def __call__(self, data):
        self.x_.append(data.wrench.force.x)
        self.y_.append(data.wrench.force.y)
        self.z_.append(data.wrench.force.z)
        self.wx_.append(data.wrench.torque.x)
        self.wy_.append(data.wrench.torque.y)
        self.wz_.append(data.wrench.torque.z)
        
        if len(self.x_) > self.len:
            self.x_.pop(0)
            self.y_.pop(0)
            self.z_.pop(0)
            self.wx_.pop(0)
            self.wy_.pop(0)
            self.wz_.pop(0)
            
        size = len(self.x_)
        self.__cur.wrench.force.x = sum(self.x_)/size
        self.__cur.wrench.force.y = sum(self.y_)/size
        self.__cur.wrench.force.z = sum(self.z_)/size
        self.__cur.wrench.torque.x = sum(self.wx_)/size
        self.__cur.wrench.torque.y = sum(self.wy_)/size
        self.__cur.wrench.torque.z = sum(self.wz_)/size
        
        return self.__cur
            
if __name__ == "__main__":
    name_node = ROSNode()
    rospy.spin()