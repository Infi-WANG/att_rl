#include <ros/ros.h>
#include <self_defined_msgs/urTelemetry.h>
#include <self_defined_msgs/armCmd.h> 
#include <geometry_msgs/WrenchStamped.h>
#include <deque> 

std::deque<self_defined_msgs::armCmd> cmd_queue;

//// current position callback ////
// for subscribing to telemetry
ros::Publisher telem_pub;
ros::Publisher wrench_pub;

void cmdCallback(const self_defined_msgs::armCmd::ConstPtr &msg){
 
    cmd_queue.push_back(*msg);


}


self_defined_msgs::armCmd current_cmd;
self_defined_msgs::urTelemetry telem;
geometry_msgs::WrenchStamped wrench;
void ISRCallback(const ros::TimerEvent&)
{	

    if( !cmd_queue.empty() ){
        current_cmd=cmd_queue.front(); // read from queue


        for(int i=0; i<current_cmd.angle.size(); i++){
            telem.angle[i]=current_cmd.angle[i];
            telem.vel[i]=current_cmd.vel[i]; 
        }


        cmd_queue.pop_front(); // remove (pop off) the first element of the queue

    }

    telem.bufferHealth=cmd_queue.size();
    telem_pub.publish(telem);

    wrench.header.frame_id = "tool0";
    wrench.header.stamp = ros::Time::now();

    wrench.wrench.force.x = 0.0;
    wrench.wrench.force.y = 0.0;
    wrench.wrench.force.z = 0.0;
    wrench.wrench.torque.x = 0.0;
    wrench.wrench.torque.y = 0.0;
    wrench.wrench.torque.z = 0.0;

    wrench_pub.publish(wrench);

}

//TODO VELOCITY CONTROLLER

 
// create a timer callback they fires at 181 hz . if points are recieved then add them to a buffer. read from this buffer from the timer. 

//// main ////
int main(int argc, char **argv) {
    ros::init(argc, argv, "aubo_sim_echo");

    // prep for ROS communcation
    ros::NodeHandle n; 

    ros::Subscriber cmd_sub = n.subscribe("/teensy/armCmd", 10, cmdCallback); // robot feedback

    telem_pub = n.advertise<self_defined_msgs::urTelemetry>("/teensy/urTelemetry", 10);
    wrench_pub = n.advertise<geometry_msgs::WrenchStamped>("/wrench",1);

    ros::Timer isr_timer = n.createTimer(ros::Duration(0.0055), ISRCallback, false); // true/false oneshot
	
 
    for(int i=0; i<6; i++){
        telem.angle[i]=0;
        telem.vel[i]=0; 
    }

 
    ros::spin();
 

} // end main

 
















