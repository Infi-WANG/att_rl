#include <ros_control_boilerplate/generic_hw_control_loop.h>
#include <ur5e_sim/ur_hw_interface.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ur_hw_interface");
  ros::NodeHandle nh; 


  // NOTE: We run the ROS loop in a separate thread as external calls such
  // as service callbacks to load controllers can block the (main) control loop
  ros::AsyncSpinner spinner(3);
  spinner.start();

  // Create the hardware interface specific to your robot
  std::shared_ptr<ur_sim_ns::urHWInterface> ur_hw_interface_instance 
   (new ur_sim_ns::urHWInterface(nh));
  ur_hw_interface_instance->init(); // size and register required interfaces inside generic_hw_interface.cpp


  // Start the control loop
  ros_control_boilerplate::GenericHWControlLoop control_loop(nh, ur_hw_interface_instance);
  control_loop.run(); // Blocks until shutdown signal recieved -> read -> update -> write -> repeat inside generic_hw_control_loop.cpp

  return 0;
}
