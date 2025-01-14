

#include <ur5e_sim/ur_hw_interface.h>

namespace ur_sim_ns
{

  urHWInterface::urHWInterface(ros::NodeHandle &nh, urdf::Model *urdf_model)
      : ros_control_boilerplate::GenericHWInterface(nh, urdf_model),
        position_controller_running_(false),
        velocity_controller_running_(false)
  {

    telemetry_sub = nh.subscribe("/teensy/urTelemetry", 1, &urHWInterface::telemetryCallback, this);

    cmd_pub = nh.advertise<self_defined_msgs::armCmd>("/teensy/armCmd", 3);
    ROS_INFO("urHWInterface declared.");
  }

  void urHWInterface::telemetryCallback(const self_defined_msgs::urTelemetry::ConstPtr &msg)
  {

    /*
   #Header header
   float32[6] angle # degrees
   float32[6] vel # deg/s
   float32[6] current # amps
   #time armReadTimestamp
   time startSyncTime
   uint32 isrTicks # this would overflow if the robot is left on for 497 days straight at 100 hz
   uint8 bufferHealth
   */

    for (int i = 0; i < num_joints_; i++)
    {
      joint_velocity_[i] = msg->vel[i]; // declared in GenericHWInterface
      joint_position_[i] = msg->angle[i];
    }

    bufferHealth = msg->bufferHealth;
  }

  void urHWInterface::init()
  {
    // Call parent class version of this function
    /*
    this looks at controller yaml "hardware" namespace to get "joints". from this list the number of joints is known so hardware interfaces are initialized.
    it starts a joint_state, position, velocity and effort iterface. joint limits are also grabbed from parameter server urdf if urdf=NULL.
    */
    ros_control_boilerplate::GenericHWInterface::init();

    // array for storing previous state (for velocity calculation)
    joint_position_prev_.resize(joint_position_.size());

    ROS_INFO("urHWInterface initiated.");
  }

  void urHWInterface::read(ros::Duration &elapsed_time)
  {
    // ros::spinOnce(); //is not required here because of asyncspinner
  }

  void urHWInterface::write(ros::Duration &elapsed_time)
  {
    static self_defined_msgs::armCmd cmd_;

    /*
    float32[6] current #amps
    float32[6] accel #deg/s^2
    float32[6] vel #deg/s
    float32[6] angle #deg
    uint32 msg_ctr # count sent msgs to detected missed messages

    // Available Commands // from ros_control arch
    std::vector<double> joint_position_command_;
    std::vector<double> joint_velocity_command_;
    std::vector<double> joint_effort_command_;
    */

    /*
    caculate at a much higher rate then needed. then only send ones needed to fill buffer.

    */
    if (position_controller_running_)
    {
      // only publish a msg if it has a change
      bool change_detected = false;
      for (int i = 0; i < num_joints_; i++)
      {
        if (joint_position_prev_[i] != joint_position_command_[i])
        {
          change_detected = true;
          i = num_joints_; // exit loop
        }
      }

      // if a new msg is available then send it
      if (change_detected)
      {
        for (int i = 0; i < num_joints_; i++)
        {
          cmd_.angle[i] = joint_position_command_[i];
          cmd_.vel[i] = (fabs(joint_position_command_[i] - joint_position_prev_[i])) / elapsed_time.toSec(); // (must be positive for aubo) joint_velocity_command_[i]*RAD_TO_DEG; joint_velocity_command_[i] calculate my own velocities
          cmd_.accel[i] = 4;                                                                                 // a max acceleration limit (must be positive for aubo)

          // cmd_.eff[i]=joint_effort_command_[i];

          joint_position_prev_[i] = joint_position_command_[i];
          // ROS_INFO_STREAM("POS: " << joint_position_command_[0]);
        }

        // if this point is needed then send it
        if (bufferHealth < DESIRED_BUFFERED_POINTS)
        {
          cmd_.msg_ctr = cmd_.msg_ctr + 1;
          cmd_pub.publish(cmd_);
        }

      } // changed detected
    }   // position controller running

    if (velocity_controller_running_)
    {
      // TODO velocity controller
        for (int i = 0; i < num_joints_; i++)
        { 
          cmd_.vel[i] = joint_velocity_command_[i];
          cmd_.angle[i] = joint_position_prev_[i] + cmd_.vel[i] * elapsed_time.toSec();
          joint_position_prev_[i] = cmd_.angle[i];
          cmd_.accel[i] = 4;                                                                                 // a max acceleration limit (must be positive for aubo)
        }
        if (bufferHealth < DESIRED_BUFFERED_POINTS)
        {
          cmd_.msg_ctr = cmd_.msg_ctr + 1;
          cmd_pub.publish(cmd_);
        }
    }

  } // write

  void urHWInterface::enforceLimits(ros::Duration &period)
  {
    // Enforces position and velocity
    // pos_jnt_sat_interface_.enforceLimits(period);
  }

  void urHWInterface::doSwitch(const std::list<hardware_interface::ControllerInfo> &start_list,
                               const std::list<hardware_interface::ControllerInfo> &stop_list)
  {
    ROS_INFO("Controller switched");
    for (auto &controller_it : stop_list)
    {
      for (auto &resource_it : controller_it.claimed_resources)
      {
        ROS_INFO_STREAM("hardware_interface:stopped: " << resource_it.hardware_interface);
        if (resource_it.hardware_interface == "hardware_interface::PositionJointInterface")
        {
          position_controller_running_ = false;
        }
        if (resource_it.hardware_interface == "hardware_interface::VelocityJointInterface")
        {
          velocity_controller_running_ = false;
        }
      }
    }

    for (auto &controller_it : start_list)
    {
      for (auto &resource_it : controller_it.claimed_resources)
      {
        ROS_INFO_STREAM("hardware_interface:stopped: " << resource_it.hardware_interface);
        if (resource_it.hardware_interface == "hardware_interface::PositionJointInterface")
        {
          position_controller_running_ = true;
        }
        if (resource_it.hardware_interface == "hardware_interface::VelocityJointInterface")
        {
          velocity_controller_running_ = true;
        }
      }
    }
  }

} // namespace
