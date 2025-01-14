#pragma once

#include "ros/ros.h"
#include <actionlib/client/simple_action_client.h>
// #include "db_self_msgs/upper_control_cmdAction.h"

#include <sensor_msgs/Joy.h>
// #include "db_self_msgs/hand_state.h"

#include <std_srvs/SetBool.h>
#include <std_srvs/Trigger.h>
#include <std_msgs/Float64MultiArray.h>
#include <geometry_msgs/Pose.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Float64MultiArray.h>

#include <Eigen/Eigen>
#include <eigen_conversions/eigen_msg.h>

#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
// #include <self_defined_msgs/SetDoubleArray.h>
// #include <robotiq_2f_gripper_control/Robotiq2FGripper_robot_input.h>

#include <string>

// #include <thread>
// #include <mutex>

// axes
#define LS_H_BIT  0 //LEFT 1 RIGHT -1
#define LS_V_BIT  1 //UP 1 DOWN -1
#define LT_BIT    2 //UP 1 DOWN -1

#define RS_H_BIT 3
#define RS_V_BIT 4
#define RT_BIT 5 

#define UP_DOWN_BIT 7 //UP 1 DOWN -1
#define LEFT_RIGHT_BIT 6 //LEFT 1 RIGHT -1

//buttom
#define A_BIT 0
#define B_BIT 1
#define X_BIT 2
#define Y_BIT 3

#define LB_BIT 4
#define RB_BIT 5

#define START_BIT 7
#define BACK_BIT  6

#define LS_B_BIT 9
#define RS_B_BIT 10

#define X_GAIN 0.05
#define Y_GAIN 0.05
// #define Z_GAIN -0.025
#define Z_GAIN -0.025
#define RX_GAIN 0.1
#define RY_GAIN -0.1
#define RZ_GAIN -0.1

#define UP(X)       0 == msg->buttons.at(X) && 1 == joy_last_state.buttons.at(X)
#define DOWN(X)     1 == msg->buttons.at(X) && 0 == joy_last_state.buttons.at(X) 

#define UP_BUTTOM       joy_last_state.axes.at(UP_DOWN_BIT)!=1.0 && msg->axes.at(UP_DOWN_BIT) == 1.0
#define DOWN_BUTTOM     joy_last_state.axes.at(UP_DOWN_BIT)!=-1.0 && msg->axes.at(UP_DOWN_BIT) == -1.0
#define LEFT_BUTTOM     joy_last_state.axes.at(LEFT_RIGHT_BIT)!=1.0 && msg->axes.at(LEFT_RIGHT_BIT) == 1.0
#define RIGHT_BUTTOM    joy_last_state.axes.at(LEFT_RIGHT_BIT)!=-1.0 && msg->axes.at(LEFT_RIGHT_BIT) == -1.0

#define HAND_POSE_X_RANGE  0.3  //m
#define HAND_POSE_Y_RANGE  0.3  //m
#define HAND_POSE_Z_RANGE  0.4  //m

class joy_ur
{
private:
    ros::NodeHandle nh, nh_;

    bool joy_control_on;

    // actionlib::SimpleActionClient<db_self_msgs::upper_control_cmdAction> ac_;
    // db_self_msgs::upper_control_cmdGoal goal_;
    ros::ServiceServer set_current_pose_as_origin;
    //arm
    ros::ServiceClient admittance_controller_switch_client, follow_trajectory_online_switch_client, clear_TOL_equilibrium_offset_client, go_home_client,
                        play_client, stop_client, zero_tf_client, set_EEF_ref_pose_client, record_SAP_data_client;
                        
                        // gripper_joy_stick_switch_client, gripper_state_client;
    // ros::Publisher cartesian_move_v_pub_l;
    ros::Publisher target_pose_pub, gripper_target_position_pub;
    tf2_ros::TransformBroadcaster broadcaster;

    // ros::ServiceClient estop_client_client;
    ros::Subscriber joy_sub, cur_pose_sub;

    // ----- DATA ---- //
    geometry_msgs::Pose target_pose_msg;
    // std_msgs::Float64MultiArray cartesian_move_v;
    std::vector<double> cartesian_move_vel;  

    sensor_msgs::Joy joy_last_state;
    double mode_gain;
    
    Eigen::Isometry3d target_pose, cur_TCP_pose;
    
    bool cartesian_started;
    bool TCP_valid;
    bool TCP_based;  //TCP based or World based
    bool real_robot;

    // ----- FUNCTIONS ---- //
    bool switch_follow_trajectory_online_controller(bool on_off);
    bool switch_admittance_controller(bool on_off);
    bool set_joy_control(bool on_off);

    void compute_target_pose();
    void pub_pose_msgs();
    void wait_dependence();

    bool reset_current_as_origin();

    bool go_home();
    bool zero_ft_sensor();
    bool set_cur_as_EEF_ref_pose();
    bool activate_sap_data_record(bool on_off);

    bool set_current_pose_as_origin_CB(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);

    // ----- UTIL --------- //
    double fsat(double value, double min = -0.1, double max = 0.1);
    int sat(int value, int min=0, int max=255);

    // void gripper_open(void);
    // void gripper_close(void);
    // void gripper_hold(void);

    std::vector<double> low_pass_filter(std::vector<double> input_vec);
    std::vector<double> low_pass_filter_position(std::vector<double> input_vec);

public:
    joy_ur(/* args */);
    ~joy_ur();

    //-----ROS SUB CALLBACK ------//
    // void gripper_states_Callback(const robotiq_2f_gripper_control::Robotiq2FGripper_robot_inputConstPtr &msg);
    void joy_Callback(const sensor_msgs::Joy::ConstPtr &msg);   
    // void hand_gesture_Callback(const db_self_msgs::hand_state::ConstPtr &msg);
    void hand_pose_Callback(const std_msgs::Float64MultiArray::ConstPtr &msg);
    void cur_TCP_pose_sub_Callback(const geometry_msgs::Pose::ConstPtr &msg);

    //-------- SPIN ------- //
    void spinner();
};
