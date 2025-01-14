#ifndef ADMITTANCE_CONTROLLER_H
#define ADMITTANCE_CONTROLLER_H

#include <signal.h>
#include <fstream>
#include <numeric>
#include <math.h>

#include "spline_interpolation/spline.h" /* https://kluge.in-chemnitz.de/opensource/spline/ */
#include "spline_interpolation/pchip.h"
#include "admittance_controller/singularity_avoidance.h"
#include <ros/ros.h>
#include <ros/package.h>

#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <geometry_msgs/WrenchStamped.h>
#include <geometry_msgs/Pose.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <sensor_msgs/JointState.h>

#include "self_defined_msgs/parameter_msg.h"
#include "self_defined_msgs/joint_trajectory.h"
#include "self_defined_msgs/parameter_srv.h"
#include "self_defined_msgs/set_equilibrium_pose_srv.h"
#include "self_defined_msgs/set_VF_srv.h"
#include "self_defined_msgs/get_MKB.h"
#include "self_defined_msgs/set_MKB.h"

#include <std_srvs/Trigger.h>
#include <std_srvs/SetBool.h>
#include <controller_manager_msgs/SwitchController.h>
#include <controller_manager_msgs/ListControllers.h>

#include <actionlib/client/simple_action_client.h>
#include <actionlib/server/simple_action_server.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <self_defined_msgs/compliance_actionAction.h>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/collision_detection_fcl/collision_world_fcl.h>
#include <moveit/collision_detection_fcl/collision_robot_fcl.h>
#include <moveit/collision_detection/collision_tools.h>

#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>

#include <Eigen/Eigen>
#include <eigen_conversions/eigen_msg.h>

using namespace Eigen;


typedef Matrix<double, 7, 1> Vector7d;
typedef Matrix<double, 6, 1> Vector6d;
typedef Matrix<double, 6, 6> Matrix6d;
typedef Array<double, 6, 1> Array6d;
enum Admittance_Mode {Fixed_Point, Follow_Trajectory_offline, Follow_Trajectory_online};
enum Collision_State{
    NORMAL,
    COLLISIONstate,
    SOLVING
  };

struct extra_data_keypoint {
    double time_keypoint;
    double data_value;
};

#define GET_VARIABLE_NAME(Variable) (#Variable)

class LowPassFilter
{
private:
    int length;
    std::vector<Vector6d> filter_queue;
public:
    LowPassFilter(int queue_length=30);
    ~LowPassFilter(){};
    Vector6d filter_step(Vector6d input_vec);
};


class admittance_control {

    public:

        admittance_control( 
            ros::NodeHandle &n, ros::Rate ros_rate,   
            std::string topic_force_sensor_subscriber, std::string topic_joint_states_subscriber,
            std::string topic_joint_trajectory_publisher, std::string topic_action_trajectory_publisher, std::string topic_joint_group_vel_controller_publisher, 
            std::string compliance_trajectory_action_name,
            std::vector<double> mass_model_matrix, std::vector<double> damping_model_matrix, std::vector<double> spring_model_matrix, 
            double force_dead_zone, double torque_dead_zone, double admittance_weight, std::vector<double> joint_limits,
            std::vector<double> maximum_velocity, std::vector<double> maximum_acceleration);

        ~admittance_control();

        void spinner (void);

        bool simple_debug;
        bool complete_debug;
        bool trajectory_debug;

//----------------------------------------------------------------------------------------------------------------------//

        // ---- Admittance Parameters ---- //
        Matrix6d mass_matrix, damping_matrix, spring_matrix;
        double force_dead_zone, torque_dead_zone, admittance_weight;
        bool use_feedback_velocity, inertia_reduction;
        
        // ---- Admittance IO ---- //
        Vector6d external_wrench, wrench_filtered, ftsensor_start_offset, x, x_dot, q_dot;
        Vector6d x_last_cycle, q_dot_last_cycle, x_dot_last_cycle;
        Vector6d xe_i; // error integration
        Vector6d F_v;
        std::vector<double> VF_buffer; //virtue force
        Vector6d target_joint_position; 
        // ---- Limits ---- //
        Vector6d joint_lim, max_vel, max_acc;

        // ---- MoveIt Robot Model ---- //
        robot_model_loader::RobotModelLoader robot_model_loader;
        robot_model::RobotModelPtr kinematic_model;
        robot_state::RobotStatePtr kinematic_state;
        const robot_state::JointModelGroup *joint_model_group;
        std::vector<std::string> joint_names;
        std::vector<std::string> link_names;
        std::size_t link_count;
        Eigen::MatrixXd J;
        Eigen::Matrix4d EEF_ref_inv;

        planning_scene::PlanningScene* planning_scene;
        collision_detection::CollisionRequest* collision_request;
        collision_detection::CollisionResult* collision_result;
        std::string EEF_name;
        Eigen::Isometry3d T_tool0_EEF;

        // ---- Trajectory Execution ---- //
        self_defined_msgs::joint_trajectory desired_trajectory;

        // ---- Trajectory Exucition (online) ---- //
        std::vector<Vector6d> trajectory_online;
        std::vector<Vector6d> trajectory_online_dot;
        std::vector<Vector6d> trajectory_online_dotdot;
        Eigen::Isometry3d target_delta_T;  //目标轨迹点的偏移值,不计算速度及加速度
        
        // ---- Compliance Trajectory Execution ---- //
        std::vector<Vector6d> ee_world_trajectory; //拟定生成的刚性轨迹
        std::vector<Vector6d> ee_world_trajectory_dot;
        moveit_msgs::RobotTrajectory compliance_desired_joint_trajectory;
        Vector6d xd; 
        double trajectory_vel_scaling;
        double time_scaling_gain;
        double compliance_trajectory_start_time;
        double compliance_trajectory_end_time;

        // ---- mode flag ---- //
        bool Do_Compliance;
        Admittance_Mode compliance_mode;
        bool VF_flag; // do virtual force control
        bool use_position_control;

        // ---- Feedback Variables ---- //
        bool force_callback, joint_state_callback;
        bool get_compliance_trajectory_flag;
        sensor_msgs::JointState joint_state;
        std::vector<double> joint_position, joint_velocity;
        // std::vector<Vector6d> filter_elements;
        LowPassFilter filter_30, filter_200;

        // ---- Debug Variables ---- //
        std::vector<Vector6d> q_dot_real_debug;
        std::vector<Vector6d> x_real_debug;
        std::vector<Vector6d> xe_real_debug;
        std::vector<Vector6d> x_dot_real_debug;  
        std::vector<Eigen::MatrixXd> J_inv_debug;
        size_t trajectory_count;

        // ----- Trajectory online record ----- //
        size_t trajectory_count_online;
        // real
        double start_time;
        std::vector<double> time_online;
        std::vector<Vector6d> x_real_online;
        std::vector<Vector6d> x_dot_real_online;  
        std::vector<Vector6d> x_dotdot_real_online;  
        std::vector<Vector6d> xe_real_online;
        //target
        std::vector<double> time_target_online;


        // ---- FT sensor Data ---- //
        bool Ft_sensor_acq_flag;
        std::vector<Vector6d> Ft_sensor_data_series;
        size_t ft_class_id;
        size_t ft_data_id;

//----------------------------------------------------------------------------------------------------------------------//

        // ---- PUBLISHERS & SUBSCRIBERS ---- //
        ros::Subscriber force_sensor_subscriber, joint_states_subscriber;
        ros::Subscriber trajectory_online_subscriber;
        ros::Publisher joint_trajectory_publisher, joint_group_vel_controller_publisher, ur_script_command_publisher;
        ros::Publisher cartesian_position_publisher, cartesian_position_rel_publisher;
        ros::Publisher joint_group_pos_controller_publisher;
        ros::Publisher wrench_filtered_publisher, virtual_force_publisher;

        // ---- ROS SERVICE CLIENTS ---- //
        ros::ServiceClient switch_controller_client, list_controllers_client, zero_ft_sensor_client;
        controller_manager_msgs::SwitchController switch_controller_srv;
        controller_manager_msgs::ListControllers list_controllers_srv;
        std_srvs::Trigger zero_ft_sensor_srv, ur_resend_robot_program_srv, ur_play_urcap_srv;
        
        // ---- ROS SERVICE SERVERS ---- //
        ros::ServiceServer admittance_controller_activation_service, change_admittance_parameters_service;
        ros::ServiceServer set_equilibrium_pose_service, set_cur_pose_as_equilibrium_service;
        ros::ServiceServer set_VF_service;
        ros::ServiceServer virtual_force_control_activation_service;
        ros::ServiceServer follow_trajectory_online_activation_service; 
        ros::ServiceServer switch_EEF_name_to_tool0_service; 
        ros::ServiceServer set_EEF_ref_pose_service;
        ros::ServiceServer set_cur_pose_as_equilibrium_service_TOL, clear_TOL_equilibrium_offset_service; // for trajectory online
        ros::ServiceServer get_MKB_service, set_MKB_service;

        bool admittance_control_request, freedrive_mode_request, trajectory_execution_request;

        // ---- ROS ACTIONS CLIENT---- //
        actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> *trajectory_client;
        control_msgs::FollowJointTrajectoryGoal trajectory_goal;

        // ---- ROS ACTIONS SERVER ----//
        actionlib::SimpleActionServer<self_defined_msgs::compliance_actionAction> *compliance_trajectory_server;
        std::string compliance_trajectory_action_name; //r
        self_defined_msgs::compliance_actionFeedback compliance_trajectory_feedback;
        self_defined_msgs::compliance_actionResult compliance_trajectory_result; 

//----------------------------------------------------------------------------------------------------------------------//

        // ---- CALLBACKS ---- //
        void force_sensor_Callback (const geometry_msgs::WrenchStamped::ConstPtr &);
        void joint_states_Callback (const sensor_msgs::JointState::ConstPtr &);
        void trajectory_online_subscriber_Callback (const geometry_msgs::Pose::ConstPtr &);

        // ---- SERVER CALLBACKS ---- //
        bool Admittance_Controller_Activation_Service_Callback (std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res);
        bool virtual_force_control_activation_Service_Callback(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res);
        bool follow_trajectory_online_activation_service_Callback(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res);
        bool Flexible_collision_detection_service_Callback(std_srvs::SetBool::Request &req,std_srvs::SetBool::Response &res);
        bool set_VF_Service_Callback(self_defined_msgs::set_VF_srv::Request &req, self_defined_msgs::set_VF_srv::Response &res);
        bool Change_Admittance_Parameters_Service_Callback (self_defined_msgs::parameter_srv::Request &req, self_defined_msgs::parameter_srv::Response &res);
        bool Set_Equilibrium_Pose_Service_Callback(self_defined_msgs::set_equilibrium_pose_srv::Request &req, self_defined_msgs::set_equilibrium_pose_srv::Response &res);
        bool Set_Cur_Equilibrium_Pose_Service_Callback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);
        bool Set_Cur_Equilibrium_Pose_Service_TOL_Callback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);
        bool Clear_Equilibrium_TOL_Offset_Callback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);
        bool set_tool0_callback(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res);
        bool set_EEF_ref_pose_Service_Callback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);
        bool get_MKB_Service_Callback(self_defined_msgs::get_MKB::Request &req, self_defined_msgs::get_MKB::Response &res);
        bool set_MKB_Service_Callback(self_defined_msgs::set_MKB::Request &req, self_defined_msgs::set_MKB::Response &res);

        // ---- ACTIONS SERVER CALLBACKS ----//
        void compliance_trajectory_executeCB (const self_defined_msgs::compliance_actionGoalConstPtr &goal);

        // ---- KINEMATIC MODEL FUNCTIONS ---- //
        Eigen::Matrix4d compute_fk (std::vector<double> joint_position, std::vector<double> joint_velocity);
        Eigen::MatrixXd compute_arm_jacobian (std::vector<double> joint_position, std::vector<double> joint_velocity);
        Matrix6d get_ee_rotation_matrix (std::vector<double> joint_position, std::vector<double> joint_velocity);
        Eigen::Vector3d get_ee_translation_vector (std::vector<double> joint_position, std::vector<double> joint_velocity);
        Eigen::Vector3d get_ee_rotation_vector (std::vector<double> joint_position, std::vector<double> joint_velocity);
        Vector6d get_x_from_current_pose(std::vector<double> joint_position, std::vector<double> joint_velocity);
        Vector6d get_x_from_current_pose(void);
        Eigen::Isometry3d get_current_link_pose(std::string link_name);
        void get_x_from_joint_states(Vector6d& x_pose, std::vector<double> joint_position);
        void get_x_from_joint_states(Vector6d& x_pose, std::vector<double> joint_position_in, moveit::core::RobotStatePtr& new_kinematic_state);

        bool get_nearest_IK(const Isometry3d& pose, std::vector<double>& joint_IK);
        void xd_offset(Vector6d& xd); //for online trajectory
        
        // ---- ADMITTANCE FUNCTIONS ---- //
        void compute_admittance (void);
        void compute_admittance_trajectory_online (void);
        void compute_admittance_trajectory (const Vector6d& waypoint, const Vector6d& waypoint_dot);  //offline trajectory

        // ---- LIMIT DYNAMIC FUNCTIONS ---- //
        Vector6d limit_joint_dynamics (Vector6d joint_velocity);
        Vector6d compute_inertia_reduction (Vector6d velocity, Vector6d wrench);

        // ---- Compliance Trajectory FUNCTIONS (OFFLINE)----//      
        bool create_TCP_z_lmove_trajectory(double length_mm, double time_ms);
        void create_TCP_z_lmove_trajectory(Vector6d init_pose, double length_mm, double time_ms);
        void Do_TCP_z_lmove_trajectory(double length_mm, double time_ms);
        void Do_TCP_z_lmove_trajectory(Vector6d init_pose, double length_mm, double time_ms);
        bool Do_compliance_jmove_trajectory(void);
        bool create_jmove_trajectory_from_joint_trajectory(void);

        // ---- SPLINE INTERPOLATION ---- //
        std::vector<tk::spline> spline_interpolation (std::vector<Vector6d> data_vector, double spline_lenght, std::string output_file);
        std::vector<tk::spline> spline_interpolation (std::vector<Array6d> data_vector, double spline_lenght, std::string output_file);
        Vector6d get_spline_value (std::vector<tk::spline> spline6d, double s);
        
        // ------ SINGULARIT COLLISION CHECK -------- //
        bool is_self_collision();
        bool is_self_collision(std::vector<double> joint_values);
        bool is_self_singular(std::vector<double> joint_values);

        // ---- CONTROL FUNCTIONS ---- //
        void send_velocity_to_robot (Vector6d velocity);
        void send_position_to_robot (Vector6d position);
        void wait_for_position_reached (Vector6d desired_position, double maximum_time);
        bool switch_controller_to_joint_group(void);
        bool switch_controller_to_scaled_pose(void);

        bool switch_off_controller(std::string controller_name);
        bool switch_on_controller(std::string controller_name);

        // ---- UR e FUNCTIONS ---- //
        void wait_for_callbacks_initialization (void);
        void zero_ft_sensor (void);

        // ---- UTIL FUNCTIONS ---- //
        void publish_wrench(ros::Publisher& publisher, const Vector6d& wrench);
        void publish_cartesian_position(std::vector<double> joint_position, std::vector<double> joint_velocity);
        // Vector6d low_pass_filter(Vector6d input_vec);
        int sign (double num);
        Eigen::Vector3d from_rotation_matrix_to_vector(Matrix3d rotation_matrix);
        Matrix3d from_rotation_vector_to_matrix(Eigen::Vector3d rotation_vector); 
        Isometry3d from_vector6d_to_iso3d(const Vector6d& x);
        Vector6d from_iso3d_to_vector6d(const Isometry3d& T);

        void set_xd_from_current_pose(void);
        void stop_robot (void);
        inline void clear_online_trajectory();
        void get_cartesian_volocity(const Vector6d& pose_cur, const Vector6d& pose_last, Vector6d& vel, double time);
        void get_cartesian_acceleration(const Vector6d& vel_cur, const Vector6d& vel_last, Vector6d& acc, double time);

        double get_max_q_distance(const std::vector<double>& joint_1, const std::vector<double>& joint_2);


        // ---- VARIABLE CREATION FUNCTIONS ---- //
        Vector6d new_vector_6d (double x, double y, double z, double roll, double pitch, double yaw);
        
        // ---- DEBUG ---- //
        std::ofstream ft_sensor_debug;
        void csv_debug (std::vector<double> vector, std::string name);
        void csv_debug (std::vector<Vector6d> vector6d, std::string name);
        void csv_debug (std::vector<tk::spline> spline6d, std::vector<double> s, std::vector<Vector6d> data_vector, std::string name);
        void trajectory_debug_csv (std::vector<sensor_msgs::JointState> trajectory, std::string trajectory_name);
        
        void compliance_trajectory_RT_debug_csv (void);
        void compliance_trajectory_CT_debug_csv (std::vector<std::vector<double>> way_points_6_joints_interpolation);
        void ft_sensor_csv(size_t class_id, size_t data_id);

        void trajectory_online_record(void);

        // std::vector<Eigen::Isometry3d> compute_robot_transformtree(std::vector<double> joint_position, std::vector<double> joint_velocity);

    private:

        ros::NodeHandle nh;
        ros::Rate loop_rate;
        singularity_avoidance *singular_avoid;
};




#endif /* ADMITTANCE_CONTROLLER_H */
