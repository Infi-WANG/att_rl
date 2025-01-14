#include "admittance_controller/admittance_controller.h"

//----------------------------------------------------- CONSTRUCTOR -----------------------------------------------------//

admittance_control::admittance_control(
    ros::NodeHandle &n, ros::Rate ros_rate,   
    std::string topic_force_sensor_subscriber, std::string topic_joint_states_subscriber,
    std::string topic_joint_trajectory_publisher, std::string topic_action_trajectory_publisher,  std::string topic_joint_group_vel_controller_publisher,
    std::string compliance_trajectory_action_name,
    std::vector<double> mass_model_matrix, std::vector<double> damping_model_matrix, std::vector<double> spring_model_matrix, 
    double force_dead_zone, double torque_dead_zone, double admittance_weight,
    std::vector<double> joint_limits, std::vector<double> maximum_velocity, std::vector<double> maximum_acceleration):

    nh(n), loop_rate(ros_rate), mass_matrix(mass_model_matrix.data()), damping_matrix(damping_model_matrix.data()), spring_matrix(spring_model_matrix.data()),
    force_dead_zone(force_dead_zone), torque_dead_zone(torque_dead_zone), admittance_weight(admittance_weight),
    joint_lim(joint_limits.data()), max_vel(maximum_velocity.data()), max_acc(maximum_acceleration.data())
{

    singular_avoid = new singularity_avoidance(0.1,0.01);

    // ---- LOAD PARAMETERS ---- //
    if (!nh.param<bool>("/admittance_controller_Node/use_feedback_velocity", use_feedback_velocity, false)) {ROS_ERROR("Couldn't retrieve the Feedback Velocity value.");}
    if (!nh.param<bool>("/admittance_controller_Node/inertia_reduction", inertia_reduction, false)) {ROS_ERROR("Couldn't retrieve the Inertia Reduction value.");}
    // if (!nh.param<bool>("/admittance_controller_Node/use_ur_real_robot", use_ur_real_robot, false)) {ROS_ERROR("Couldn't retrieve the Use Real Robot value.");}
    if (!nh.param<bool>("/admittance_controller_Node/auto_start_admittance", admittance_control_request, true)) {ROS_ERROR("Couldn't retrieve the Auto Start Admittance value.");}
    if (!nh.param<bool>("/admittance_controller_Node/use_position_control", use_position_control, true)){ ROS_ERROR("Couldn't retrieve the use_position_control value.");}
    // ---- ROS PUBLISHERS ---- //
    joint_trajectory_publisher = nh.advertise<trajectory_msgs::JointTrajectory>(topic_joint_trajectory_publisher, 1);
    joint_group_vel_controller_publisher = nh.advertise<std_msgs::Float64MultiArray>(topic_joint_group_vel_controller_publisher, 1);
    joint_group_pos_controller_publisher = nh.advertise<std_msgs::Float64MultiArray>("/joint_group_pos_controller/command", 1);
    
    ur_script_command_publisher = nh.advertise<std_msgs::String>("/ur_hardware_interface/script_command",1);
    cartesian_position_publisher = nh.advertise<geometry_msgs::Pose>("/ur_cartesian_pose",1);
    cartesian_position_rel_publisher = nh.advertise<geometry_msgs::Pose>("/ur_cartesian_pose_rel",1);
    wrench_filtered_publisher = nh.advertise<geometry_msgs::WrenchStamped>("/wrench_filter",1);
    virtual_force_publisher = nh.advertise<geometry_msgs::WrenchStamped>("virtual_force",1);

    // ---- ROS SUBSCRIBERS ---- //
    force_sensor_subscriber = nh.subscribe(topic_force_sensor_subscriber, 1, &admittance_control::force_sensor_Callback, this);
    joint_states_subscriber = nh.subscribe(topic_joint_states_subscriber, 1, &admittance_control::joint_states_Callback, this);
    trajectory_online_subscriber = nh.subscribe("/admittance_controller/trajectory_online_target_pose", 1, &admittance_control::trajectory_online_subscriber_Callback, this);

    // ---- ROS SERVICE SERVERS ---- //
    admittance_controller_activation_service = nh.advertiseService("/admittance_controller/admittance_controller_activation_service", &admittance_control::Admittance_Controller_Activation_Service_Callback, this);//柔順控制服務請求
    virtual_force_control_activation_service = nh.advertiseService("/admittance_controller/virtual_force_control_activation_service", &admittance_control::virtual_force_control_activation_Service_Callback, this);//虛擬力控制服務請求。虛擬力初始化爲0，打開虛擬力控制標志位
    follow_trajectory_online_activation_service = nh.advertiseService("/admittance_controller/follow_trajectory_online_activation_service",&admittance_control::follow_trajectory_online_activation_service_Callback, this);//軌跡跟蹤服務請求
    set_VF_service = nh.advertiseService("/admittance_controller/set_VF_service", &admittance_control::set_VF_Service_Callback, this);//設置虛擬力控制參數
    change_admittance_parameters_service = nh.advertiseService("/admittance_controller/change_admittance_parameters_service", &admittance_control::Change_Admittance_Parameters_Service_Callback, this);//調節柔順控制參數
    set_equilibrium_pose_service = nh.advertiseService("/admittance_controller/set_equilibrium_point_service", &admittance_control::Set_Equilibrium_Pose_Service_Callback, this);//設置平衡姿態服務
    set_cur_pose_as_equilibrium_service = nh.advertiseService("/admittance_controller/set_cur_pose_as_equilibrium_service", &admittance_control::Set_Cur_Equilibrium_Pose_Service_Callback, this);//設置當前位姿作爲平衡狀態
    set_cur_pose_as_equilibrium_service_TOL =nh.advertiseService("/admittance_controller/set_cur_pose_as_equilibrium_TOL_service", &admittance_control::Set_Cur_Equilibrium_Pose_Service_TOL_Callback, this);//通过对目标点偏移設置當前位姿作爲平衡狀態,不影响输入轨迹速度和加速度 for trajectory online
    clear_TOL_equilibrium_offset_service = nh.advertiseService("/admittance_controller/clear_equilibrium_TOL_offset_service", &admittance_control::Clear_Equilibrium_TOL_Offset_Callback, this);//清除影响平衡狀態的偏移,不影响输入轨迹速度和加速度 for trajectory online
    switch_EEF_name_to_tool0_service =  nh.advertiseService("/admittance_controller/switch_EEF_to_tool0", &admittance_control::set_tool0_callback, this);
    set_EEF_ref_pose_service = nh.advertiseService("/admittance_controller/set_EEF_ref_pose_service",  &admittance_control::set_EEF_ref_pose_Service_Callback, this);
    get_MKB_service = nh.advertiseService("/admittance_controller/get_MKB_service", &admittance_control::get_MKB_Service_Callback, this);
    set_MKB_service = nh.advertiseService("/admittance_controller/set_MKB_service", &admittance_control::set_MKB_Service_Callback, this);

    // ---- ROS SERVICE CLIENTS ---- //
    switch_controller_client    = nh.serviceClient<controller_manager_msgs::SwitchController>("/controller_manager/switch_controller");
    list_controllers_client     = nh.serviceClient<controller_manager_msgs::ListControllers>("/controller_manager/list_controllers");
    zero_ft_sensor_client       = nh.serviceClient<std_srvs::Trigger>("/ur_hardware_interface/zero_ftsensor");
  
    // ---- ROS ACTIONS ---- //
    trajectory_client = new actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction>(topic_action_trajectory_publisher, true);

    compliance_trajectory_server = new actionlib::SimpleActionServer<self_defined_msgs::compliance_actionAction>(
                                                nh, compliance_trajectory_action_name, 
                                                boost::bind(&admittance_control::compliance_trajectory_executeCB, this, _1),false);

    // Initializing the Class Variables
    ee_world_trajectory.reserve(3000);
    ee_world_trajectory_dot.reserve(3000);
    // trajectory_vel_scaling = 0.65;
    if (nh.param<double> ("/admittance_controller_Node/trajectory_vel_scaling", trajectory_vel_scaling, 0.65)){
        ROS_INFO("trajctory velocity scaling factor: %f", trajectory_vel_scaling);
    }
    if (nh.param<double> ("/admittance_controller_Node/time_scaling_gain", time_scaling_gain, 0.9)){
        ROS_INFO("trajctory velocity scaling factor: %f", time_scaling_gain);
    }
    if (nh.param<double> ("/admittance_controller_Node/compliance_trajectory_start_time", compliance_trajectory_start_time, 50)){
        ROS_INFO("compliance_trajectory_start_time: %f", compliance_trajectory_start_time);
    }
    if (nh.param<double> ("/admittance_controller_Node/compliance_trajectory_end_time", compliance_trajectory_end_time, 200)){
        ROS_INFO("compliance_trajectory_end_time: %f", compliance_trajectory_end_time);
    } 

    joint_position.resize(6);
    joint_velocity.resize(6);
    external_wrench.setZero();
    wrench_filtered.setZero();
    xd.setZero();  //the desired pose, before or after compliance movement
    x.setZero();   //current pose
    x_dot.setZero();
    q_dot.setZero();
    x_last_cycle.setZero();
    x_dot_last_cycle.setZero();
    q_dot_last_cycle.setZero();
    xe_i.setZero();
    F_v.setZero();//虛擬力
    VF_buffer.resize(6);

    //trajectory online
    target_delta_T.setIdentity();
    clear_online_trajectory();

    VF_flag = true;//虛擬力控制標志位

    get_compliance_trajectory_flag = false;
    force_callback = false;
    joint_state_callback = false;
    freedrive_mode_request = false;
    trajectory_execution_request = false;

    // --- Debug Setting --- //
    simple_debug = false;
    complete_debug = false;
    trajectory_debug = false;

    // ---- MoveIt Robot Model ---- //
    robot_model_loader = robot_model_loader::RobotModelLoader("/admittance/robot_description");
    kinematic_model = robot_model_loader.getModel();
    kinematic_state = robot_state::RobotStatePtr(new robot_state::RobotState(kinematic_model));
    kinematic_state->setToDefaultValues();
    joint_model_group = kinematic_model->getJointModelGroup("manipulator");
    joint_names = joint_model_group->getJointModelNames();

    planning_scene = new planning_scene::PlanningScene(kinematic_model);
    collision_request = new collision_detection::CollisionRequest();
    collision_result = new  collision_detection::CollisionResult();

    nh.param<std::string>("/admittance_controller_Node/EEF_name", EEF_name, "TCP");
    T_tool0_EEF = get_current_link_pose("tool0") * get_current_link_pose(EEF_name).inverse();

    // ---- FT SENSOR DATA ---- //
    Ft_sensor_data_series.clear();
    ft_class_id = 0;
    ft_data_id = 0;
    Ft_sensor_acq_flag = false;
    filter_30 = LowPassFilter(30);
    filter_200 = LowPassFilter(200);

    // ---- DEBUG PRINT ---- //
    if (simple_debug) {
        std::cout << std::endl;
        ROS_INFO_STREAM_ONCE("Mass Matrix:" << std::endl << std::endl << mass_matrix << std::endl);
        ROS_INFO_STREAM_ONCE("Damping Matrix:" << std::endl << std::endl << damping_matrix << std::endl);
        ROS_INFO_ONCE("Maximum Velocity:     %.2f %.2f %.2f %.2f %.2f %.2f", max_vel[0], max_vel[1], max_vel[2], max_vel[3], max_vel[4], max_vel[5]);
        ROS_INFO_ONCE("Maximum Acceleration: %.2f %.2f %.2f %.2f %.2f %.2f \n", max_acc[0], max_acc[1], max_acc[2], max_acc[3], max_acc[4], max_acc[5]);
        ROS_INFO_ONCE("Force Dead Zone:   %.2f", force_dead_zone);
        ROS_INFO_ONCE("Troque Dead Zone:  %.2f", torque_dead_zone);
        ROS_INFO_ONCE("Admittance Weight: %.2f", admittance_weight);
        ROS_INFO_ONCE("Inertia Reduction: %s", inertia_reduction ? "true" : "false");
        ROS_INFO_STREAM_ONCE("Cycle Time: " << loop_rate.expectedCycleTime().toSec()*1000 << " ms\n");
    }
    
    if (complete_debug) {
        std::string package_path = ros::package::getPath("admittance_controller");
        ROS_INFO_STREAM_ONCE("Package Path:  " << package_path << std::endl);
        std::string save_file = package_path + "/debug/ft_sensor.csv";
        ft_sensor_debug = std::ofstream(save_file);
    }

    if (trajectory_debug){
        q_dot_real_debug.reserve(10000);
        x_real_debug.reserve(10000);
        xe_real_debug.reserve(10000);
        x_dot_real_debug.reserve(10000);
        J_inv_debug.reserve(10000);
        trajectory_count = 0;
    }
    
    // ---- trajectory online record -----/
    trajectory_count_online = 0;
    time_online.reserve(10000);
    x_real_online.reserve(10000);
    x_dot_real_online.reserve(10000);  
    x_dotdot_real_online.reserve(10000);  
    xe_real_online.reserve(10000);
    //target
    time_target_online.reserve(10000);

    // ---- clear Do_Compliance flag (default rigid trajectory) ---- //
    Do_Compliance = false;

    // ---- set state flag to fixed point mode ---- //
    compliance_mode = Fixed_Point;

    // ---- WAIT FOR INITIALIZATION ---- //
    wait_for_callbacks_initialization();

    // ---- Set Xd ---- //
    set_xd_from_current_pose();

    // ---- ZERO FT SENSOR ---- //
    zero_ft_sensor();

    // ---- start compliance trajectory server ---- //
    compliance_trajectory_server->start();

}

admittance_control::~admittance_control() {
    ft_sensor_debug.close(); 
    delete singular_avoid;
    delete collision_result;
    delete collision_request;
    delete planning_scene;
}

//--------------------------------------------------- TOPICS CALLBACK ---------------------------------------------------//


void admittance_control::force_sensor_Callback (const geometry_msgs::WrenchStamped::ConstPtr &msg) {

    geometry_msgs::WrenchStamped force_sensor = *msg;

    external_wrench[0] = force_sensor.wrench.force.x;   
    external_wrench[1] = force_sensor.wrench.force.y;   
    external_wrench[2] = force_sensor.wrench.force.z;   
    external_wrench[3] = force_sensor.wrench.torque.x;  
    external_wrench[4] = force_sensor.wrench.torque.y;  
    external_wrench[5] = force_sensor.wrench.torque.z; 
    
    // ---- DEBUG ---- //
    if (complete_debug) {
        for (int i = 0; i < 6; i++) {ft_sensor_debug << external_wrench[i] << ",";}
        ft_sensor_debug << "\n";
    }

    // ---- DEBUG ---- //
    if (complete_debug) ROS_INFO_THROTTLE_NAMED(2, "FTSensor", "Sensor Force/Torque  ->  Fx: %.2f  Fy: %.2f  Fz: %.2f  |  Tx: %.2f  Ty: %.2f  Tz: %.2f", external_wrench[0], external_wrench[1], external_wrench[2], external_wrench[3], external_wrench[4], external_wrench[5]);
    
    // ---- DEBUG ---- //
    if (complete_debug) ROS_INFO_THROTTLE_NAMED(2, "FTSensor", "Sensor Force/Torque Clamped  ->  Fx: %.2f  Fy: %.2f  Fz: %.2f  |  Tx: %.2f  Ty: %.2f  Tz: %.2f", external_wrench[0], external_wrench[1], external_wrench[2], external_wrench[3], external_wrench[4], external_wrench[5]);
    
    // LowPass Filter
    wrench_filtered = filter_200.filter_step(external_wrench);
    external_wrench = filter_30.filter_step(external_wrench);

    //當虛擬力服務已經發送請求
    if(VF_flag){
        for(size_t i=0; i<6; ++i){
            F_v(i) = VF_buffer.at(i);
        }
    }else{
        F_v.setZero();
    }
    for (size_t i = 0; i < 6; ++i)
    {
        if (isnan(external_wrench[i]) || isinf(external_wrench[i]))
            external_wrench[i] = 0.0;
        if (isnan(F_v[i]) || isinf(F_v[i]))
            F_v[i] = 0.0;
    }

    static int pub_count = 0;
    //recorder
    if(25 == pub_count){
        publish_wrench(wrench_filtered_publisher, wrench_filtered);
        publish_wrench(virtual_force_publisher, F_v);
        pub_count = 0;
    }
    pub_count++;

    // std::cout << external_wrench << std::endl;
    force_callback = true;

}

void admittance_control::joint_states_Callback(const sensor_msgs::JointState::ConstPtr &msg)
{
    static int cartesian_pub_count = 0;
  joint_state = *msg;
    
    static std::vector<std::string> valid_names 
          {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint" };

    int idx;
    for(auto i=0; i<6; i++){
        auto iter = std::find(joint_state.name.begin(), joint_state.name.end(), valid_names[i]);
        if (iter!= valid_names.end()){
            idx = std::distance(joint_state.name.begin(), iter);
            joint_position.at(i) = joint_state.position.at(idx);
            joint_velocity.at(i) = joint_state.velocity.at(idx);
        }else{
            ROS_WARN("Wrong Joint States");
            return;
        }
    }
    joint_state_callback = true;
        // Covert Joint Position in Cartesian (geomerty_msgs::Pose)
    if (25 == cartesian_pub_count){ // 10 Hz
        publish_cartesian_position(joint_position, joint_velocity);
        cartesian_pub_count = 0;
    }
    cartesian_pub_count++;
    // ROS_INFO_THROTTLE_NAMED(2, "Joint Position", "joint position: %.2f %.2f %.2f %.2f %.2f %.2f",
    //  joint_position[0], joint_position[1], joint_position[2], joint_position[3], joint_position[4], joint_position[5]);
}

void admittance_control::trajectory_online_subscriber_Callback(const geometry_msgs::Pose::ConstPtr & msg){
    
    if(!Do_Compliance || compliance_mode != Follow_Trajectory_online){
        return;
    }
    //當導納控制器服務和軌跡跟蹤服務已發送請求開始往下執行
    assert(trajectory_online.size() == trajectory_online_dot.size() && trajectory_online_dot.size() == trajectory_online_dotdot.size());

    static ros::Time time_stamp_last = ros::Time::now();

    ros::Time  time_stamp = ros::Time::now();
    ros::Duration duration = time_stamp - time_stamp_last;

    Eigen::Isometry3d target_pose, target_tool0_pose;
    tf::poseMsgToEigen(*msg, target_pose);
    target_tool0_pose = target_pose * T_tool0_EEF;

    static Eigen::Isometry3d last_target_pose = target_pose;

    static std::vector<double> target_joint_value;
    if(get_nearest_IK(target_pose, target_joint_value)){
        if(is_self_collision(target_joint_value)){
            ROS_WARN_THROTTLE(2, "Robot stop: collision warning");
            target_pose = last_target_pose;
        }
        else if(powf(target_tool0_pose.translation().x(),2) + powf(target_tool0_pose.translation().y(),2) < 0.0225  || target_tool0_pose.translation().norm() > 1.3 ){
            ROS_WARN_THROTTLE(2, "Robot stop: bad region warning");
            target_pose = last_target_pose;
        }
        else if(is_self_singular(target_joint_value)){
            ROS_WARN_THROTTLE(2, "Robot stop: singularity warning");
            target_pose = last_target_pose;
        }
    }else{
        ROS_WARN_THROTTLE(2, "Robot stop: bad IK result");
        target_pose = last_target_pose;
    }

    Vector6d pose, vel, acc;
    pose.topLeftCorner(3,1) = target_pose.translation();
    //旋轉矩陣轉歐拉角
    pose.bottomLeftCorner(3,1) = from_rotation_matrix_to_vector(target_pose.rotation());

    //calculate velocity
    if(trajectory_online_dot.size()>0){
        get_cartesian_volocity(pose, *(trajectory_online.end()-1), vel, duration.toSec());
    }else{
        vel.setZero();
    }
    
    //calculate acceleration
    if(trajectory_online_dotdot.size()>1) {
        get_cartesian_acceleration(vel, *(trajectory_online_dot.end()-1),acc, duration.toSec());
    }else{
        acc.setZero();
    }
    
    trajectory_online.push_back(pose);
    trajectory_online_dot.push_back(vel);
    trajectory_online_dotdot.push_back(acc);
    time_target_online.push_back(ros::Time::now().toSec() - start_time);

    time_stamp_last = time_stamp;
    last_target_pose = target_pose;
}

//-------------------------------------------------- SERVICES CALLBACK --------------------------------------------------//


bool admittance_control::Admittance_Controller_Activation_Service_Callback (std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res) {
    
    // Activate / Deactivate Admittance Controller
    ros::Time start, end;
    start = ros::Time::now();
    Do_Compliance = req.data;
    if (Do_Compliance)
    {
        // ROS_WARN_STREAM("Q_DOT" << q_dot);
        q_dot.setZero();
        q_dot_last_cycle.setZero();
        x_dot_last_cycle.setZero();
        send_velocity_to_robot(q_dot);

        xd = get_x_from_current_pose();
        
        if (switch_controller_to_joint_group())
        {
            // zero_ft_sensor();
            compliance_mode = Fixed_Point;
            if (use_position_control);
            {
                for (int i = 0; i < 6; ++i)
                {
                    target_joint_position[i] = joint_position[i];
                }
            }
            res.success = true;   
            end = ros::Time::now();
            ROS_INFO("Succesfully turn on the compliance mode, time cost: %f", (end - start).toSec());  
            return true;
        }
        else
        {
            res.success = false; 
            ROS_WARN("Failed to turn on the compliance mode");            
            return false;
        }
    }
    else
    {
        // stop robot
        q_dot.setZero();
        send_velocity_to_robot(q_dot);

        // switch controller
        if (switch_controller_to_scaled_pose())
        {
            res.success = true;
            compliance_mode = Fixed_Point;
            end = ros::Time::now();
            ROS_INFO("Succesfully turn off the compliance mode, time cost: %f", (end - start).toSec());     
            return true;
        }
        else
        {
            res.success = false;   
            ROS_WARN("Failed to turn off the compliance mode");            
            return false;            
        }
    }
}

bool admittance_control::virtual_force_control_activation_Service_Callback(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res){
    VF_flag = req.data;
    VF_buffer = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    res.success = true;
    return true;
}

bool admittance_control::follow_trajectory_online_activation_service_Callback(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res){
    
    std::string msg_str;

    if(!Do_Compliance){
        msg_str = "Admittance controller is not on Do_compliance";
        ROS_WARN("msgs:%s",msg_str.c_str());
        res.success = false;
        res.message = msg_str;
        return true;
    }
    stop_robot();
    if (use_position_control);
    {
        for (int i = 0; i < 6; ++i)
        {
            target_joint_position[i] = joint_position[i];
        }
    }

    if(req.data){
        start_time = ros::Time::now().toSec();
        clear_online_trajectory(); //READY TO RECORD
        msg_str = "Successfully turn on the trajectory (online) mode";
        ROS_INFO("msgs:%s",msg_str.c_str());
        compliance_mode = Follow_Trajectory_online;
        res.success = true;
        res.message = msg_str;
        return true;        
    }else{
        compliance_mode = Fixed_Point;
        msg_str = "Successfully turn off the trajectory (online) mode, switched to the fixed point mode(compliance)";
        ROS_INFO("msgs: %s", msg_str.c_str());
        res.success = true;
        res.message = msg_str;
        trajectory_online_record();
        return true;     
    }
}

bool admittance_control::set_VF_Service_Callback(self_defined_msgs::set_VF_srv::Request &req, self_defined_msgs::set_VF_srv::Response &res){
    if(6 != req.virtual_force.size()){
        VF_buffer = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        res.success = false;
        return false;
    }else{
        VF_buffer = req.virtual_force;
        res.success = true;
        return true;
    }

}

bool admittance_control::Change_Admittance_Parameters_Service_Callback (self_defined_msgs::parameter_srv::Request &req, self_defined_msgs::parameter_srv::Response &res) {

    std::string parameter_name = req.parameter_name;
    double parameter_value = req.parameter_value;

    if (parameter_name == "mass") {

        mass_matrix = Array6d(parameter_value).matrix().asDiagonal();
        ROS_INFO_STREAM("Mass Matrix:" << std::endl << std::endl << mass_matrix << std::endl);

    } else if (parameter_name == "damping") {

        damping_matrix = Array6d(parameter_value).matrix().asDiagonal();
        ROS_INFO_STREAM("Damping Matrix:" << std::endl << std::endl << damping_matrix << std::endl);

    }

    res.success = true;
    return true;
}

bool admittance_control::Set_Equilibrium_Pose_Service_Callback(self_defined_msgs::set_equilibrium_pose_srv::Request &req, self_defined_msgs::set_equilibrium_pose_srv::Response &res){
    if (req.equilibrium_pose.size() != 6){
        ROS_WARN("Set wrong pose, which should be in form of (j1,j2,j3,j4,j5,j6)");
        res.success = false;
        return false;
    }else{
        get_x_from_joint_states(xd, req.equilibrium_pose);//將關節空間轉爲笛卡爾位姿數據 平衡的目標位姿 xd末端執行器目標位姿
        ROS_INFO_STREAM("Successfully set the equilibrium pose at:" << xd.transpose() << "\r\n");
        res.success = true;
        return true;
    }
}

bool admittance_control::Set_Cur_Equilibrium_Pose_Service_Callback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res){
    xd = get_x_from_current_pose();//訂閱當前位姿信息作爲平衡目標位姿
    ROS_INFO_STREAM("Successfully set the equilibrium pose at:" << xd.transpose() << "\r\n");    
    res.success = true;
    return true;
}

bool admittance_control::Set_Cur_Equilibrium_Pose_Service_TOL_Callback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res){
    if(trajectory_online.size() == 0){
        ROS_WARN("No trajectory input yet");
        res.success = false;
        return true;
    }
    Isometry3d xd = from_vector6d_to_iso3d(*(trajectory_online.end()-1));
    Isometry3d cur = from_vector6d_to_iso3d(get_x_from_current_pose());
    // xd * t = cur -> t = xd.inverse * cur
    target_delta_T = xd.inverse() * cur;
    // ROS_INFO_STREAM("T:" << target_delta_T.translation());
    res.success = true;
    return true;
}

bool admittance_control::Clear_Equilibrium_TOL_Offset_Callback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res){
    target_delta_T.setIdentity();
    res.success = true;
    return true;    
}


bool admittance_control::set_tool0_callback (std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res){
    static std::string EEF_name_BK = EEF_name;
    if(req.data){
        EEF_name = "tool0";
    }else{
        EEF_name = EEF_name_BK;
    }
    ROS_INFO("current EEF name: %s ", EEF_name.c_str());
    res.message= "current EEF name: " + EEF_name;
    res.success = true;
    return true;
}


//---------------------------------------------- ACTION CALLBACK FUNCTIONS ----------------------------------------------//

void admittance_control::compliance_trajectory_executeCB (const self_defined_msgs::compliance_actionGoalConstPtr &goal){

    compliance_mode = Follow_Trajectory_offline;
    bool success = true;
    x_dot.setZero();
    q_dot.setZero();
    x_last_cycle.setZero();
    x_dot_last_cycle.setZero();
    q_dot_last_cycle.setZero();

    compliance_trajectory_feedback.progress_rate = 0.0;

    // publish info to the console for the user
    ROS_INFO("Compliance_trajectory execution!");

    if(goal->command == "compliance_lmove"){

//-------------------------------------  compliance lmove begin--------------------------------------------------//
        ros::spinOnce();
        double length_mm = goal->lmove_length_mm;
        double time_ms = goal->lmove_time_ms;
        if (create_TCP_z_lmove_trajectory(length_mm, time_ms)){
             xd = ee_world_trajectory.back();

            // ---- Control loop ---- // 
            for(long unsigned int i=0; i<ee_world_trajectory.size(); ++i){
                if (compliance_trajectory_server->isPreemptRequested() || !ros::ok())
                {
                    ROS_INFO("Preempted");
                    // set the action state to preempted
                    compliance_trajectory_server->setPreempted();
                    success = false;
                    xd = get_x_from_current_pose(joint_position, joint_velocity);
                    break;
                }

                ros::spinOnce();
                compute_admittance_trajectory(ee_world_trajectory.at(i), ee_world_trajectory_dot.at(i));
                send_velocity_to_robot(q_dot);
                
                //feedback 
                compliance_trajectory_feedback.progress_rate = (double)(i+1)/(double)ee_world_trajectory.size();
                compliance_trajectory_server->publishFeedback(compliance_trajectory_feedback);

                loop_rate.sleep();
            }

            ee_world_trajectory.clear();
            ee_world_trajectory_dot.clear();
            
        }else{success = false;}

        if(success) {
            compliance_trajectory_result.execution_result = true;
            compliance_trajectory_result.result_info = "Compliance lmove trajectory executed";
            ROS_INFO("lmove Succeeded");
            // set the action state to succeeded
            compliance_trajectory_server->setSucceeded(compliance_trajectory_result);
        }else{
            compliance_trajectory_result.execution_result = false;
            compliance_trajectory_result.result_info = "error";
            compliance_trajectory_server->setAborted(compliance_trajectory_result);
        }
    }
//-------------------------------------  compliance lmove end--------------------------------------------------//


    else if(goal->command == "compliance_jmove"){
//-------------------------------------  compliance jmove begin------------------------------------------------//
        ros::spinOnce();
        compliance_desired_joint_trajectory = goal->jmove_robot_trajectory;
        Ft_sensor_acq_flag = goal->record_ft_data;
        xd = ee_world_trajectory.back();
        if (create_jmove_trajectory_from_joint_trajectory()){
            xe_i.setZero();
            if(trajectory_debug){
                // --- clear debug variables --- //

                q_dot_real_debug.clear();
                x_real_debug.clear();
                xe_real_debug.clear();
                x_dot_real_debug.clear();
                J_inv_debug.clear();       
            }
            if(Ft_sensor_acq_flag){
                Ft_sensor_data_series.clear();
            }
            for(long unsigned int i=0; i<ee_world_trajectory.size(); ++i){

                if (compliance_trajectory_server->isPreemptRequested() || !ros::ok()){
                    ROS_INFO("Preempted");
                    // set the action state to preempted
                    compliance_trajectory_server->setPreempted();
                    success = false;
                    xd = get_x_from_current_pose(joint_position, joint_velocity);
                    break;
                }

                ros::spinOnce();
                compute_admittance_trajectory(ee_world_trajectory.at(i), ee_world_trajectory_dot.at(i));
                send_velocity_to_robot(q_dot);

                if(Ft_sensor_acq_flag){
                    Ft_sensor_data_series.push_back(wrench_filtered);
                }

                //feedback 
                compliance_trajectory_feedback.progress_rate = (double)(i+1)/(double)ee_world_trajectory.size();
                compliance_trajectory_server->publishFeedback(compliance_trajectory_feedback);
                
                loop_rate.sleep();
            }

            ee_world_trajectory.clear();
            ee_world_trajectory_dot.clear();
            
            if(trajectory_debug){
                // --- save debug variables --- //
                compliance_trajectory_RT_debug_csv();
                // --- debug file name count ++ ---//
                trajectory_count++;
            }
            if(Ft_sensor_acq_flag){
                ft_sensor_csv(ft_class_id, ft_data_id);
                ft_data_id ++;
            }

        } else {success = false;}

        if(success) {
            compliance_trajectory_result.execution_result = true;
            compliance_trajectory_result.result_info = "Compliance jmove trajectory executed";
            ROS_INFO("jmove Succeeded");
            // set the action state to succeeded
            compliance_trajectory_server->setSucceeded(compliance_trajectory_result);
        }else{
            compliance_trajectory_result.execution_result = false;
            compliance_trajectory_result.result_info = "error";
            compliance_trajectory_server->setAborted(compliance_trajectory_result);
        }
        
//-------------------------------------  compliance jmove end------------------------------------------------//
    }
    else {
        ROS_WARN("Wrong \"command\" input! \"command\" can only be \"compliance_lmove\" or \"compliance_jmove\". Execution aborted!");
        compliance_trajectory_result.execution_result = false;
        compliance_trajectory_result.result_info = "error";
        compliance_trajectory_server->setAborted(compliance_trajectory_result);
    }
    compliance_mode = Fixed_Point;
    xe_i.setZero();
}

//------------------------------------------------- KINEMATIC FUNCTIONS -------------------------------------------------//


Eigen::Matrix4d admittance_control::compute_fk (std::vector<double> joint_position, std::vector<double> joint_velocity) {

    ros::spinOnce();

    //Update MoveIt! Kinematic Model
    kinematic_state->setJointGroupPositions(joint_model_group, joint_position);
    kinematic_state->setJointGroupVelocities(joint_model_group, joint_velocity);
    kinematic_state->enforceBounds();

    // Computing the actual position of the end-effector using Forward Kinematic respect "world"
    const Eigen::Affine3d& end_effector_state = kinematic_state->getGlobalLinkTransform(EEF_name);

    // Get the Translation Vector and Rotation Matrix
    Eigen::Vector3d translation_vector = end_effector_state.translation();
    Eigen::Matrix3d rotation_matrix = end_effector_state.rotation();

    //Transformation Matrix
    Eigen::Matrix4d transformation_matrix;
    transformation_matrix.setZero();

    //Set Identity to make bottom row of Matrix 0,0,0,1
    transformation_matrix.setIdentity();

    transformation_matrix.block<3,3>(0,0) = rotation_matrix;
    transformation_matrix.block<3,1>(0,3) = translation_vector;

    return transformation_matrix;

}

Eigen::MatrixXd admittance_control::compute_arm_jacobian (std::vector<double> joint_position, std::vector<double> joint_velocity) {

    ros::spinOnce();

    //Update MoveIt! Kinematic Model
    kinematic_state->setJointGroupPositions(joint_model_group, joint_position);
    kinematic_state->setJointGroupVelocities(joint_model_group, joint_velocity);
    kinematic_state->enforceBounds();

    // Computing the Jacobian of the arm
    Eigen::Vector3d reference_point_position(0.0,0.0,0.0);
    Eigen::MatrixXd jacobian;

    kinematic_state->getJacobian(joint_model_group, kinematic_state->getLinkModel(EEF_name), reference_point_position, jacobian);

    // ---- DEBUG ---- //
    if (complete_debug) ROS_INFO_STREAM_THROTTLE_NAMED(2, "Manipulator Jacobian", "Manipulator Jacobian: " << std::endl << std::endl << jacobian << std::endl);
    if (complete_debug) ROS_INFO_STREAM_THROTTLE_NAMED(2, "Manipulator Inverse Jacobian", "Manipulator Inverse Jacobian: " << std::endl << std::endl << jacobian.inverse() << std::endl);

    return jacobian;

}

Eigen::Vector3d admittance_control::get_ee_translation_vector (std::vector<double> joint_position, std::vector<double> joint_velocity){

    ros::spinOnce();

    //Update MoveIt! Kinematic Model
    kinematic_state->setJointGroupPositions(joint_model_group, joint_position);
    kinematic_state->setJointGroupVelocities(joint_model_group, joint_velocity);
    kinematic_state->enforceBounds();

    // Computing the actual position of the end-effector using Forward Kinematic respect "world"
    const Eigen::Affine3d& end_effector_state = kinematic_state->getGlobalLinkTransform(EEF_name);

    // translation vector 3x1
    Eigen::Vector3d translation_vector;
    translation_vector.setZero();

    translation_vector = end_effector_state.translation();

    // ---- DEBUG ---- //
    if (complete_debug) {
        ROS_INFO_THROTTLE_NAMED(2, "Translation Vector",  "Translation Vector   ->   X: %.3f  Y: %.3f  Z: %.3f", end_effector_state.translation().x(), end_effector_state.translation().y(), end_effector_state.translation().z());
    }

    return translation_vector;
}

Matrix6d admittance_control::get_ee_rotation_matrix (std::vector<double> joint_position, std::vector<double> joint_velocity) {

    ros::spinOnce();

    //Update MoveIt! Kinematic Model
    kinematic_state->setJointGroupPositions(joint_model_group, joint_position);
    kinematic_state->setJointGroupVelocities(joint_model_group, joint_velocity);
    kinematic_state->enforceBounds();

    // Computing the actual position of the end-effector using Forward Kinematic respect "world"
    const Eigen::Affine3d& end_effector_state = kinematic_state->getGlobalLinkTransform(EEF_name);

    // Rotation Matrix 6x6
    Matrix6d rotation_matrix;
    rotation_matrix.setZero();

    rotation_matrix.topLeftCorner(3, 3) = end_effector_state.rotation();
    rotation_matrix.bottomRightCorner(3, 3) = end_effector_state.rotation();

    Eigen::Vector3d euler_angles = end_effector_state.rotation().eulerAngles(0, 1, 2);
    Eigen::Quaterniond rotation_quaternion(end_effector_state.rotation());

    // ---- DEBUG ---- //
    if (complete_debug) {

        ROS_INFO_THROTTLE_NAMED(2, "Translation Vector",  "Translation Vector   ->   X: %.3f  Y: %.3f  Z: %.3f", end_effector_state.translation().x(), end_effector_state.translation().y(), end_effector_state.translation().z());
        ROS_INFO_THROTTLE_NAMED(2, "Euler Angles",        "Euler Angles         ->   R: %.3f  P: %.3f  Y: %.3f", euler_angles[0], euler_angles[1], euler_angles[2]);
        ROS_INFO_THROTTLE_NAMED(2, "Rotation Quaternion", "Rotation Quaternion  ->   X: %.3f  Y: %.3f  Z: %.3f  W: %.3f", rotation_quaternion.x(), rotation_quaternion.y(), rotation_quaternion.z(), rotation_quaternion.w());

        ROS_INFO_STREAM_THROTTLE_NAMED(2, "Rotation Matrix from Model", "Rotation Matrix from Model:" << std::endl << std::endl << end_effector_state.rotation() << std::endl);
        ROS_INFO_STREAM_THROTTLE_NAMED(2, "Rotation Matrix 6x6",        "Rotation Matrix 6x6:" << std::endl << std::endl << rotation_matrix << std::endl);
    }

    return rotation_matrix;
}

Eigen::Vector3d admittance_control::get_ee_rotation_vector (std::vector<double> joint_position, std::vector<double> joint_velocity) {

    ros::spinOnce();

    //Update MoveIt! Kinematic Model
    kinematic_state->setJointGroupPositions(joint_model_group, joint_position);
    kinematic_state->setJointGroupVelocities(joint_model_group, joint_velocity);
    kinematic_state->enforceBounds();

    // Computing the actual position of the end-effector using Forward Kinematic respect "world"
    const Eigen::Affine3d& end_effector_state = kinematic_state->getGlobalLinkTransform(EEF_name);

    Eigen::AngleAxisd rotation_vector;
    rotation_vector.fromRotationMatrix(end_effector_state.rotation());

    return rotation_vector.angle()*rotation_vector.axis();
}

Vector6d admittance_control::get_x_from_current_pose(std::vector<double> joint_position, std::vector<double> joint_velocity) {
    
    ros::spinOnce();
    Vector6d x_cur;
    x_cur.setZero();

    x_cur.topLeftCorner(3, 1) = get_ee_translation_vector(joint_position, joint_velocity);
    x_cur.bottomLeftCorner(3, 1) = get_ee_rotation_vector(joint_position, joint_velocity);  //此处可以再优化

    // ROS_INFO_STREAM_THROTTLE(2, "x_cur:" << xd.transpose() << "\r\n");

    return  x_cur;
}

Eigen::Isometry3d admittance_control::get_current_link_pose(std::string link_name){
    moveit::core::RobotStatePtr new_kinematic_state = robot_state::RobotStatePtr(new robot_state::RobotState(kinematic_model));

    new_kinematic_state->setJointGroupPositions(joint_model_group, joint_position);
    std::vector<double> zere_velocity{0,0,0,0,0,0};
    new_kinematic_state->setJointGroupVelocities(joint_model_group, zere_velocity);
    new_kinematic_state->enforceBounds();

    // Computing the actual position of the end-effector using Forward Kinematic respect "world"
    return new_kinematic_state->getGlobalLinkTransform(link_name);
}


Vector6d admittance_control::get_x_from_current_pose(void) {
    
    ros::spinOnce();
    Vector6d x_cur;
    x_cur.setZero();

    x_cur.topLeftCorner(3, 1) = get_ee_translation_vector(joint_position, joint_velocity);
    x_cur.bottomLeftCorner(3, 1) = get_ee_rotation_vector(joint_position, joint_velocity);  //此处可以再优化

    // ROS_INFO_STREAM_THROTTLE(2, "x_cur:" << xd.transpose() << "\r\n");

    return  x_cur;
}

/**
 * @brief get the pose by the specific joints state
 *        the pose is in the form of (x,y,z,rx,ry,rz), the rotation is, namely, axis_angle form
 * 
 * @param x_pose 
 * @param joint_position_in 
 */
void  admittance_control::get_x_from_joint_states(Vector6d& x_pose, std::vector<double> joint_position_in){

    // ros::spinOnce();
    moveit::core::RobotStatePtr new_kinematic_state = robot_state::RobotStatePtr(new robot_state::RobotState(kinematic_model));

    new_kinematic_state->setJointGroupPositions(joint_model_group, joint_position_in);
    std::vector<double> zere_velocity{0,0,0,0,0,0};
    new_kinematic_state->setJointGroupVelocities(joint_model_group, zere_velocity);
    new_kinematic_state->enforceBounds();

    // Computing the actual position of the end-effector using Forward Kinematic respect "world"
    const Eigen::Affine3d& end_effector_state = new_kinematic_state->getGlobalLinkTransform(EEF_name);
    x_pose.topLeftCorner(3,1) = end_effector_state.translation();

    Eigen::AngleAxisd rotation_vector;
    rotation_vector.fromRotationMatrix(end_effector_state.rotation());

    x_pose.bottomLeftCorner(3, 1) = rotation_vector.angle()*rotation_vector.axis();

    // ROS_INFO_STREAM_THROTTLE(2, "pose:" << x_pose.transpose() << "\r\n");
}

void  admittance_control::get_x_from_joint_states(Vector6d& x_pose, std::vector<double> joint_position_in, moveit::core::RobotStatePtr& new_kinematic_state){

    new_kinematic_state->setJointGroupPositions(joint_model_group, joint_position_in);
    std::vector<double> zere_velocity{0,0,0,0,0,0};
    new_kinematic_state->setJointGroupVelocities(joint_model_group, zere_velocity);
    new_kinematic_state->enforceBounds();

    // Computing the actual position of the end-effector using Forward Kinematic respect "world"
    const Eigen::Affine3d& end_effector_state = new_kinematic_state->getGlobalLinkTransform(EEF_name);
    x_pose.topLeftCorner(3,1) = end_effector_state.translation();

    Eigen::AngleAxisd rotation_vector;
    rotation_vector.fromRotationMatrix(end_effector_state.rotation());

    x_pose.bottomLeftCorner(3, 1) = rotation_vector.angle()*rotation_vector.axis();

    // ROS_INFO_STREAM_THROTTLE(2, "pose:" << x_pose.transpose() << "\r\n");
}

/**
 * @brief get the IK result near to current joint values
 * 
 * @param pose 
 * @param joint_IK 
 * @return true 
 * @return false 
 */
bool admittance_control::get_nearest_IK(const Isometry3d& pose, std::vector<double>& joint_IK){
    double timeout = 0.1;

    bool found_ik;
    
    for(auto i=1; i<10; i++){
        // ros::spinOnce();
        found_ik = kinematic_state->setFromIK(joint_model_group, pose, EEF_name, timeout); //todo 基于末端到及坐标,需要验证
        if(found_ik){
            kinematic_state->copyJointGroupPositions(joint_model_group, joint_IK);
            if (0.53 > get_max_q_distance(joint_position, joint_IK)) return true;  // pi/6
        }
    }
    ROS_WARN_THROTTLE(2, "Warning: failed to find IK solution");
    return false;
}

/**
 * @brief generate a new xd by transformation target_delta_T
 * 
 * @param xd 
 */
void admittance_control::xd_offset(Vector6d& xd){ 
    // std::cout << "before: " << xd;
    xd = from_iso3d_to_vector6d((from_vector6d_to_iso3d(xd) * target_delta_T));
    // std::cout << "after: " << xd;
}

//------------------------------------------------- ADMITTANCE FUNCTION -------------------------------------------------//


void admittance_control::compute_admittance (void) {

    ros::spinOnce();
    // Compute Manipulator Jacobian 計算雅可比矩陣
    J = compute_arm_jacobian(joint_position, joint_velocity);
    //對雅可比矩陣進行奇異值分解 阻尼最小二乘法


    // Compute current x , which is in the from of (x,y,z,rx,ry,rz)
    //當前時刻的位姿數據
    x = get_x_from_current_pose(joint_position, joint_velocity);

    // Compute xe , Re = R * Rd^T
    Eigen::Vector3d temp_rotation_vector(x(3),x(4),x(5));
    Matrix3d x_rotation_matrix = from_rotation_vector_to_matrix(temp_rotation_vector); 
    
    temp_rotation_vector.setZero();
    temp_rotation_vector << xd(3),xd(4),xd(5);    
    Matrix3d xd_rotation_matrix = from_rotation_vector_to_matrix(temp_rotation_vector);

    Matrix3d xe_ratation_matrix = x_rotation_matrix * xd_rotation_matrix.transpose();
    Eigen::Vector3d xe_rotation_vector = from_rotation_matrix_to_vector(xe_ratation_matrix);

    Vector6d xe = x - xd;
    xe.bottomLeftCorner(3,1) = xe_rotation_vector;

    // if (use_feedback_velocity) {
        
    //     Vector6d joint_velocity_eigen = Eigen::Map<Vector6d>(joint_velocity.data());

    //     // Compute Cartesian Velocity
    //     x_dot = J * joint_velocity_eigen;
        
    //     // ---- DEBUG ---- //
    //     if (complete_debug) {ROS_INFO_STREAM_ONCE_NAMED("Start Velocity", "Start Velocity: " << std::endl << std::endl << x_dot << std::endl);}
    
    // } else {
        
        // Use the Cartesian Speed obtained the last cycle
        // Vector6d joint_velocity_eigen = Eigen::Map<Vector6d>(joint_velocity.data());
        x_dot = x_dot_last_cycle;
        
    // }

    // Subtract FTSensor Starting Offset
    for (unsigned i = 0; i < external_wrench.size(); i++) {external_wrench[i] += - ftsensor_start_offset[i];}

    // Compute Acceleration with Admittance
    Matrix6d eef_rotation = get_ee_rotation_matrix(joint_position, joint_velocity);
    Vector6d external_wrench_world = eef_rotation * external_wrench;
    Vector6d F_v_world = eef_rotation * F_v;
    // external_wrench_world[0] = 0.0;
    // external_wrench_world[1] = 0.0;
    // external_wrench_world[2] = 0.0;
    // external_wrench_world[3] = 0.0;
    // external_wrench_world[4] = 0.0;
    // external_wrench_world[5] = 0.0;
    
    Vector6d arm_desired_accelaration_cartesian = mass_matrix.inverse() * ( - damping_matrix * x_dot  
                                                - spring_matrix * xe
                                                + admittance_weight * external_wrench_world
                                                + F_v_world   // virtual force
                                                  );

    // Vector6d arm_desired_accelaration_cartesian = mass_matrix.inverse() * ( - damping_matrix * x_dot  
    //                                             + admittance_weight * 
    //                                               (get_ee_rotation_matrix(joint_position, joint_velocity) * external_wrench));


    // Integrate for Velocity Based Interface
    ros::Duration duration = loop_rate.expectedCycleTime();
    // ROS_INFO_STREAM_ONCE("Cycle Time: " << duration.toSec()*1000 << " ms");
    x_dot  += arm_desired_accelaration_cartesian * duration.toSec();

    // Inertia Reduction Function
    // if (inertia_reduction) {x_dot = compute_inertia_reduction(x_dot, external_wrench);}
    
    // Inverse Kinematic for Joint Velocity
    q_dot = J.inverse() * x_dot;

    // Limit System Dynamic
    q_dot = limit_joint_dynamics(q_dot);
    x_dot_last_cycle = J * q_dot;
    
    if (use_position_control)
    {
        for (int i = 0; i < 6; ++i)
        {
            // target_joint_position[i] = joint_position[i] + q_dot[i] * duration.toSec();
            target_joint_position[i] += q_dot[i] * duration.toSec();
        }
    }

    // ---- DEBUG ---- //
    if (complete_debug) {ROS_INFO_THROTTLE_NAMED(2, "Desired Cartesian Velocity", "Desired Cartesian Velocity:  %.2f  %.2f  %.2f  %.2f  %.2f  %.2f", x_dot[0], x_dot[1], x_dot[2], x_dot[3], x_dot[4], x_dot[5]);}
    if (complete_debug) {ROS_INFO_THROTTLE_NAMED(2, "Desired Joints Velocity",    "Desired Joints Velocity:     %.2f  %.2f  %.2f  %.2f  %.2f  %.2f", q_dot[0], q_dot[1], q_dot[2], q_dot[3], q_dot[4], q_dot[5]);}

}

// TODO DEBUG 軌跡跟蹤
void admittance_control::compute_admittance_trajectory_online (void){

    static auto time_last = std::chrono::high_resolution_clock::now();

    ros::spinOnce();
    if(trajectory_online.size()==0){ return; }
    
    // Eigen::MatrixXd geoJacob_k_inv;


    // Compute Manipulator Jacobian
    J = compute_arm_jacobian(joint_position, joint_velocity);
    // Calculate the minimum singular value
    double singuar_value = singular_avoid->singuar_value_decomposition(J);
    //Calculate the determinant value
    double det_Jacobian_value = J.determinant();
    //Calculate Jacobian matrix transpose


    Eigen::MatrixXd Jacobian_transpose = J.transpose();
    // Compute current x , which is in the from of (x,y,z,rx,ry,rz)
    x = get_x_from_current_pose(joint_position, joint_velocity);

    xd = *(trajectory_online.end()-1);
    xd_offset(xd); //xd * target_delta_T
    Vector6d xd_dot = *(trajectory_online_dot.end()-1);
    Vector6d xd_dotdot = *(trajectory_online_dotdot.end()-1);
    
    Vector6d xe, xe_dot;
    get_cartesian_volocity(x, xd, xe, 1.0); // xe = x - xd; let time eqaul to 1.0, namely, minus function
    
    // Compute xe_dot
    x_dot = x_dot_last_cycle;
    xe_dot = x_dot - xd_dot;

    // Subtract FTSensor Starting Offset
    for (unsigned i = 0; i < external_wrench.size(); i++) {external_wrench[i] += - ftsensor_start_offset[i];}

    // Compute Acceleration with Admittance

    Matrix6d eef_rotation = get_ee_rotation_matrix(joint_position, joint_velocity);
    Vector6d external_wrench_world = eef_rotation * external_wrench;
    Vector6d F_v_world = eef_rotation * F_v;

    Vector6d arm_desired_accelaration_cartesian = xd_dotdot + mass_matrix.inverse() * ( - damping_matrix * xe_dot  
                                                - spring_matrix * xe
                                                + admittance_weight * external_wrench_world
                                                + F_v_world
                                                  );
    // Vector6d arm_desired_accelaration_cartesian = mass_matrix.inverse() * ( - damping_matrix * xe_dot  
    //                                             - spring_matrix * xe
    //                                             + admittance_weight * external_wrench_world
    //                                             + F_v_world
    //                                               );

    // Integrate for Velocity Based Interface
    // ros::Duration duration = loop_rate.expectedCycleTime();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (std::chrono::high_resolution_clock::now() - time_last).count()/1.0e9;
    time_last = std::chrono::high_resolution_clock::now();

    // ROS_INFO("ros duration:%lf", duration.toSec());
    // ROS_INFO("duration:%lf", duration2);
    // ROS_INFO_STREAM_ONCE("Cycle Time: " << duration.toSec()*1000 << " ms");
    x_dot  += arm_desired_accelaration_cartesian * duration;

    // Inertia Reduction Function
    if (inertia_reduction) {x_dot = compute_inertia_reduction(x_dot, external_wrench);}


    q_dot = J.inverse() * x_dot;

    // Limit System Dynamic
    q_dot = limit_joint_dynamics(q_dot);
    x_dot_last_cycle = J * q_dot;

    if (use_position_control)
    {
        for (int i = 0; i < 6; ++i)
        {
            // target_joint_position[i] = joint_position[i] + q_dot[i] * duration;
            target_joint_position[i] += q_dot[i] * duration;
        }
    }

    // ---- DEBUG ---- //
    if (complete_debug) {ROS_INFO_THROTTLE_NAMED(2, "Desired Cartesian Velocity", "Desired Cartesian Velocity:  %.2f  %.2f  %.2f  %.2f  %.2f  %.2f", x_dot[0], x_dot[1], x_dot[2], x_dot[3], x_dot[4], x_dot[5]);}
    if (complete_debug) {ROS_INFO_THROTTLE_NAMED(2, "Desired Joints Velocity",    "Desired Joints Velocity:     %.2f  %.2f  %.2f  %.2f  %.2f  %.2f", q_dot[0], q_dot[1], q_dot[2], q_dot[3], q_dot[4], q_dot[5]);}

    // ROS_INFO_THROTTLE_NAMED(1, "joint_position", "%.2f  %.2f  %.2f  %.2f  %.2f  %.2f",joint_position[0],joint_position[1],joint_position[2],joint_position[3],joint_position[4],joint_position[5]);
    
    // record
    static size_t record_count = 0;
    if(record_count == 0){
        x_real_online.push_back(x);
        x_dot_real_online.push_back(x_dot);
        x_dotdot_real_online.push_back(arm_desired_accelaration_cartesian);
        xe_real_online.push_back(xe);
        time_online.push_back(ros::Time::now().toSec() - start_time);
    }
    if(record_count == 49){
        record_count = 0;
    }else{
        record_count++;
    }

}

void admittance_control::compute_admittance_trajectory (const Vector6d& waypoint, const Vector6d& waypoint_dot) {

    ros::spinOnce();

    // Compute Manipulator Jacobian
    J = compute_arm_jacobian(joint_position, joint_velocity);

    // Compute current x , which is in the from of (x,y,z,rx,ry,rz)
    x = get_x_from_current_pose(joint_position, joint_velocity);

    xd = waypoint;
    // Compute xe , Re = R * Rd^T
    Eigen::Vector3d temp_rotation_vector(x(3),x(4),x(5));
    Matrix3d x_rotation_matrix = from_rotation_vector_to_matrix(temp_rotation_vector); 
    
    temp_rotation_vector.setZero();
    temp_rotation_vector << xd(3),xd(4),xd(5);    
    Matrix3d xd_rotation_matrix = from_rotation_vector_to_matrix(temp_rotation_vector);

    Matrix3d xe_ratation_matrix = x_rotation_matrix * xd_rotation_matrix.transpose();
    Eigen::Vector3d xe_rotation_vector = from_rotation_matrix_to_vector(xe_ratation_matrix);

    Vector6d xe = x - xd;
    xe.bottomLeftCorner(3,1) = xe_rotation_vector;
    xe_i += xe;
    xe_i.bottomLeftCorner(3,1) = Eigen::Vector3d(0.0,0.0,0.0);
    // ROS_INFO_STREAM("XE_I: " << xe_i.transpose());

    // Compute x_dot
    x_dot = x_dot_last_cycle;

    Vector6d xe_dot = x_dot - waypoint_dot * trajectory_vel_scaling;

    // Subtract FTSensor Starting Offset
    for (unsigned i = 0; i < external_wrench.size(); i++) {external_wrench[i] += - ftsensor_start_offset[i];}

    // Compute Acceleration with Admittance
    Vector6d external_wrench_world = get_ee_rotation_matrix(joint_position, joint_velocity) * external_wrench;
    // external_wrench_world[0] = 0.0;
    // external_wrench_world[1] = 0.0;
    // external_wrench_world[2] = 0.0;
    // external_wrench_world[3] = 0.0;
    // external_wrench_world[4] = 0.0;
    // external_wrench_world[5] = 0.0;
    
    Vector6d arm_desired_accelaration_cartesian = mass_matrix.inverse() * ( - damping_matrix * xe_dot  
                                                - spring_matrix * xe
                                                + admittance_weight * external_wrench_world
                                                  );

    // Vector6d arm_desired_accelaration_cartesian = mass_matrix.inverse() * ( - damping_matrix * x_dot  
    //                                             + admittance_weight * 
    //                                               (get_ee_rotation_matrix(joint_position, joint_velocity) * external_wrench));


    // Integrate for Velocity Based Interface
    ros::Duration duration = loop_rate.expectedCycleTime();
    // ROS_INFO_STREAM_ONCE("Cycle Time: " << duration.toSec()*1000 << " ms");
    x_dot  += arm_desired_accelaration_cartesian * duration.toSec();

    // Inertia Reduction Function
    if (inertia_reduction) {x_dot = compute_inertia_reduction(x_dot, external_wrench);}
    
    // Inverse Kinematic for Joint Velocity
    q_dot = J.inverse() * x_dot;

    // Limit System Dynamic
    q_dot = limit_joint_dynamics(q_dot);
    x_dot_last_cycle = J * q_dot;

    if (use_position_control)
    {
        for (int i = 0; i < 6; ++i)
        {
            // target_joint_position[i] = joint_position[i] + q_dot[i] * duration.toSec();
            target_joint_position[i] += q_dot[i] * duration.toSec();
        }
    }

    // ---- DEBUG ---- //
    if (complete_debug) {ROS_INFO_THROTTLE_NAMED(2, "Desired Cartesian Velocity", "Desired Cartesian Velocity:  %.2f  %.2f  %.2f  %.2f  %.2f  %.2f", x_dot[0], x_dot[1], x_dot[2], x_dot[3], x_dot[4], x_dot[5]);}
    if (complete_debug) {ROS_INFO_THROTTLE_NAMED(2, "Desired Joints Velocity",    "Desired Joints Velocity:     %.2f  %.2f  %.2f  %.2f  %.2f  %.2f", q_dot[0], q_dot[1], q_dot[2], q_dot[3], q_dot[4], q_dot[5]);}

    // ROS_INFO_THROTTLE_NAMED(1, "joint_position", "%.2f  %.2f  %.2f  %.2f  %.2f  %.2f",joint_position[0],joint_position[1],joint_position[2],joint_position[3],joint_position[4],joint_position[5]);
    
    if (trajectory_debug){
        q_dot_real_debug.push_back(q_dot);
        x_real_debug.push_back(x);
        xe_real_debug.push_back(xe);
        x_dot_real_debug.push_back(x_dot);
        J_inv_debug.push_back(J.inverse());
    }
}

//----------------------------------------------- LIMIT DYNAMICS FUNCTIONS ----------------------------------------------//


Vector6d admittance_control::limit_joint_dynamics (Vector6d joint_velocity) {

    double duration = loop_rate.expectedCycleTime().toSec();

    // Limit Joint Velocity

    for (int i = 0; i < joint_velocity.size(); i++) {

        if (fabs(joint_velocity[i]) > max_vel[i]) {

            // ---- DEBUG ---- //
            // ROS_INFO_NAMED("Reached Maximum Velocity", "Reached Maximum Velocity on Joint %d   ->   Velocity: %.3f   Limited at: %.3f", i, joint_velocity[i], sign(joint_velocity[i]) * max_vel[i]);
            joint_velocity[i] = sign(joint_velocity[i]) * max_vel[i];
        }
    }

    // Limit Joint Acceleration

    for (int i = 0; i < joint_velocity.size(); i++) {

        if (fabs(joint_velocity[i] - q_dot_last_cycle[i]) > max_acc[i] * duration) {

            // ---- DEBUG ---- //
            // ROS_INFO_NAMED("Reached Maximum Acceleration", "Reached Maximum Acceleration on Joint %d   ->   Acceleration: %.3f   Limited at: %.3f", i, (joint_velocity[i] - q_dot_last_cycle[i]) / duration, q_dot_last_cycle[i] +  sign(joint_velocity[i] - q_dot_last_cycle[i]) * max_acc[i]);
            joint_velocity[i] = q_dot_last_cycle[i] + sign(joint_velocity[i] - q_dot_last_cycle[i]) * max_acc[i] * duration;
        
        }

    }

    q_dot_last_cycle = joint_velocity;
    return joint_velocity;

}

Vector6d admittance_control::compute_inertia_reduction (Vector6d velocity, Vector6d wrench) {

    Array6d reduction, x_vel_array(velocity);

    // Returns 0 if the Wrench is 0, 1 otherwise
    for (unsigned i = 0; i < wrench.size(); i++) {
        
        if (wrench[i] == 0.0) {reduction[i] = 0;}
        else {reduction[i] = 1;}
    
    }

    return Vector6d(x_vel_array *= reduction);

}

void admittance_control::stop_robot (void) {
    
    Vector6d stop;
    stop.setZero();

    send_velocity_to_robot(stop);

}

inline void admittance_control::clear_online_trajectory(){
    trajectory_online.clear();
    trajectory_online_dot.clear();
    trajectory_online_dotdot.clear();

    x_real_online.clear();
    x_dot_real_online.clear();  
    x_dotdot_real_online.clear();  
    xe_real_online.clear();
}

void admittance_control::get_cartesian_volocity(const Vector6d& pose_cur, const Vector6d& pose_last, Vector6d& vel, double time){
    double control_f = (double)1.0/time;

    Matrix3d rotation_e;
    Eigen::Vector3d v, //velocity
             w; //rotation_omega
    Vector6d ee_pose_vel_temp;
    vel.setZero();

    v = control_f * (pose_cur-pose_last).topLeftCorner(3,1);
    rotation_e = from_rotation_vector_to_matrix(pose_cur.bottomLeftCorner(3,1))
                *from_rotation_vector_to_matrix(pose_last.bottomLeftCorner(3,1)).transpose();
    w = control_f * from_rotation_matrix_to_vector(rotation_e);
    vel << v , w;
}

void admittance_control::get_cartesian_acceleration(const Vector6d& vel_cur, const Vector6d& vel_last, Vector6d& acc, double time){
    acc = (vel_cur - vel_last)/time;
}

double admittance_control::get_max_q_distance(const std::vector<double>& joint_1, const std::vector<double>& joint_2){
    double max = 0.0;
    double cur_dis = 0.0;
    for(auto i=0; i<joint_1.size(); ++i){
        cur_dis = fabs(joint_1[i] - joint_2[i]);
        if(max < cur_dis) max = cur_dis;
    }
    return max;
}


//------------------------------------------------Compliance Trajectory FUNCTIONS-------------------------------------------------//
        
bool admittance_control::create_TCP_z_lmove_trajectory(double length_mm, double time_ms){

    if ( length_mm == 0.0 || time_ms<= 0.0){
        ROS_WARN("Wrong length or time set");
        return false;
    }
    Vector6d current_TCP_world = get_x_from_current_pose(joint_position, joint_velocity);
    create_TCP_z_lmove_trajectory(current_TCP_world, length_mm, time_ms);
    return true;
}
        
void admittance_control::create_TCP_z_lmove_trajectory(Vector6d init_pose, double length_mm, double time_ms){
    
    ee_world_trajectory.clear();
    ee_world_trajectory_dot.clear();

    Vector6d current_TCP_world = init_pose;

    Eigen::Isometry3d target_TCP_world_iso3 = Eigen::Isometry3d::Identity();
    Eigen::Vector3d translation(current_TCP_world(0),current_TCP_world(1),current_TCP_world(2)); 
    target_TCP_world_iso3.pretranslate(translation);
    target_TCP_world_iso3.rotate(
        from_rotation_vector_to_matrix(current_TCP_world.bottomLeftCorner(3,1))
    );
    Eigen::Vector3d target_TCP_translation {0, 0, length_mm/1000};
    target_TCP_world_iso3.translate(target_TCP_translation);

    Vector6d target_TCP_world;
    target_TCP_world.topLeftCorner(3,1) = target_TCP_world_iso3.translation();
    target_TCP_world.bottomLeftCorner(3,1) = from_rotation_matrix_to_vector(target_TCP_world_iso3.rotation());

    // double step = std::round(length_mm/time_ms);
    double control_period_ms = loop_rate.expectedCycleTime().toSec()*1000;
    unsigned int points_num = (unsigned int)(std::round((time_ms+400)/control_period_ms));

    /****************pchip插值******************/
    /* usage example
    /* double x[5],y[5],x_new[200],y_new[200];
    /* pchip(x,y,5,x_new,200,y_new);
    /******************************************/
    std::vector<double> way_points_x;
    std::vector<double> way_points_y;
    std::vector<double> way_points_z;
    std::vector<double> new_x(points_num);
    std::vector<double> new_y(points_num);
    std::vector<double> new_z(points_num);

    //线性初插值，保证线性度
    size_t mid_waypoints_num = 5; // interpotate 5 mid-waypoints
    std::vector<Vector6d> mid_waypoints(mid_waypoints_num);

    for(size_t i=0; i<mid_waypoints_num; ++i){
        // mid_waypoints.at(i) = (current_TCP_world * (mid_waypoints_num  - i) + target_TCP_world * (i+1))/(mid_waypoints_num+1);
        mid_waypoints.at(i) = current_TCP_world + (i+1)*(target_TCP_world - current_TCP_world)/(mid_waypoints_num+1);
    }

    for(unsigned int i=0;i<3;i++){way_points_x.push_back(current_TCP_world(0));}
    for(unsigned int i=0;i<mid_waypoints_num;i++){way_points_x.push_back(mid_waypoints.at(i)(0));}
    for(unsigned int i=0;i<3;i++){way_points_x.push_back(target_TCP_world(0));}

    for(unsigned int i=0;i<3;i++){way_points_y.push_back(current_TCP_world(1));}
    for(unsigned int i=0;i<mid_waypoints_num;i++){way_points_y.push_back(mid_waypoints.at(i)(1));}
    for(unsigned int i=0;i<3;i++){way_points_y.push_back(target_TCP_world(1));}

    for(unsigned int i=0;i<3;i++){way_points_z.push_back(current_TCP_world(2));}
    for(unsigned int i=0;i<mid_waypoints_num;i++){way_points_z.push_back(mid_waypoints.at(i)(2));}
    for(unsigned int i=0;i<3;i++){way_points_z.push_back(target_TCP_world(2));}
    
    //pchip
    std::vector<double> time; //未插值时各路点对应的时间
    time.reserve(mid_waypoints_num+6);
    time = {0, 0.1, 0.2};
    for(size_t i=0; i<mid_waypoints_num; ++i){
        time.push_back((i+1)*time_ms/(1000*(mid_waypoints_num+1))+0.2);
    }
    time.push_back(time_ms/1000+0.2);   
    time.push_back(time_ms/1000+0.3);
    time.push_back(time_ms/1000+0.4);   


    double time_new[points_num];
    for(unsigned int i=0;i<points_num;++i){time_new[i]=(double)i*control_period_ms/1000.0;}

    pchip(&time[0], &way_points_x[0],mid_waypoints_num+6,time_new,points_num,&new_x[0]);
    pchip(&time[0], &way_points_y[0],mid_waypoints_num+6,time_new,points_num,&new_y[0]);
    pchip(&time[0], &way_points_z[0],mid_waypoints_num+6,time_new,points_num,&new_z[0]);
    
    Vector6d temp;
    temp.setZero();
    temp.bottomLeftCorner(3,1)=current_TCP_world.bottomLeftCorner(3,1);
    for(unsigned int i=0; i<points_num; ++i){
        temp.topLeftCorner(3,1) = Eigen::Vector3d(new_x.at(i), new_y.at(i), new_z.at(i));
        ee_world_trajectory.push_back(temp);
    }
    ee_world_trajectory.push_back(target_TCP_world);   

    double control_f = 1000/control_period_ms;
    //求解速度
    temp.setZero();
    for(auto iter = ee_world_trajectory.begin(); iter != ee_world_trajectory.end()-1; ++iter ){
        ee_world_trajectory_dot.push_back( control_f * (*(iter+1) - *iter) );
    }
    //因为是直线，角度全为0就完事了，如果轨迹为任意轨迹，需要用Re=R Rd^T来求解
    ee_world_trajectory_dot.push_back(temp);
    
    if(trajectory_debug){
        ROS_INFO("The size of desired waypoints after interpolation: %ld", ee_world_trajectory.size());
    }
}

void admittance_control::Do_TCP_z_lmove_trajectory(double length_mm, double time_ms){
    
    ros::spinOnce();
    create_TCP_z_lmove_trajectory(length_mm, time_ms);
    for(long unsigned int i=0; i<ee_world_trajectory.size(); ++i){
        compute_admittance_trajectory(ee_world_trajectory.at(i), ee_world_trajectory_dot.at(i));
        send_velocity_to_robot(q_dot);
        ros::spinOnce();
        loop_rate.sleep();
    }
    ee_world_trajectory.clear();
    ee_world_trajectory_dot.clear();

}

void admittance_control::Do_TCP_z_lmove_trajectory(Vector6d init_pose, double length_mm, double time_ms){
    
    ros::spinOnce();
    create_TCP_z_lmove_trajectory(init_pose, length_mm, time_ms);
    for(long unsigned int i=0; i<ee_world_trajectory.size(); ++i){
        compute_admittance_trajectory(ee_world_trajectory.at(i), ee_world_trajectory_dot.at(i));
        send_velocity_to_robot(q_dot);
        ros::spinOnce();
        loop_rate.sleep();
    }
    ee_world_trajectory.clear();
    ee_world_trajectory_dot.clear();
}

bool admittance_control::Do_compliance_jmove_trajectory(void){
    
    ros::spinOnce();
    if(!get_compliance_trajectory_flag || ee_world_trajectory.empty() || ee_world_trajectory_dot.empty()){
        ROS_WARN("Trajectory execution failed!");
        return false;
    }

    if(trajectory_debug){
        // --- clear debug variables --- //
        q_dot_real_debug.clear();
        x_real_debug.clear();
        xe_real_debug.clear();
        x_dot_real_debug.clear();
        J_inv_debug.clear();       
    }

    for(long unsigned int i=0; i<ee_world_trajectory.size(); ++i){
        compute_admittance_trajectory(ee_world_trajectory.at(i), ee_world_trajectory_dot.at(i));
        send_velocity_to_robot(q_dot);
        ros::spinOnce();
        loop_rate.sleep();
    }
    xd = ee_world_trajectory.back();
    ee_world_trajectory.clear();
    ee_world_trajectory_dot.clear();
    
    if(trajectory_debug){
        // --- save debug variables --- //
        compliance_trajectory_RT_debug_csv();
        // --- debug file name count ++ ---//
        trajectory_count++;
    }

    // --- switch mode back to fixed_point ---//
    compliance_mode = Fixed_Point;

    // --- reset trajectory flag ---//
    get_compliance_trajectory_flag = false;

    return true;
}

/**
 * @brief create interpolated trajectory from preset joint trajectory,  compliance_desired_joint_trajectory should be set first
 * 
 * @param waypoint 
 * @param waypoint_dot 
 */
bool admittance_control::create_jmove_trajectory_from_joint_trajectory(void){
    ros::spinOnce();

    if(compliance_desired_joint_trajectory.joint_trajectory.points.empty()){
        ROS_WARN("No Jmove data!");
        return false;
    }

    ee_world_trajectory.clear();
    ee_world_trajectory_dot.clear();

    // double time_scaling_gain = 0.9;
    double duration_time_ms = time_scaling_gain * 1000 * compliance_desired_joint_trajectory.joint_trajectory.points.back().time_from_start.toSec();
    std::vector<std::vector<double>> way_points_6_joints;
    std::vector<double> way_points_joint_temp;
    for(size_t i=0;i<6;++i){
        way_points_joint_temp.clear();
        //3个头
        for(unsigned int j=0;j<3;j++){way_points_joint_temp.push_back(compliance_desired_joint_trajectory.joint_trajectory.points.front().positions.at(i));}
        //掐头去尾中间点
        for(auto iter= compliance_desired_joint_trajectory.joint_trajectory.points.begin()+1; iter!=compliance_desired_joint_trajectory.joint_trajectory.points.end()-1;iter++){
            way_points_joint_temp.push_back(iter->positions.at(i));
        }
        //3个尾
        for(unsigned int j=0;j<3;j++){way_points_joint_temp.push_back(compliance_desired_joint_trajectory.joint_trajectory.points.back().positions.at(i));}
        way_points_6_joints.push_back(way_points_joint_temp);
    }

    /****************pchip插值******************/
    /* double x[5],y[5],x_new[200],y_new[200];
    /* pchip(x,y,5,x_new,200,y_new);
    /******************************************/
    double control_period_ms = loop_rate.expectedCycleTime().toSec()*1000;
    unsigned int points_num_interpolation = (unsigned int)(std::round((duration_time_ms
                                                                       + compliance_trajectory_start_time
                                                                       + compliance_trajectory_end_time)
                                                                       / control_period_ms));
    size_t way_points_num = 4 + compliance_desired_joint_trajectory.joint_trajectory.points.size();
    
    std::vector<double> time; //未插值时各路点对应的时间
    time.reserve(way_points_num);
    time = {0, compliance_trajectory_start_time/2000};
    for(auto iter = compliance_desired_joint_trajectory.joint_trajectory.points.begin();iter!=compliance_desired_joint_trajectory.joint_trajectory.points.end();++iter){
        time.push_back(iter->time_from_start.toSec()*time_scaling_gain + compliance_trajectory_start_time/1000);
    }
    time.push_back(time.back()+compliance_trajectory_end_time/2000);
    time.push_back(time.back()+compliance_trajectory_end_time/1000);   

    std::vector<std::vector<double>> way_points_6_joints_interpolation;
    std::vector<double> way_points_joint_temp_interpolation(points_num_interpolation);

    double time_new[points_num_interpolation];
    std::vector<double> joint_temp;
    for(unsigned int i=0;i<points_num_interpolation;++i){time_new[i]=(double)i*control_period_ms/1000;}
    for(size_t i=0;i<6;++i){
        joint_temp = way_points_6_joints.at(i);
        pchip(&time[0], &joint_temp[0],way_points_num,time_new,points_num_interpolation,&way_points_joint_temp_interpolation[0]);
                                    //第i个轴的第0个值的引用的指针
        way_points_6_joints_interpolation.push_back(way_points_joint_temp_interpolation);
    }

    //fk
    ee_world_trajectory.clear();
    ee_world_trajectory.reserve(points_num_interpolation);
    std::vector<double> joint_state_temp;
    Vector6d ee_pose_temp;

    moveit::core::RobotStatePtr kinematic_state_test = robot_state::RobotStatePtr(new robot_state::RobotState(kinematic_model));

    for(size_t i=0;i<points_num_interpolation;++i){
        joint_state_temp.clear();
        joint_state_temp = {way_points_6_joints_interpolation[0][i],
                            way_points_6_joints_interpolation[1][i],
                            way_points_6_joints_interpolation[2][i],
                            way_points_6_joints_interpolation[3][i],
                            way_points_6_joints_interpolation[4][i],
                            way_points_6_joints_interpolation[5][i]};
        get_x_from_joint_states(ee_pose_temp, joint_state_temp, kinematic_state_test);
        ee_world_trajectory.push_back(ee_pose_temp);
    }
    // auto x = get_x_from_current_pose(joint_position, joint_velocity);
    //求解速度
    double control_f = (double)1000/control_period_ms;
    Matrix3d rotation_e;
    Eigen::Vector3d ee_v; //velocity
    Eigen::Vector3d ee_w; //rotation
    Vector6d ee_pose_vel_temp;
    ee_pose_vel_temp.setZero();

    for(auto iter = ee_world_trajectory.begin(); iter != ee_world_trajectory.end()-1; ++iter ){
        ee_v = control_f * (*(iter+1) - *iter).topLeftCorner(3,1);
        rotation_e = from_rotation_vector_to_matrix((iter+1)->bottomLeftCorner(3,1))
                    *from_rotation_vector_to_matrix(iter->bottomLeftCorner(3,1)).transpose();
        ee_w = control_f * from_rotation_matrix_to_vector(rotation_e);
        ee_pose_vel_temp << ee_v, ee_w;
        ee_world_trajectory_dot.push_back(ee_pose_vel_temp);
    }
    ee_world_trajectory_dot.push_back(ee_pose_vel_temp);
    //角度需要用Re=Rt+1 Rt^T来求解  

    if(trajectory_debug){
        // --- Save Debug Variables ---//
        compliance_trajectory_CT_debug_csv(way_points_6_joints_interpolation);
    }
    ROS_INFO("Executing Compliance Trajectory, waypoints size: %ld \n", ee_world_trajectory.size());
    
    compliance_desired_joint_trajectory.joint_trajectory.points.clear();

    return true;
}


// std::vector<Vector6d> admittance_control::compute_desired_velocities (std::vector<Vector6d> q_des, double sampling_time) {

//     std::vector<Vector6d> q_dot_des;

//     // Add Stop-Point in the end of q_des
//     q_des.push_back(q_des[q_des.size()-1]);

//     // Compute q̇_des[k] = (q_des[k+1] - q_des[k]) / τ, τ = sampling_time
//     for (unsigned int i = 0; i < q_des.size() - 1; i++) {

//         Array6d q_dot_des_array = (q_des[i+1].array() - q_des[i].array()) / sampling_time;
//         q_dot_des.push_back(Vector6d(q_dot_des_array));
        
//     }

//     // ---- DEBUG ---- //
//     if (complete_debug) {csv_debug(q_dot_des, "q_dot_des");}

//     return q_dot_des;

// }


//-------------------------------------------- SPLINE INTERPOLATION FUNCTION --------------------------------------------//


std::vector<tk::spline> admittance_control::spline_interpolation (std::vector<Vector6d> data_vector, double spline_lenght, std::string output_file) {

    // Creation of Spline6d Vector -> Usage: spline6d[joint_number](s) = q(s)
    std::vector<tk::spline> spline6d;

    // Creation of s € [0,spline_lenght] vector
    std::vector<double> s;
    for (unsigned i = 0; i < data_vector.size(); i++) {
        double s_i = spline_lenght * (i / (double(data_vector.size()) - 1));
        s.push_back(s_i);
    }

    // Compute Spline for each Joint
    for (unsigned joint_number = 0; joint_number < 6; joint_number++) {

        // Create a Single-Joint Vector
        std::vector<double> waypoints_1d;
        for (unsigned i = 0; i < data_vector.size(); i++) {waypoints_1d.push_back(data_vector[i][joint_number]);}

        // Compute Cubic Spline [Q(s), s € [0,T]]
        tk::spline spline1d;
        spline1d.set_points(s, waypoints_1d);

        // Add Results to "spline6d" Vector
        spline6d.push_back(spline1d);

    }

    // ---- DEBUG ---- //
    if (complete_debug) {csv_debug(spline6d, s, data_vector, output_file);}
    
    return spline6d;

}

std::vector<tk::spline> admittance_control::spline_interpolation (std::vector<Array6d> data_vector, double spline_lenght, std::string output_file) {

    // Converting Array into Vector
    std::vector<Vector6d> data_vector_temp;
    for (unsigned int i = 0; i < data_vector.size(); i++) {data_vector_temp.push_back(data_vector[i].matrix());}

    return spline_interpolation(data_vector_temp, spline_lenght, output_file);

}

Vector6d admittance_control::get_spline_value (std::vector<tk::spline> spline6d, double s) {

    Vector6d spline_value;
    spline_value.resize(6);
    
    // Get spline1d value for each joint
    for (unsigned int i = 0; i < 6; i++) {spline_value[i] = spline6d[i](s);}

    return spline_value;

}


        
// ------------------------------------------------- COLLISION CHECK --------------------------------------------------- //
/**
 * @brief check if the robot is in the self collision state in current joint states
 * 
 * @return true 
 * @return false 
 */

bool admittance_control::is_self_collision(){
    ros::spinOnce();
    return is_self_collision(joint_position);
}

/**
 * @brief check if the robot is in the self collision state in given joint states
 * 
 * @param joint_values 
 * @return true collision
 * @return false no collision
 */
bool admittance_control::is_self_collision(std::vector<double> joint_values){
    kinematic_state->setJointGroupPositions(joint_model_group, joint_values);
    kinematic_state->setJointGroupVelocities(joint_model_group, joint_velocity);

    collision_result->clear();

    planning_scene->checkSelfCollision(*collision_request, *collision_result, *kinematic_state);

    return collision_result->collision;
}

/**
 * @brief check if the robot is in the singularity in given joint states
 * 
 * @param joint_values 
 * @return true 
 * @return false 
 */
bool admittance_control::is_self_singular(std::vector<double> joint_values){
    // Compute Manipulator Jacobian
    auto Jacobian = compute_arm_jacobian(joint_values, joint_velocity);
    // Calculate the minimum singular value
    double singular_value = singular_avoid->singuar_value_decomposition(Jacobian);
    // ROS_INFO_THROTTLE(2, "min singular value: %f", singular_value );
    //Calculate the determinant value
    double det_Jacobian_value = J.determinant();

    return singular_value<0.1 || det_Jacobian_value==0;
}

//-------------------------------------------------- CONTROL FUNCTIONS --------------------------------------------------//


void admittance_control::send_velocity_to_robot (Vector6d velocity) {

    std_msgs::Float64MultiArray msg;

    std::vector<double> velocity_vector(velocity.data(), velocity.data() + velocity.size());

    msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
    msg.layout.dim[0].size = velocity.size();
    msg.layout.dim[0].stride = 1;
    msg.layout.dim[0].label = "velocity";

    // copy in the data
    msg.data.clear();
    msg.data.insert(msg.data.end(), velocity_vector.begin(), velocity_vector.end());

    joint_group_vel_controller_publisher.publish(msg);

}

void admittance_control::send_position_to_robot(Vector6d position)
{
    std_msgs::Float64MultiArray msg;

    std::vector<double> velocity_vector(position.data(), position.data() + position.size());

    msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
    msg.layout.dim[0].size = position.size();
    msg.layout.dim[0].stride = 1;
    msg.layout.dim[0].label = "position";

    // copy in the data
    msg.data.clear();
    msg.data.insert(msg.data.end(), velocity_vector.begin(), velocity_vector.end());

    joint_group_pos_controller_publisher.publish(msg);
}

void admittance_control::wait_for_position_reached (Vector6d desired_position, double maximum_time) {

    joint_state_callback = false;

    // Wait for Joint State Callback
    while (ros::ok() && !joint_state_callback) {ros::spinOnce();}

    Vector6d current_position(joint_state.position.data());

    ros::Time start_time = ros::Time::now();

    // Wait until desired_position and current_position are equal, with a little tolerance
    while (ros::ok() && (Eigen::abs(desired_position.array() - current_position.array()) > 0.0001).all() && (ros::Time::now() - start_time).toSec() < maximum_time) {

        ros::spinOnce();
        current_position = Vector6d(joint_state.position.data());
        ROS_INFO_ONCE("Wait for Starting Position...");
        
    }

}

bool admittance_control::switch_controller_to_joint_group(void)
{
    std::string which_controller = use_position_control ? "joint_group_pos_controller" : "joint_group_vel_controller";
    
    // Switch Controller (Position_traj to admittance_interface) 
    if (switch_off_controller("scaled_pos_joint_traj_controller") && switch_on_controller(which_controller))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool admittance_control::switch_controller_to_scaled_pose(void)
{
    std::string which_controller = use_position_control ? "joint_group_pos_controller" : "joint_group_vel_controller";
    // Switch Controller (admittance_interface to Position_traj)
    if (switch_off_controller(which_controller) && switch_on_controller("scaled_pos_joint_traj_controller"))
    {
        return true;
    }
    else
    {
        return false;
    }
}


bool admittance_control::switch_on_controller(std::string controller_name){
    
    list_controllers_client.call(list_controllers_srv);
    switch_controller_srv.request.start_controllers.clear();
    switch_controller_srv.request.stop_controllers.clear();   
     
    std::vector<controller_manager_msgs::ControllerState> controller_list = list_controllers_srv.response.controller;
    for(auto iter = controller_list.begin();iter!=controller_list.end();++iter){
        if(iter->name == controller_name){
            if(iter->state == "stopped" || iter->state == "initialized"){
                switch_controller_srv.request.start_controllers.push_back( controller_name );
                switch_controller_srv.request.strictness = switch_controller_srv.request.STRICT;
                switch_controller_client.call(switch_controller_srv);
                if (switch_controller_srv.response.ok) {
                    while(true){
                        ros::Duration(0.1).sleep(); 
                        list_controllers_client.call(list_controllers_srv);
                        controller_list = list_controllers_srv.response.controller;
                        for(auto iter = controller_list.begin();iter!=controller_list.end();++iter){
                            if(iter->name == controller_name){
                                if(iter->state == "running"){
                                    ROS_INFO("Started: \"%s\" ", switch_controller_srv.request.start_controllers[0].c_str());
                                    return true;
                                }
                            }else if(iter == controller_list.end()-1){
                                ROS_WARN("Unknown error with controller: %s ", controller_name.c_str());
                                return false;
                            }
                        }
                    }
                } 
                else {
                    ROS_WARN("Failed to Start Controllers \"%s\" ", switch_controller_srv.request.start_controllers[0].c_str());
                    return false;
                }
            }
            else {ROS_WARN("%s is already running ", controller_name.c_str());}
            return true;
        }
        else if(iter == controller_list.end()-1){
            ROS_WARN("%s is not loaded", controller_name.c_str());
            return false;
        }
    }
}

bool admittance_control::switch_off_controller(std::string controller_name){
    
    list_controllers_client.call(list_controllers_srv);
    switch_controller_srv.request.start_controllers.clear();
    switch_controller_srv.request.stop_controllers.clear();

    std::vector<controller_manager_msgs::ControllerState> controller_list = list_controllers_srv.response.controller;

    for(auto iter = controller_list.begin();iter!=controller_list.end();++iter){
        if(iter->name == controller_name){
            if(iter->state == "running"){

                switch_controller_srv.request.stop_controllers.push_back(controller_name);
                switch_controller_srv.request.strictness = switch_controller_srv.request.STRICT;
                switch_controller_client.call(switch_controller_srv);
                if (switch_controller_srv.response.ok) {
                    while(true){
                        ros::Duration(0.1).sleep(); 
                        list_controllers_client.call(list_controllers_srv);
                        controller_list = list_controllers_srv.response.controller;
                        for(auto iter = controller_list.begin();iter!=controller_list.end();++iter){
                            if(iter->name == controller_name){
                                if(iter->state == "stopped"){
                                    ROS_INFO("Stoped: \"%s\" ", switch_controller_srv.request.stop_controllers[0].c_str());
                                    ros::Duration(0.1).sleep();
                                    return true;
                                }
                            }else if(iter == controller_list.end()-1){
                                ROS_WARN("Unknown error with controller: %s ", controller_name.c_str());
                                return false;
                            }
                        }
                    }
                } 
                else {
                    ROS_WARN("Failed to Stoped Controllers \"%s\"", switch_controller_srv.request.stop_controllers[0].c_str());
                    return false;
                }
            } 
            else {ROS_WARN("%s is already stopped", controller_name.c_str());}
            return true;
        }
        else if(iter == controller_list.end()-1){
            ROS_WARN("%s is not loaded", controller_name.c_str());
            return false;
        }
    }
}


//--------------------------------------------------- UR e FUNCTIONS ---------------------------------------------------//

void admittance_control::wait_for_callbacks_initialization (void) {

    ros::spinOnce();

    // Wait for the Callbacks
    while (ros::ok() && (!force_callback || !joint_state_callback)) {

        ros::spinOnce();
        
        if (!force_callback) {ROS_WARN_THROTTLE(2, "Wait for Force Sensor");}
        if (!joint_state_callback) {ROS_WARN_THROTTLE(2, "Wait for Joint State Feedback");}

    }

    std::cout << std::endl;

}

void admittance_control::zero_ft_sensor (void) {
    //TODO 
    // while (use_ur_real_robot && !zero_ft_sensor_client.call(zero_ft_sensor_srv)) {ROS_WARN_THROTTLE(2,"Wait for Service: \"/ur_hardware_interface/zero_ftsensor\"");}
    // if (zero_ft_sensor_client.call(zero_ft_sensor_srv)) ROS_INFO("Succesful Request \"zero_ftsensor\" ");

    // ur_send_script_command("zero_ftsensor()");

}

//--------------------------------------------------- USEFUL FUNCTIONS ---------------------------------------------------//


void admittance_control::publish_cartesian_position (std::vector<double> joint_position,  std::vector<double> joint_velocity) {
    //ee_position 末端執行器位姿矩陣 T
    Eigen::Matrix4d ee_position = compute_fk (joint_position, joint_velocity);

    Eigen::Matrix3d rotation_matrix = ee_position.block<3,3>(0,0);
    Eigen::Vector3d translation_vector = ee_position.block<3,1>(0,3);
    Eigen::Quaterniond rotation_quaternion(rotation_matrix);

    geometry_msgs::Pose pose;

    pose.position.x = translation_vector[0];
    pose.position.y = translation_vector[1];
    pose.position.z = translation_vector[2];

    pose.orientation.x = rotation_quaternion.x();
    pose.orientation.y = rotation_quaternion.y();
    pose.orientation.z = rotation_quaternion.z();
    pose.orientation.w = rotation_quaternion.w();
    //轉換成ROS geometry_msgs::Pose 消息格式並發布出去
    cartesian_position_publisher.publish(pose);


    //publish the pose based on ref pose 
    Eigen::Matrix4d ee_cylinder_position = EEF_ref_inv * ee_position;

    rotation_matrix = ee_cylinder_position.block<3,3>(0,0);
    translation_vector = ee_cylinder_position.block<3,1>(0,3);
    rotation_quaternion =  rotation_matrix;

    pose.position.x = translation_vector[0];
    pose.position.y = translation_vector[1];
    pose.position.z = translation_vector[2];

    pose.orientation.x = rotation_quaternion.x();
    pose.orientation.y = rotation_quaternion.y();
    pose.orientation.z = rotation_quaternion.z();
    pose.orientation.w = rotation_quaternion.w();

    cartesian_position_rel_publisher.publish(pose);
}

bool admittance_control::set_EEF_ref_pose_Service_Callback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res){
    ros::spinOnce();

    auto pose = compute_fk(joint_position, joint_velocity);
    EEF_ref_inv = pose.inverse();
    ROS_INFO_STREAM("Set ref pose at: " << pose);
    res.success = true;
    return true;
}

bool admittance_control::get_MKB_Service_Callback(self_defined_msgs::get_MKB::Request &req, self_defined_msgs::get_MKB::Response &res){
    for(size_t i=0; i<6; ++i){
        res.M.push_back(mass_matrix(i,i));
        res.B.push_back(damping_matrix(i,i));
        res.K.push_back(spring_matrix(i,i));
    }
    res.success = true;
    return true;
}

bool admittance_control::set_MKB_Service_Callback(self_defined_msgs::set_MKB::Request &req, self_defined_msgs::set_MKB::Response &res){
    if(req.B.size()!=6 || req.M.size()!=6 || req.M.size()!=6){
        res.success = false;
        ROS_WARN("Wrong parameter dimension, M,K,B should be in shape of (1*6), we got: M(1*%lu), K(1*%lu), B(1*%lu)", req.M.size(), req.K.size(), req.B.size());
    }else{
        for(size_t i=0; i<6; ++i){
            mass_matrix(i,i) = req.M[i];
            spring_matrix(i,i) = req.K[i];
            damping_matrix(i,i) = req.B[i];
        }
        ROS_INFO_STREAM("Current M:"<< std::endl << mass_matrix    << std::endl <<
                        "Current K:"<< std::endl << spring_matrix  << std::endl <<
                        "Current B:"<< std::endl << damping_matrix << std::endl );
        res.success = true;
    }
    return true;
}

int admittance_control::sign (double num) {

    if (num >= 0) {return +1;}
    else {return -1;}
    
}

Eigen::Vector3d admittance_control::from_rotation_matrix_to_vector(Matrix3d rotation_matrix) {
    //旋轉矩陣轉歐拉角
    AngleAxisd rotation_vector;
    rotation_vector.fromRotationMatrix(rotation_matrix);
    return rotation_vector.angle()*rotation_vector.axis();

}

Matrix3d admittance_control::from_rotation_vector_to_matrix(Eigen::Vector3d rotation_vector) {

    AngleAxisd rotation_vector_angleaxisd;
    rotation_vector_angleaxisd.angle() = rotation_vector.norm();
    rotation_vector_angleaxisd.axis() = rotation_vector.normalized();

    return rotation_vector_angleaxisd.matrix();

}

Isometry3d admittance_control::from_vector6d_to_iso3d(const Vector6d& x){
    Eigen::Vector3d translation(x[0],x[1],x[2]);
    Matrix3d rotation = from_rotation_vector_to_matrix(Eigen::Vector3d(x[3], x[4], x[5]));
    Isometry3d result = Isometry3d::Identity();
    result.translate(translation);
    result.rotate(rotation);
    return result;
}

Vector6d admittance_control::from_iso3d_to_vector6d(const Isometry3d& T){
    Eigen::Vector3d rotation = from_rotation_matrix_to_vector(T.rotation());
    Vector6d result;
    result.topLeftCorner(3,1) = T.translation();
    result.bottomLeftCorner(3,1) = rotation;
    return result;
}



void admittance_control::set_xd_from_current_pose(void) {
    
    ros::spinOnce();
    xd.setZero();
    xd = get_x_from_current_pose(joint_position, joint_velocity);
    ROS_INFO("succesfully set xd at current pose");
    ROS_INFO_STREAM("set xd:" << xd.transpose() << "\r\n");

}



//--------------------------------------------- VARIABLE CREATION FUNCTIONS ---------------------------------------------//

Vector6d admittance_control::new_vector_6d (double x, double y, double z, double roll, double pitch, double yaw) {

    Vector6d temp;
    temp.setZero();

    temp[0] = x;    temp[1] = y;     temp[2] = z;
    temp[3] = roll; temp[4] = pitch; temp[5] = yaw;

    return temp;

}


//------------------------------------------------------- DEBUG --------------------------------------------------------//


void admittance_control::csv_debug (std::vector<double> vector, std::string name) {

    std::string package_path = ros::package::getPath("admittance_controller");
    std::string save_file = package_path + "/debug/" + name + "_debug.csv";
    std::ofstream vector_debug = std::ofstream(save_file);
    vector_debug << name << "\n";
    for (unsigned int i = 0; i < vector.size(); i++) {vector_debug << vector[i] << "\n";}
    vector_debug.close();

}

void admittance_control::csv_debug (std::vector<Vector6d> vector6d, std::string name) {

    std::string package_path = ros::package::getPath("admittance_controller");
    std::string save_file = package_path + "/debug/" + name + "_debug.csv";
    std::ofstream vector_debug = std::ofstream(save_file);
    vector_debug << name << "\n";
    for (unsigned int i = 0; i < vector6d.size(); i++) {
        for (unsigned int joint_n = 0; joint_n < 6; joint_n++) {vector_debug << vector6d[i][joint_n] << ",";} vector_debug << "\n";}
    vector_debug.close();

}

void admittance_control::csv_debug (std::vector<tk::spline> spline6d, std::vector<double> s, std::vector<Vector6d> data_vector, std::string name) {
    
    std::string package_path = ros::package::getPath("admittance_controller");
    std::string save_file = package_path + "/debug/" + name + "_spline6d_debug.csv";
    std::ofstream spline6d_debug = std::ofstream(save_file);
    spline6d_debug << "s, ,Joint1,Joint2,Joint3,Joint4,Joint5,Joint6\n\n";
    for (unsigned int i = 0; i < data_vector.size(); i++) { 
        spline6d_debug << s[i] << ", ,";
        for (unsigned int joint_n = 0; joint_n < 6; joint_n++) {spline6d_debug << spline6d[joint_n](s[i]) << ",";}
        spline6d_debug << "\n";
    }
    spline6d_debug.close();

}

void admittance_control::trajectory_debug_csv (std::vector<sensor_msgs::JointState> trajectory, std::string trajectory_name) {

    std::string package_path = ros::package::getPath("admittance_controller");
    std::string save_file = package_path + "/debug/" + trajectory_name + "_debug.csv";
    std::ofstream trajectory_debug = std::ofstream(save_file);
    trajectory_debug << "frame_id,seq,sec,nsec,     ,pos_joint1,pos_joint2,pos_joint2,pos_joint4,pos_joint5,pos_joint6,     ,vel_joint1,vel_joint2,vel_joint3,vel_joint4,vel_joint5,vel_joint6\n\n";
    for (unsigned int i = 0; i < trajectory.size(); i++) {
        trajectory_debug << trajectory[i].header.frame_id << "," << trajectory[i].header.seq << "," << trajectory[i].header.stamp.sec << "," << trajectory[i].header.stamp.nsec << ", ,";
        for (unsigned int joint_n = 0; joint_n < 6; joint_n++) {trajectory_debug << trajectory[i].position[joint_n] << ",";} trajectory_debug << " ,";
        for (unsigned int joint_n = 0; joint_n < 6; joint_n++) {trajectory_debug << trajectory[i].velocity[joint_n] << ",";} trajectory_debug << "\n";
    }
    trajectory_debug.close();
        
}

/**
 * @brief real trajectory debug
 * 
 * @param way_points_6_joints_interpolation 
 */
void admittance_control::compliance_trajectory_RT_debug_csv (void) {
    std::string package_path = ros::package::getPath("admittance_controller");
    std::string save_file = package_path + "/compliance_debug/" + std::to_string(trajectory_count) + "_trajectory_RT_debug.csv";
    std::ofstream trajectory_debug = std::ofstream(save_file);
    trajectory_debug << "q1_dot,q2_dot,q3_dot,q4_dot,q5_dot,q6_dot,x,y,z,rx,ry,rz,ex,ey,ez,erx,ery,erz,x_dot,y_dot,z_dot,wx,wy,wz\n\n";
    for (size_t i = 0; i < q_dot_real_debug.size(); i++) {
        for (size_t joint_n = 0; joint_n < 6; joint_n++) {trajectory_debug << q_dot_real_debug[i][joint_n] << ",";}
        for (size_t n = 0; n < 6; n++) {trajectory_debug << x_real_debug[i][n] << ",";} 
        for (size_t n = 0; n < 6; n++) {trajectory_debug << xe_real_debug[i][n] << ",";}
        for (size_t n = 0; n < 6; n++) {trajectory_debug << x_dot_real_debug[i][n] << ",";}      
        trajectory_debug << "\n";
    }
    trajectory_debug.close();
    // trajectory_debug.clear();

    save_file = package_path + "/compliance_debug/" + std::to_string(trajectory_count) + "_J_inv.csv";
    trajectory_debug = std::ofstream(save_file);    
    for (size_t i = 0; i < J_inv_debug.size(); i++) {
        for (size_t row = 0; row < 3; row++) {
            for (size_t col = 0; col < 3; col++) {
                trajectory_debug << J_inv_debug[i](row,col) << ",";
            }
        }
        trajectory_debug << "\n";
    }
    trajectory_debug.close();    
}

/**
 * @brief created trajectory debug
 * 
 * @param way_points_6_joints_interpolation 
 */
void admittance_control::compliance_trajectory_CT_debug_csv (std::vector<std::vector<double>> way_points_6_joints_interpolation) {

    std::string package_path = ros::package::getPath("admittance_controller");
    std::string save_file = package_path + "/compliance_debug/" + std::to_string(trajectory_count) + "_trajectory_CT_debug.csv";
    std::ofstream trajectory_debug = std::ofstream(save_file);
    trajectory_debug << "q1d,q2d,q3d,q4d,q5d,q6d,xd,yd,zd,rxd,ryd,rzd,x_dot_d,y_dot_d,z_dot_d,wx_d,wy_d,wz_d\n\n";
    for (size_t i = 0; i < way_points_6_joints_interpolation.at(0).size(); i++) {
        for (size_t joint_n = 0; joint_n < 6; joint_n++) {trajectory_debug << way_points_6_joints_interpolation[joint_n][i] << ",";}
        for (size_t n = 0; n < 6; n++) {trajectory_debug << ee_world_trajectory[i][n] << ",";} 
        for (size_t n = 0; n < 6; n++) {trajectory_debug << ee_world_trajectory_dot[i][n] << ",";}   
        trajectory_debug << "\n";
    }
    trajectory_debug.close();

}

/**
 * @brief acquire ft sensor data by @var Ft_sensor_data_series
 * 
 */
void admittance_control::ft_sensor_csv(size_t class_id, size_t data_id){
    std::string package_path = ros::package::getPath("admittance_controller");
    std::string save_file = package_path + "/ft_sensor/" + std::to_string(class_id) + "_" + std::to_string(data_id) + "_ft_data.csv";
    std::ofstream tf_sensor_fd = std::ofstream(save_file);
    tf_sensor_fd << "x,y,z,rx,ry,rz\n\n";
    for (auto iter = Ft_sensor_data_series.begin(); iter != Ft_sensor_data_series.end(); ++iter){
        for (size_t n = 0; n < 6; n++){tf_sensor_fd << (*iter)(n) << ",";}
        tf_sensor_fd << "\n";
    }
    tf_sensor_fd.close();
}

void admittance_control::trajectory_online_record(void){
    std::string package_path = ros::package::getPath("admittance_controller");
    std::string save_file = package_path + "/trajectory_online/" + std::to_string(trajectory_count_online) + "_trajectory_real_online.csv";
    std::ofstream trajectory_record = std::ofstream(save_file);
    trajectory_record << "No." << trajectory_count_online << "\n";
    trajectory_record << "time,x,y,z,rx,ry,rz,x_dot,y_dot,z_dot,wx,wy,wz,ax,ay,az,arpha_x,arpha_y,arpha_z,ex,ey,ez,erx,ery,erz\n\n";
    for (size_t i = 0; i < x_real_online.size(); i++) {
        // for (size_t joint_n = 0; joint_n < 6; joint_n++) {trajectory_record << q_dot_real_debug[i][joint_n] << ",";}
        trajectory_record << time_online[i] << ",";
        for (size_t n = 0; n < 6; n++) {trajectory_record << x_real_online[i][n] << ",";} 
        for (size_t n = 0; n < 6; n++) {trajectory_record << x_dot_real_online[i][n] << ",";} 
        for (size_t n = 0; n < 6; n++) {trajectory_record << x_dotdot_real_online[i][n] << ",";} 
        for (size_t n = 0; n < 6; n++) {trajectory_record << xe_real_online[i][n] << ",";}    
        trajectory_record << "\n";
    }
    trajectory_record.close();

    package_path = ros::package::getPath("admittance_controller");
    save_file = package_path + "/trajectory_online/" + std::to_string(trajectory_count_online) + "_trajectory_target_online.csv";
    trajectory_record = std::ofstream(save_file);
    trajectory_record << "No." << trajectory_count_online << "\n";
    trajectory_record << "time,x,y,z,rx,ry,rz,x_dot,y_dot,z_dot,wx,wy,wz,ax,ay,az,arpha_x,arpha_y,arpha_z\n\n";
    for (size_t i = 0; i < trajectory_online.size(); i++) {
        // for (size_t joint_n = 0; joint_n < 6; joint_n++) {trajectory_record << q_dot_real_debug[i][joint_n] << ",";}
        trajectory_record << time_online[i] << ",";
        for (size_t n = 0; n < 6; n++) {trajectory_record << trajectory_online[i][n] << ",";} 
        for (size_t n = 0; n < 6; n++) {trajectory_record << trajectory_online_dot[i][n] << ",";} 
        for (size_t n = 0; n < 6; n++) {trajectory_record << trajectory_online_dotdot[i][n] << ",";} 
        trajectory_record << "\n";
    }
    trajectory_record.close();

    trajectory_count_online++;
}

void admittance_control::publish_wrench(ros::Publisher &publisher, const Vector6d &wrench)
{
    geometry_msgs::WrenchStamped wrench_pub;
    wrench_pub.header.stamp = ros::Time::now();
    wrench_pub.wrench.force.x  = wrench[0];
    wrench_pub.wrench.force.y  = wrench[1];
    wrench_pub.wrench.force.z  = wrench[2];
    wrench_pub.wrench.torque.x = wrench[3];
    wrench_pub.wrench.torque.y = wrench[4];
    wrench_pub.wrench.torque.z = wrench[5];

    publisher.publish(wrench_pub);
}

//-------------------------------------------------------- MAIN --------------------------------------------------------//


void admittance_control::spinner (void) {
    ros::spinOnce();
    if(Do_Compliance){
        switch (compliance_mode)
        {
        case Fixed_Point:
            ros::spinOnce();
            compute_admittance();
            if (use_position_control)
                send_position_to_robot(target_joint_position);
            else
                send_velocity_to_robot(q_dot);
            loop_rate.sleep();
            break;

        case Follow_Trajectory_offline:
            loop_rate.sleep();
            break;
        
        case Follow_Trajectory_online:
            compute_admittance_trajectory_online();
            if (use_position_control)
                send_position_to_robot(target_joint_position);
            else
                send_velocity_to_robot(q_dot);
            loop_rate.sleep();
            break;
        
        default:
            loop_rate.sleep();
            break;
        }
    }
    else
    {
        loop_rate.sleep();
    }

}


LowPassFilter::LowPassFilter(int queue_length){
    filter_queue.clear();
    length = queue_length;
}

Vector6d LowPassFilter::filter_step(Vector6d input_vec) {

    // Adding new element to filter vector    
    filter_queue.push_back(input_vec);

    while (ros::ok() && filter_queue.size() > length) {filter_queue.erase(filter_queue.begin());}

    // Median Filter (media = sum / N_elements)
    Vector6d sum, median;
    for (unsigned int i = 0; i < filter_queue.size(); i++) {sum += filter_queue[i];}
    median = sum / filter_queue.size();
    
    return median;
}