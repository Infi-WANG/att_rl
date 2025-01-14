#include "joy_ur/joy_ur.h"

joy_ur::joy_ur(/* args */):
    nh("/"), nh_("~"), joy_control_on(false), mode_gain(1.0)
{   
    // service server
    set_current_pose_as_origin = nh.advertiseService("/admittance_controller/set_current_as_origin_for_TOL", &joy_ur::set_current_pose_as_origin_CB, this);

    //service client
    admittance_controller_switch_client = nh.serviceClient<std_srvs::SetBool>("/admittance_controller/admittance_controller_activation_service");
    follow_trajectory_online_switch_client = nh.serviceClient<std_srvs::SetBool>("/admittance_controller/follow_trajectory_online_activation_service");
    clear_TOL_equilibrium_offset_client = nh.serviceClient<std_srvs::Trigger>("/admittance_controller/clear_equilibrium_TOL_offset_service");
    go_home_client = nh.serviceClient<std_srvs::Trigger>("/UR_go_home");
    play_client = nh.serviceClient<std_srvs::Trigger>("/ur_hardware_interface/dashboard/play");
    stop_client = nh.serviceClient<std_srvs::Trigger>("/ur_hardware_interface/dashboard/stop");
    zero_tf_client = nh.serviceClient<std_srvs::Trigger>("/ur_hardware_interface/zero_ftsensor");
    set_EEF_ref_pose_client = nh.serviceClient<std_srvs::Trigger>("/admittance_controller/set_EEF_ref_pose_service");
    record_SAP_data_client = nh.serviceClient<std_srvs::SetBool>("/activate_sap_data_record");

    joy_sub = nh.subscribe("/joy", 1, &joy_ur::joy_Callback, this);

    cur_pose_sub = nh.subscribe("/ur_cartesian_pose",1, &joy_ur::cur_TCP_pose_sub_Callback, this);

    target_pose_pub = nh.advertise<geometry_msgs::Pose>("/admittance_controller/trajectory_online_target_pose", 1);
    gripper_target_position_pub = nh.advertise<std_msgs::Int32>("/robotiq_controller/target_position", 1);

    // ----- DATA -----//
    target_pose = Eigen::Isometry3d::Identity();

    cartesian_move_vel.resize(6);

    joy_last_state.axes.resize(8); //init joy states
    joy_last_state.axes.at(LT_BIT) = joy_last_state.axes.at(RT_BIT) = 1.0;
    joy_last_state.buttons.resize(11);

    if (!nh_.getParam("real_robot", real_robot)) {ROS_ERROR("Couldn't retrieve the real_robot parameter."); real_robot = false;}
    
    //flag
    cartesian_started = false;
    TCP_valid = false;
    TCP_based = true; //move based on the TCP frame by default

    wait_dependence();
}

joy_ur::~joy_ur()
{
}

void joy_ur::wait_dependence(){
    while(!TCP_valid && ros::ok()){
        ros::spinOnce();
        ROS_INFO_THROTTLE(1, "Waiting for the current TCP pose msg...");
        ros::Duration(0.1).sleep();
    }
    target_pose = cur_TCP_pose;
    ROS_INFO("Set current TCP pose as the target pose");
    zero_tf_client.waitForExistence();
    if(real_robot){
        ROS_INFO("Wait for the UR driver...");
        play_client.waitForExistence();
        stop_client.waitForExistence();
        ROS_INFO("Connected to the UR driver");
    }
}

bool joy_ur::set_current_pose_as_origin_CB(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res){
    res.success = reset_current_as_origin();
    return true;
}


bool joy_ur::reset_current_as_origin(){
    ros::spinOnce();
    if(TCP_valid){
        target_pose = cur_TCP_pose; 
        // target_pose = cur_TCP_pose;
        ROS_INFO("Set current TCP pose as the target pose");
    }else{
        ROS_WARN("Failed to read the current TCP pose");
        return false;
    }
    TCP_valid = false;
    zero_ft_sensor();
    return true;
}

bool joy_ur::go_home(){

    std::fill(cartesian_move_vel.begin(),cartesian_move_vel.end(), 0.0);

    if(joy_control_on){
        if(set_joy_control(false)) joy_control_on = false;
        ROS_INFO("Going home... turn off the admittance controller first");
    }
 
    std_srvs::Trigger srv;
    ROS_WARN_STREAM("Going home... ");

    if (!go_home_client.call(srv)){
        ROS_WARN("No response from position controller");
        if(!joy_control_on){
            if(set_joy_control(true)) joy_control_on = true;
        }
        reset_current_as_origin();
        return false;
    }else{
        set_cur_as_EEF_ref_pose();
        zero_ft_sensor();
        reset_current_as_origin();
        if(!joy_control_on){
            if(set_joy_control(true)) joy_control_on = true;
        }
        return srv.response.success;
    }
}

bool joy_ur::zero_ft_sensor(){
    std_srvs::Trigger srv;
    return(zero_tf_client.call(srv));
}

bool joy_ur::set_cur_as_EEF_ref_pose(){
    std_srvs::Trigger srv;    
    if(set_EEF_ref_pose_client.call(srv)){
        if(srv.response.success){
            ROS_INFO("Successfully call the set_EEF_ref_pose service");
            return true;
        }else{
            ROS_WARN("Failed to call the set_EEF_ref_pose service");
            return false;
        }
    }
    return false;
}

bool joy_ur::activate_sap_data_record(bool on_off){
    std_srvs::SetBool srv;
    srv.request.data = on_off;
    std::string msg = on_off? "on":"off";

    if(record_SAP_data_client.call(srv)){
        if(srv.response.success){
            ROS_INFO("Successfully set the SAP data record service %s", msg.c_str());
            return true;
        }else{
            return false;
            ROS_WARN("Failed to set  SAP data record service %s", msg.c_str());
        }
    }
    ROS_WARN("No response from the SAP data recorder");
    return false;
}


void joy_ur::joy_Callback(const sensor_msgs::Joy::ConstPtr &msg){
    if(8 != msg->axes.size() || 11 != msg->buttons.size()){
        ROS_WARN("Wrong joy stick mode");
        std::fill(cartesian_move_vel.begin(),cartesian_move_vel.end(), 0.0);
        return;
    }

    if(DOWN(RS_B_BIT)){
        if (go_home()){
            ROS_INFO("Move to init pose done");
        }else{
            ROS_WARN("Failed to Move to init pose");
        }
    }

    if(DOWN(LS_B_BIT)){
        reset_current_as_origin();
    }

    if(1 == msg->buttons.at(A_BIT) && 1 == msg->buttons.at(B_BIT) && 1 == msg->buttons.at(X_BIT) && 1 == msg->buttons.at(Y_BIT)){
        // STOP ROBOT
        std::fill(cartesian_move_vel.begin(),cartesian_move_vel.end(), 0.0);
        ROS_WARN("Activate emergency stop");
        ros::Duration(2).sleep();
        return;
    }

    if(DOWN(START_BIT)){
        // START JOY CONTROLL
        std_srvs::Trigger srv;
        bool flag = true;
        flag = flag & zero_tf_client.call(srv);
        if(real_robot){
            flag = flag & stop_client.call(srv);
            ros::Duration(0.5).sleep();
            flag = flag & play_client.call(srv);
            ros::Duration(0.5).sleep();
            if(!flag) ROS_WARN("No response from UR driver");
        }
        if(set_joy_control(true)) joy_control_on = true;
        std::fill(cartesian_move_vel.begin(),cartesian_move_vel.end(), 0.0);
    }

    if(DOWN(BACK_BIT)){
        // STOP JOY CONTROLL
        if(set_joy_control(false)) joy_control_on = false;
        std::fill(cartesian_move_vel.begin(),cartesian_move_vel.end(), 0.0);
    }

    // if(DOWN(X_BIT)){
    //     gripper_target_position_vel = -10;
    // }

    // if(DOWN(B_BIT)){
    //     gripper_target_position_vel = 10;
    // }

    // if(UP(X_BIT) || UP(B_BIT)){
    //     gripper_target_position_vel = 0;
    // }

    // if(UP_BUTTOM){

    // }

    if(DOWN_BUTTOM){
        // ROS_INFO("DOWN!");
        std_srvs::Trigger srv;
        if(!clear_TOL_equilibrium_offset_client.call(srv)){
            ROS_WARN("Clear equilibrium offset failed: No responses from Admittance controller");
        }else{
            ROS_INFO("Clear equilibrium offset");
        }
    }

    if(LEFT_BUTTOM){
        set_cur_as_EEF_ref_pose();
    }

    if(RIGHT_BUTTOM){
        static bool on_off = true;
        activate_sap_data_record(on_off);
        on_off = !on_off;
    }

    if(DOWN(A_BIT)){
        ROS_INFO("Slow mode");
        mode_gain = 0.25;
    }else if(UP(A_BIT)){
        ROS_INFO("Fast mode");
        mode_gain = 1.0;
    }

    cartesian_move_vel.at(0) = mode_gain * fsat(X_GAIN * msg->axes.at(LS_H_BIT));
    cartesian_move_vel.at(1) = mode_gain * fsat(Y_GAIN * msg->axes.at(LS_V_BIT));
    cartesian_move_vel.at(2) = mode_gain * fsat(Z_GAIN * (msg->axes.at(RT_BIT) - msg->axes.at(LT_BIT)));
    cartesian_move_vel.at(3) = 2 * mode_gain * fsat(RX_GAIN * msg->axes.at(RS_V_BIT), -0.2, 0.2);
    cartesian_move_vel.at(4) = 2 * mode_gain * fsat(RY_GAIN * msg->axes.at(RS_H_BIT), -0.2, 0.2);
    cartesian_move_vel.at(5) = (TCP_based? 1:-1) * 2 * mode_gain * fsat(RZ_GAIN * (msg->buttons.at(LB_BIT) - msg->buttons.at(RB_BIT)), -0.3, 0.3);

    if(0 == msg->buttons.at(Y_BIT) && 1 == joy_last_state.buttons.at(Y_BIT)){
        TCP_based = !TCP_based;
        ROS_INFO("Move based on the %s frame.", TCP_based? "TCP":"Base_link");
    }

    // if(0 == msg->buttons.at(B_BIT) && 1 == joy_last_state.buttons.at(B_BIT)){
    //     gripper_open();
    //     ROS_INFO("Send gripper open cmd");
    // }else if(0 == msg->buttons.at(X_BIT) && 1 == joy_last_state.buttons.at(X_BIT)){
    //     gripper_close();
    //     ROS_INFO("Send gripper close cmd");
    // }else if(0 == msg->buttons.at(Y_BIT) && 1 == joy_last_state.buttons.at(Y_BIT)){
    //     gripper_hold();
    //     ROS_INFO("Send gripper hold cmd");
    // }

    joy_last_state = *msg;
}

void joy_ur::cur_TCP_pose_sub_Callback(const geometry_msgs::Pose::ConstPtr &msg){
    // geometry_msgs::Pose pose = *msg;
    Eigen::Vector3d trans(msg->position.x, msg->position.y, msg->position.z);
    Eigen::Quaterniond rot (msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);

    cur_TCP_pose.setIdentity();
    cur_TCP_pose.translate(trans);
    cur_TCP_pose.rotate(rot);

    TCP_valid = true;
}


double joy_ur::fsat(double value, double min, double max){
    if(value > max){
        return max;
    }
    if(value < min){
        return min;
    }
    return value;
}

int joy_ur::sat(int value, int min, int max){
    if(value > max){
        return max;
    }
    if(value < min){
        return min;
    }
    return value;
}

bool joy_ur::set_joy_control(bool on_off){
    std_srvs::SetBool velocity_controller_switch;
    if(on_off){
        if(switch_admittance_controller(true) && switch_follow_trajectory_online_controller(true)){
            return cartesian_started = true;
        }else{
            return cartesian_started = false;
        }
    }else{
        if(switch_follow_trajectory_online_controller(false) && switch_admittance_controller(false)){
            cartesian_started = false;
            return true;
        }else{
            return false;
        }
    }
}

bool joy_ur::switch_follow_trajectory_online_controller(bool on_off){
    std_srvs::SetBool switch_srv;
    switch_srv.request.data = on_off;

    std::string arm_selection_info =  "arm";

    if(follow_trajectory_online_switch_client.call(switch_srv)){
        if(switch_srv.response.success){
            ROS_INFO_STREAM("Successfully switch the cartesian controller " << arm_selection_info);
            return true;
        }else{
            ROS_WARN_STREAM("Failed to switch cartesian controller " << arm_selection_info);
            return false;
        }
    }else{
        ROS_WARN_STREAM("Failed to call cartesian controller switch server " << arm_selection_info);
        return false;
    }     
}

bool joy_ur::switch_admittance_controller(bool on_off){
    std_srvs::SetBool switch_srv;
    switch_srv.request.data = on_off;

    std::string arm_selection_info =  "arm";

    if(admittance_controller_switch_client.call(switch_srv)){
        if(switch_srv.response.success){
            ROS_INFO_STREAM("Successfully switch the velocity controller " << arm_selection_info);
            return true;
        }else{
            ROS_WARN_STREAM("Failed to switch velocity controller " << arm_selection_info);
            return false;
        }
    }else{
        ROS_WARN_STREAM("Failed to call velocity controller switch server " << arm_selection_info);
        return false;
    }  
}

void joy_ur::compute_target_pose(){
    static ros::Time last = ros::Time::now();
    double delta_t = (ros::Time::now() - last).toSec();
    std::vector<double> target_vel = cartesian_move_vel;

    std::vector<double> vel_filtered = low_pass_filter(target_vel);

    Eigen::Vector3d translation (delta_t*vel_filtered[0], delta_t*vel_filtered[1], delta_t*vel_filtered[2]);
    Eigen::Vector3d rotation_v  (delta_t*vel_filtered[3], delta_t*vel_filtered[4], delta_t*vel_filtered[5]);
    Eigen::AngleAxisd rotation_aa (rotation_v.norm(), rotation_v.normalized());
    Eigen::Isometry3d delta_transformation = Eigen::Isometry3d::Identity();
    delta_transformation.pretranslate(translation);
    delta_transformation.rotate(rotation_aa);
    if(TCP_based){
        target_pose = target_pose * delta_transformation;
    }else{
        target_pose = delta_transformation * target_pose;
    }

    tf::poseEigenToMsg(target_pose, target_pose_msg);

    last = ros::Time::now();
}

void joy_ur::pub_pose_msgs(){

    static size_t seq = 0;
    //  Creat data of Brodcaster
    geometry_msgs::TransformStamped tfs;
    //  parent fram
    
    tfs.header.frame_id = "base_link";
    tfs.header.stamp = ros::Time::now();
    tfs.header.seq = seq;
    seq++;
    //  child fram
    tfs.child_frame_id = "target_pose_joy";
    
    tfs.transform.translation.x = target_pose_msg.position.x;
    tfs.transform.translation.y = target_pose_msg.position.y;
    tfs.transform.translation.z = target_pose_msg.position.z;
    //  |---------set rotation------------------------------ 
    tfs.transform.rotation.x = target_pose_msg.orientation.x;
    tfs.transform.rotation.y = target_pose_msg.orientation.y;
    tfs.transform.rotation.z = target_pose_msg.orientation.z;
    tfs.transform.rotation.w = target_pose_msg.orientation.w;
    
    
    //tf
    broadcaster.sendTransform(tfs);

    //target pose
    target_pose_pub.publish(target_pose_msg);

    // hand target pose
    // std_msgs::Int32 hand_target_position_msg;
    // hand_target_position_msg.data = gripper_target_position;
    // gripper_target_position_pub.publish(hand_target_position_msg);
}

std::vector<double> joy_ur::low_pass_filter(std::vector<double> input_vec) {
    
    static std::vector<Eigen::Matrix<double,6,1>> filter_elements;
    Eigen::Matrix<double,6,1> temp_v6;
    temp_v6 << input_vec.at(0),input_vec.at(1),input_vec.at(2),input_vec.at(3),input_vec.at(4),input_vec.at(5);
    // Adding new element to filter vector    
    filter_elements.push_back(temp_v6);

    // Keep only the last 100 values of the vector
    while (ros::ok() && filter_elements.size() > 5) {filter_elements.erase(filter_elements.begin());}

    // Median Filter (media = sum / N_elements)
    Eigen::Matrix<double,6,1> sum, median;
    for (unsigned int i = 0; i < filter_elements.size(); i++) {sum += filter_elements[i];}
    median = sum / filter_elements.size();
    
    return std::vector<double>{median(0),median(1),median(2),median(3),median(4),median(5)};
}


//-------- SPIN ------- //
void joy_ur::spinner(){
    ros::Rate loop_rate(20);

    while(ros::ok()){
        ros::spinOnce();
        compute_target_pose();
        pub_pose_msgs();
        loop_rate.sleep();
    }
}