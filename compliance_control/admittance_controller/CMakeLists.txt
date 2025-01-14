cmake_minimum_required(VERSION 3.0.2)
project(admittance_controller)

add_compile_options(-std=c++14)

find_package(Eigen3 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  actionlib
  actionlib_msgs
  message_generation

  std_msgs
  geometry_msgs
  trajectory_msgs
  sensor_msgs
  control_msgs
  std_srvs
  controller_manager_msgs
  
  moveit_core
  moveit_ros_planning_interface

  self_defined_msgs
  eigen_conversions
)

################################################
## Declare ROS messages, services and actions ##
################################################

## moved to package self_defined_msgs

# add_message_files(
#   FILES
#   joint_trajectory.msg
#   parameter_msg.msg
# )

# add_service_files(
#   FILES
#   parameter_srv.srv
#   set_equilibrium_pose_srv.srv
# )

# add_action_files(
#   DIRECTORY action
#   FILES compliance_action.action
# )

# generate_messages(
#   DEPENDENCIES
#   actionlib_msgs
#   std_msgs
#   sensor_msgs
#   moveit_msgs
#   ${PROJECT_NAME}
# )

###################################
## catkin specific configuration ##
###################################

catkin_package(
  INCLUDE_DIRS 
  include
  lib
  CATKIN_DEPENDS
  # message_runtime
  # actionlib_msgs
)

###########
## Build ##
###########

include_directories(
  include
  lib
  ${EIGEN3_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)


add_executable(admittance_controller_Node
	src/admittance_controller/admittance_controller_Node.cpp 
	src/admittance_controller/admittance_controller.cpp
  src/admittance_controller/pchip.c
  src/admittance_controller/singularity_avoidance.cpp
  )

target_link_libraries(admittance_controller_Node ${catkin_LIBRARIES})
add_dependencies(admittance_controller_Node ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})
