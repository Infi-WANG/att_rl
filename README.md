gazebo的顶层启动文件（非conda环境）：
roslaunch robot_gazebo gazebo_admittance_top.launch
需要配有手柄进行遥控操作

数据记录程序：
robot_driver/scripts/data_recorder.py

记录后数据位置：
robot_driver/data

由于数据已经记录完成，调试SAP无需开启上述文件，只需要执行SAP顶层文件（需要在conda环境下）：
robot_driver/scripts/SAP.py