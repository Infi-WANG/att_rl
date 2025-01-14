# 实验

修改文件:
src/peg_in_hole_RL_/admitance_ur/compliance_control/admittance_controller/launch/admittance_real_top.launch
```xml
    <!-- 正式运行时 -->
  <arg name="load_ur_robot_driver"       default="true"/>
  <arg name="load_robot_model"      		  default="true"/>
    <!-- 单步调试时 -->
  <arg name="load_ur_robot_driver"       default="false"/>
  <arg name="load_robot_model"      		  default="false"/>  
```
调试时先运行UR驱动以及moveit包 及手动加载模型文件,不然会报错(系统bug):

```bash
roslaunch admittance_controller load_robot_description.launch 
```
launch文件:

```bash
    roslaunch admittance_controller admittance_real_top.launch
```

# 仿真
```bash
    roslaunch admittance_controller admittance_sim_top.launch
```

# 关键service

  /admittance_controller/set_equilibrium_point_service 设置平衡点
  /admittance_controller/admittance_controller_activation_service  开启柔顺性控制
  /admittance_controller/virtual_force_control_activation_service 开启虚拟力控制
  /admittance_controller/set_VF_service 设置虚拟力大小 VECTOR

# debug

仿真单步调试注意模型文件和vscode冲突问题,仿照实验设置修改.
可以使用rqt_controller_manager查看控制器情况.


