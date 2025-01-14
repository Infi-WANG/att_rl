#include "joy_ur/joy_ur.h"


int main(int argc, char **argv) {

    ros::init(argc, argv, "joy_controller_Node");
    joy_ur joy_ur_node;

    joy_ur_node.spinner();

    return 0;
}
