#ifndef SINGULARITY_AVOIDANCE_H
#define SINGULARITY_AVOIDANCE_H

#include <ros/ros.h>
#include <iostream>

#include <Eigen/Eigen>

using namespace Eigen;

typedef Eigen::Matrix<double,6,1> Vector6d;
typedef Eigen::Matrix<double,6,6> Matrix6d;
class singularity_avoidance{

    public:
        singularity_avoidance(double minSingular_value_boundary,double damping_coefficient);
                              
        double singuar_value_decomposition(Eigen::MatrixXd jacobian);

        Eigen::VectorXd sort_vector(const VectorXd&vec);

        Vector6d computer_jointpoint(Eigen::MatrixXd jacobian,double singuar_value,const Vector6d& posion_current,
                                                            const Vector6d& posion_next,
                                                            const Vector6d& velocity_upper_limit,
                                                           const Vector6d& velocity_lower_limit);

        Vector6d limit_q_velocity(const Vector6d &velocity_upper_limit,const Vector6d &velocity_lower_limit,const Vector6d &q_velocity);
        virtual ~singularity_avoidance(void){};

    public:
        double minSingular_value_boundary;
        double damping_coefficient;
        double variable_damping_coefficient;
        
        Eigen::MatrixXd Singular_Robustness_Inverse;

};


#endif