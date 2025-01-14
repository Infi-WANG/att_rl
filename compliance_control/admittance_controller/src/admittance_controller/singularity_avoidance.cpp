#include "admittance_controller/singularity_avoidance.h"

singularity_avoidance::singularity_avoidance(double minSingular_value_boundary,double damping_coefficient):
                                                   minSingular_value_boundary(minSingular_value_boundary),
                                                   damping_coefficient(damping_coefficient)
                                                   {
                                                       variable_damping_coefficient=0.0;
                                                   }
double singularity_avoidance::singuar_value_decomposition(Eigen::MatrixXd jacobian){

    double min_singularity;
    //Eigen::VectorXi ind;
    Eigen::VectorXd sort_value_vector;
    JacobiSVD<MatrixXd>svd(jacobian,ComputeFullU|ComputeFullV);
    Eigen::VectorXd singular_value_vector = svd.singularValues();
    sort_value_vector = sort_vector(singular_value_vector);
    min_singularity = sort_value_vector(5);
    return min_singularity;
}                                        

Eigen::VectorXd singularity_avoidance::sort_vector(const VectorXd&vec){
    Eigen::VectorXd sorted_vec;
    Eigen::VectorXi  ind = VectorXi::LinSpaced(vec.size(), 0, vec.size() - 1);
    auto rule = [vec](int i, int j)->bool {
		return vec(i) > vec(j);
	};//正则表达式，作为sort的谓词
    std::sort(ind.data(), ind.data() + ind.size(), rule);
	sorted_vec.resize(vec.size());
	for (int i = 0; i < vec.size(); i++) {
		sorted_vec(i) = vec(ind(i));
	}
    return sorted_vec;
}
Vector6d singularity_avoidance::computer_jointpoint(Eigen::MatrixXd jacobian,double singuar_value,const Vector6d& posion_current,const Vector6d& posion_next,const Vector6d& velocity_upper_limit,const Vector6d& velocity_lower_limit){
    Vector6d dx,dq,q;
    Eigen::MatrixXd geoJacob_k_inv;
    Matrix6d eye;
    double det_Jacobian_value;
    Eigen::MatrixXd Jacobian_transpose;
    Jacobian_transpose = jacobian.transpose();
    det_Jacobian_value = jacobian.determinant();
    eye.setIdentity();
    if(singuar_value<minSingular_value_boundary||det_Jacobian_value ==0){

        variable_damping_coefficient = damping_coefficient*(1-pow((singuar_value/minSingular_value_boundary),2));
        geoJacob_k_inv = (Jacobian_transpose*jacobian+pow(variable_damping_coefficient,2)*eye).inverse() * Jacobian_transpose;
    }
    else{

        geoJacob_k_inv=jacobian.inverse();
    }
    
    dq = geoJacob_k_inv*dx;
    dq = limit_q_velocity(velocity_upper_limit,velocity_lower_limit,dq);
    q += dq; 
    return q;
}
Vector6d singularity_avoidance::limit_q_velocity(const Vector6d &velocity_upper_limit,const Vector6d &velocity_lower_limit,const Vector6d &q_velocity){

    Vector6d q_dot_new;
    q_dot_new = q_velocity;
    for(int i = 1;i<=6;i++){
        if(q_dot_new(i)>velocity_upper_limit(i))
        {
            q_dot_new(i) = velocity_upper_limit(i);
        }
        else if (q_dot_new(i)<velocity_lower_limit(i))
        {
            q_dot_new(i) = velocity_lower_limit(i);
        }
    }
    return q_dot_new;
}