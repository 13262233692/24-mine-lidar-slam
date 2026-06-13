#include "imu_preintegration.h"
#include <iostream>
#include <cmath>

namespace mine_slam {

IMUPreintegrator::IMUPreintegrator(const IMUPreintegrationParams& params)
    : params_(params) {
    reset();
}

void IMUPreintegrator::reset() {
    delta_p_.setZero();
    delta_v_.setZero();
    delta_q_.setIdentity();
    covariance_.setZero();
    jacobian_bias_.setZero();
    accel_bias_ = params_.init_accel_bias;
    gyro_bias_ = params_.init_gyro_bias;
    delta_t_ = 0.0;
    num_measurements_ = 0;
}

void IMUPreintegrator::integrateMeasurement(const IMUMeasurement& imu, double dt) {
    if (dt <= 0.0) return;

    Eigen::Vector3d accel_corrected = imu.accel - accel_bias_;
    Eigen::Vector3d gyro_corrected = imu.gyro - gyro_bias_;

    Eigen::Quaterniond delta_q_step;
    double angle = gyro_corrected.norm();
    if (angle > 1e-10) {
        delta_q_step = Eigen::Quaterniond(
            Eigen::AngleAxisd(angle * dt, gyro_corrected / angle));
    } else {
        delta_q_step = Eigen::Quaterniond::Identity();
    }
    delta_q_step.normalize();

    Eigen::Quaterniond delta_q_new = (delta_q_ * delta_q_step).normalized();

    Eigen::Vector3d accel_body = 0.5 * (delta_q_.toRotationMatrix() * accel_corrected +
                                          delta_q_new.toRotationMatrix() * accel_corrected);

    Eigen::Vector3d new_delta_p = delta_p_ + delta_v_ * dt + 0.5 * accel_body * dt * dt;
    Eigen::Vector3d new_delta_v = delta_v_ + accel_body * dt;

    updateCovariance(dt, accel_corrected, gyro_corrected);
    updateJacobianBias(dt, delta_q_new, accel_corrected);

    delta_p_ = new_delta_p;
    delta_v_ = new_delta_v;
    delta_q_ = delta_q_new;
    delta_t_ += dt;
    num_measurements_++;
}

void IMUPreintegrator::updateCovariance(double dt,
                                          const Eigen::Vector3d& accel_corrected,
                                          const Eigen::Vector3d& gyro_corrected) {
    Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();

    Eigen::Matrix3d R_k = delta_q_.toRotationMatrix();

    F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;

    Eigen::Matrix3d acc_skew;
    acc_skew << 0.0, -accel_corrected.z(), accel_corrected.y(),
                accel_corrected.z(), 0.0, -accel_corrected.x(),
                -accel_corrected.y(), accel_corrected.x(), 0.0;

    F.block<3, 3>(3, 6) = -R_k * dt;
    F.block<3, 3>(0, 6) = -0.5 * R_k * acc_skew * dt * dt;

    Eigen::Matrix<double, 9, 6> G = Eigen::Matrix<double, 9, 6>::Zero();
    G.block<3, 3>(0, 0) = 0.5 * R_k * dt * dt;
    G.block<3, 3>(3, 0) = R_k * dt;
    G.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity() * dt;

    Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
    Q.block<3, 3>(0, 0) = (params_.accel_noise_sigma * params_.accel_noise_sigma) *
                            Eigen::Matrix3d::Identity();
    Q.block<3, 3>(3, 3) = (params_.gyro_noise_sigma * params_.gyro_noise_sigma) *
                            Eigen::Matrix3d::Identity();

    covariance_ = F * covariance_ * F.transpose() + G * Q * G.transpose();
}

void IMUPreintegrator::updateJacobianBias(double dt,
                                            const Eigen::Quaterniond& delta_q_new,
                                            const Eigen::Vector3d& accel_corrected) {
    Eigen::Matrix<double, 9, 6> J_new = Eigen::Matrix<double, 9, 6>::Zero();

    Eigen::Matrix3d R_new = delta_q_new.toRotationMatrix();

    J_new.block<3, 3>(0, 0) = jacobian_bias_.block<3, 3>(0, 0) +
                                0.5 * R_new * dt * dt * (-Eigen::Matrix3d::Identity());
    J_new.block<3, 3>(0, 3) = jacobian_bias_.block<3, 3>(0, 3) +
                                jacobian_bias_.block<3, 3>(3, 3) * (-dt);

    J_new.block<3, 3>(3, 0) = jacobian_bias_.block<3, 3>(3, 0) +
                                R_new * dt * (-Eigen::Matrix3d::Identity());

    J_new.block<3, 3>(6, 3) = jacobian_bias_.block<3, 3>(6, 3) +
                                Eigen::Matrix3d::Identity() * (-dt);

    J_new.block<3, 3>(3, 3) = jacobian_bias_.block<3, 3>(3, 3);
    J_new.block<3, 3>(0, 0) += jacobian_bias_.block<3, 3>(3, 0) * dt;

    jacobian_bias_ = J_new;
}

IMUPreintegratedResult IMUPreintegrator::compute() const {
    IMUPreintegratedResult result;
    result.delta_p = delta_p_;
    result.delta_v = delta_v_;
    result.delta_q = delta_q_;
    result.covariance = covariance_;
    result.jacobian_bias = jacobian_bias_;
    result.delta_t = delta_t_;
    result.num_imu_measurements = num_measurements_;
    return result;
}

NavState IMUPreintegrator::predict(const NavState& state_i) const {
    NavState state_j;

    state_j.rotation = (state_i.rotation * delta_q_).normalized();

    Eigen::Vector3d dp_world = state_i.rotation * delta_p_;
    state_j.position = state_i.position + state_i.velocity * delta_t_ + dp_world;

    Eigen::Vector3d dv_world = state_i.rotation * delta_v_;
    state_j.velocity = state_i.velocity + params_.gravity * delta_t_ + dv_world;

    state_j.accel_bias = state_i.accel_bias;
    state_j.gyro_bias = state_i.gyro_bias;

    return state_j;
}

Eigen::Matrix<double, 9, 1> IMUPreintegrator::computeResidual(
    const NavState& state_i,
    const NavState& state_j) const {

    Eigen::Matrix<double, 9, 1> residual;

    Eigen::Quaterniond dq_ij = (state_i.rotation.inverse() * state_j.rotation).normalized();
    Eigen::Quaterniond err_q = (delta_q_.inverse() * dq_ij).normalized();

    residual.segment<3>(0) = state_i.rotation.inverse() *
        (state_j.position - state_i.position - state_i.velocity * delta_t_ -
         0.5 * params_.gravity * delta_t_ * delta_t_) - delta_p_;

    residual.segment<3>(3) = state_i.rotation.inverse() *
        (state_j.velocity - state_i.velocity - params_.gravity * delta_t_) - delta_v_;

    Eigen::Vector3d err_rot = Eigen::AngleAxisd(err_q).angle() *
                              Eigen::AngleAxisd(err_q).axis();
    residual.segment<3>(6) = err_rot;

    return residual;
}

} // namespace mine_slam
