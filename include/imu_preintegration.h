#pragma once

#include "common_types.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <memory>

namespace mine_slam {

struct IMUPreintegrationParams {
    Eigen::Vector3d gravity = Eigen::Vector3d(0.0, 0.0, -9.81);

    double accel_noise_sigma = 0.01;
    double gyro_noise_sigma = 0.001;
    double accel_bias_rw_sigma = 0.0001;
    double gyro_bias_rw_sigma = 0.00001;

    Eigen::Vector3d init_accel_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d init_gyro_bias = Eigen::Vector3d::Zero();
};

struct IMUPreintegratedResult {
    Eigen::Vector3d delta_p = Eigen::Vector3d::Zero();
    Eigen::Vector3d delta_v = Eigen::Vector3d::Zero();
    Eigen::Quaterniond delta_q = Eigen::Quaterniond::Identity();

    Eigen::Matrix<double, 9, 9> covariance = Eigen::Matrix<double, 9, 9>::Zero();
    Eigen::Matrix<double, 9, 6> jacobian_bias = Eigen::Matrix<double, 9, 6>::Zero();

    double delta_t = 0.0;
    int num_imu_measurements = 0;
};

class IMUPreintegrator {
public:
    explicit IMUPreintegrator(const IMUPreintegrationParams& params = IMUPreintegrationParams());

    void reset();

    void integrateMeasurement(const IMUMeasurement& imu, double dt);

    IMUPreintegratedResult compute() const;

    NavState predict(const NavState& state_i) const;

    Eigen::Matrix<double, 9, 1> computeResidual(
        const NavState& state_i,
        const NavState& state_j) const;

    const IMUPreintegrationParams& params() const { return params_; }

    double deltaTime() const { return delta_t_; }

    bool empty() const { return delta_t_ < 1e-9; }

private:
    void updateCovariance(double dt,
                          const Eigen::Vector3d& accel_corrected,
                          const Eigen::Vector3d& gyro_corrected);
    void updateJacobianBias(double dt,
                            const Eigen::Quaterniond& delta_q_new,
                            const Eigen::Vector3d& accel_corrected);

    IMUPreintegrationParams params_;

    Eigen::Vector3d delta_p_;
    Eigen::Vector3d delta_v_;
    Eigen::Quaterniond delta_q_;

    Eigen::Matrix<double, 9, 9> covariance_;
    Eigen::Matrix<double, 9, 6> jacobian_bias_;

    Eigen::Vector3d accel_bias_;
    Eigen::Vector3d gyro_bias_;

    double delta_t_;
    int num_measurements_;
};

using IMUPreintegratorPtr = std::shared_ptr<IMUPreintegrator>;

} // namespace mine_slam
