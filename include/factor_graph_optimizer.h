#pragma once

#include "common_types.h"
#include "imu_preintegration.h"
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/base/Vector.h>
#include <memory>
#include <vector>
#include <string>

namespace mine_slam {

class FactorGraphOptimizer {
public:
    FactorGraphOptimizer();
    ~FactorGraphOptimizer() = default;

    void setMaxIterations(int iterations);
    void setOptimizerType(const std::string& type);

    void enableISAM2(bool enable);
    void setISAM2Params(int relinearize_skip = 10,
                        double relinearize_threshold = 0.1);

    void reset();

    void addPriorFactor(size_t pose_id, const Pose3D& pose,
                        double noise_x, double noise_y, double noise_z,
                        double noise_rot);

    void addOdometryFactor(size_t from_id, size_t to_id,
                           const Pose3D& relative_pose,
                           double noise_x, double noise_y, double noise_z,
                           double noise_rot);

    void addLoopClosureFactor(size_t from_id, size_t to_id,
                              const Pose3D& relative_pose,
                              double noise_x, double noise_y, double noise_z,
                              double noise_rot);

    void addPoseEstimate(size_t pose_id, const Pose3D& pose);

    void addVelocityEstimate(size_t key, const Eigen::Vector3d& velocity);

    void addIMUBiasEstimate(size_t key, const Eigen::Vector3d& accel_bias,
                            const Eigen::Vector3d& gyro_bias);

    void addIMUPriorFactor(size_t key, const Eigen::Vector3d& velocity,
                           double velocity_noise,
                           const Eigen::Vector3d& accel_bias,
                           const Eigen::Vector3d& gyro_bias,
                           double bias_noise);

    void addIMUFactor(size_t from_key, size_t to_key,
                      const IMUPreintegratedResult& preint,
                      const IMUPreintegrationParams& imu_params);

    void addBiasRandomWalkFactor(size_t from_key, size_t to_key,
                                 double accel_bias_rw_sigma,
                                 double gyro_bias_rw_sigma);

    bool optimize();

    bool incrementalUpdate();

    Pose3D getOptimizedPose(size_t pose_id) const;
    Eigen::Vector3d getOptimizedVelocity(size_t key) const;
    gtsam::imuBias::ConstantBias getOptimizedIMUBias(size_t key) const;
    NavState getOptimizedNavState(size_t key) const;
    std::vector<Pose3D> getAllOptimizedPoses() const;

    size_t getPoseCount() const { return pose_ids_.size(); }
    size_t getFactorCount() const;

    double getInitialError() const { return initial_error_; }
    double getFinalError() const { return final_error_; }

private:
    gtsam::Key velKey(size_t pose_id) const;
    gtsam::Key biasKey(size_t pose_id) const;

    static gtsam::Pose3 toGtsamPose(const Pose3D& pose);
    static Pose3D fromGtsamPose(const gtsam::Pose3& gtsam_pose);
    static gtsam::noiseModel::Diagonal::shared_ptr createNoiseModel(
        double nx, double ny, double nz, double nrx, double nry, double nrz);

    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values initial_estimates_;
    gtsam::Values optimized_estimates_;

    std::unique_ptr<gtsam::ISAM2> isam2_;
    gtsam::ISAM2Params isam2_params_;
    bool use_isam2_;
    bool isam2_initialized_;

    std::vector<size_t> pose_ids_;

    int max_iterations_;
    std::string optimizer_type_;

    double initial_error_;
    double final_error_;
};

using FactorGraphOptimizerPtr = std::shared_ptr<FactorGraphOptimizer>;

} // namespace mine_slam
