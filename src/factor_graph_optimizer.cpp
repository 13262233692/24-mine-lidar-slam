#include "factor_graph_optimizer.h"
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <iostream>
#include <algorithm>

using namespace gtsam;

namespace mine_slam {

FactorGraphOptimizer::FactorGraphOptimizer()
    : use_isam2_(false)
    , isam2_initialized_(false)
    , max_iterations_(100)
    , optimizer_type_("levenberg_marquardt")
    , initial_error_(0.0)
    , final_error_(0.0) {
    isam2_params_.relinearizeSkip = 10;
    isam2_params_.relinearizeThreshold = 0.1;
    isam2_params_.factorization = ISAM2Params::CHOLESKY;
    isam2_params_.enablePartialRelinearizationCheck = true;
}

void FactorGraphOptimizer::setMaxIterations(int iterations) {
    max_iterations_ = iterations;
}

void FactorGraphOptimizer::setOptimizerType(const std::string& type) {
    optimizer_type_ = type;
}

void FactorGraphOptimizer::enableISAM2(bool enable) {
    use_isam2_ = enable;
    if (use_isam2_ && !isam2_) {
        isam2_ = std::make_unique<ISAM2>(isam2_params_);
        isam2_initialized_ = true;
        std::cout << "[Optimizer] ISAM2 incremental smoother enabled" << std::endl;
    }
}

void FactorGraphOptimizer::setISAM2Params(int relinearize_skip,
                                           double relinearize_threshold) {
    isam2_params_.relinearizeSkip = relinearize_skip;
    isam2_params_.relinearizeThreshold = relinearize_threshold;
}

void FactorGraphOptimizer::reset() {
    graph_ = NonlinearFactorGraph();
    initial_estimates_.clear();
    optimized_estimates_.clear();
    pose_ids_.clear();
    initial_error_ = 0.0;
    final_error_ = 0.0;
    if (isam2_) {
        isam2_ = std::make_unique<ISAM2>(isam2_params_);
        isam2_initialized_ = true;
    }
}

void FactorGraphOptimizer::addPriorFactor(size_t pose_id, const Pose3D& pose,
                                          double noise_x, double noise_y, double noise_z,
                                          double noise_rot) {
    Pose3 gtsam_pose = toGtsamPose(pose);
    auto noise = createNoiseModel(noise_x, noise_y, noise_z, noise_rot, noise_rot, noise_rot);
    graph_.add(PriorFactor<Pose3>(pose_id, gtsam_pose, noise));

    if (std::find(pose_ids_.begin(), pose_ids_.end(), pose_id) == pose_ids_.end()) {
        pose_ids_.push_back(pose_id);
    }
}

void FactorGraphOptimizer::addOdometryFactor(size_t from_id, size_t to_id,
                                             const Pose3D& relative_pose,
                                             double noise_x, double noise_y, double noise_z,
                                             double noise_rot) {
    Pose3 rel_pose = toGtsamPose(relative_pose);
    auto noise = createNoiseModel(noise_x, noise_y, noise_z, noise_rot, noise_rot, noise_rot);
    graph_.add(BetweenFactor<Pose3>(from_id, to_id, rel_pose, noise));

    for (size_t id : {from_id, to_id}) {
        if (std::find(pose_ids_.begin(), pose_ids_.end(), id) == pose_ids_.end()) {
            pose_ids_.push_back(id);
        }
    }
}

void FactorGraphOptimizer::addLoopClosureFactor(size_t from_id, size_t to_id,
                                                const Pose3D& relative_pose,
                                                double noise_x, double noise_y, double noise_z,
                                                double noise_rot) {
    Pose3 rel_pose = toGtsamPose(relative_pose);
    auto noise = createNoiseModel(noise_x, noise_y, noise_z, noise_rot, noise_rot, noise_rot);
    graph_.add(BetweenFactor<Pose3>(from_id, to_id, rel_pose, noise));

    for (size_t id : {from_id, to_id}) {
        if (std::find(pose_ids_.begin(), pose_ids_.end(), id) == pose_ids_.end()) {
            pose_ids_.push_back(id);
        }
    }
}

void FactorGraphOptimizer::addPoseEstimate(size_t pose_id, const Pose3D& pose) {
    Pose3 gtsam_pose = toGtsamPose(pose);
    if (!initial_estimates_.exists(pose_id)) {
        initial_estimates_.insert(pose_id, gtsam_pose);
    }

    if (std::find(pose_ids_.begin(), pose_ids_.end(), pose_id) == pose_ids_.end()) {
        pose_ids_.push_back(pose_id);
    }
}

void FactorGraphOptimizer::addVelocityEstimate(size_t key, const Eigen::Vector3d& velocity) {
    gtsam::Key v_key = velKey(key);
    if (!initial_estimates_.exists(v_key)) {
        initial_estimates_.insert(v_key, Vector3(velocity));
    }
}

void FactorGraphOptimizer::addIMUBiasEstimate(size_t key, const Eigen::Vector3d& accel_bias,
                                               const Eigen::Vector3d& gyro_bias) {
    gtsam::Key b_key = biasKey(key);
    if (!initial_estimates_.exists(b_key)) {
        initial_estimates_.insert(b_key,
            imuBias::ConstantBias(accel_bias, gyro_bias));
    }
}

void FactorGraphOptimizer::addIMUPriorFactor(size_t key, const Eigen::Vector3d& velocity,
                                              double velocity_noise,
                                              const Eigen::Vector3d& accel_bias,
                                              const Eigen::Vector3d& gyro_bias,
                                              double bias_noise) {
    gtsam::Key v_key = velKey(key);
    gtsam::Key b_key = biasKey(key);

    auto vel_noise_model = noiseModel::Diagonal::Sigmas(
        (Vector3() << velocity_noise, velocity_noise, velocity_noise).finished());

    graph_.add(PriorFactor<Vector3>(v_key, Vector3(velocity), vel_noise_model));

    auto bias_noise_model = noiseModel::Diagonal::Sigmas(
        (Vector6() << bias_noise, bias_noise, bias_noise,
                      bias_noise, bias_noise, bias_noise).finished());

    graph_.add(PriorFactor<imuBias::ConstantBias>(
        b_key, imuBias::ConstantBias(accel_bias, gyro_bias), bias_noise_model));
}

void FactorGraphOptimizer::addIMUFactor(size_t from_key, size_t to_key,
                                         const IMUPreintegratedResult& preint,
                                         const IMUPreintegrationParams& imu_params) {
    auto preint_params = boost::make_shared<PreintegrationParams>(imu_params.gravity);
    preint_params->accelerometerCovariance =
        pow(imu_params.accel_noise_sigma, 2) * Eigen::Matrix3d::Identity();
    preint_params->gyroscopeCovariance =
        pow(imu_params.gyro_noise_sigma, 2) * Eigen::Matrix3d::Identity();
    preint_params->integrationCovariance =
        1e-4 * Eigen::Matrix3d::Identity();

    PreintegratedImuMeasurements pim(
        preint_params,
        imuBias::ConstantBias(imu_params.init_accel_bias, imu_params.init_gyro_bias));

    Eigen::Vector3d mean_accel = Eigen::Vector3d::Zero();
    if (preint.delta_t > 1e-9) {
        mean_accel = (preint.delta_v - Eigen::Vector3d(0, 0, 0)) / preint.delta_t;
    }

    double dt_per_step = 0.005;
    int n_steps = std::max(1, static_cast<int>(preint.delta_t / dt_per_step));
    double dt_actual = preint.delta_t / static_cast<double>(n_steps);

    Eigen::Vector3d gyro_step = Eigen::Vector3d::Zero();
    Eigen::Vector3d delta_rot_vec = Eigen::AngleAxisd(preint.delta_q).angle() *
                                     Eigen::AngleAxisd(preint.delta_q).axis();
    if (n_steps > 0) {
        gyro_step = delta_rot_vec / preint.delta_t;
    }

    for (int i = 0; i < n_steps; ++i) {
        pim.integrateMeasurement(mean_accel, gyro_step, dt_actual);
    }

    gtsam::Key v_i = velKey(from_key);
    gtsam::Key v_j = velKey(to_key);
    gtsam::Key b_i = biasKey(from_key);
    gtsam::Key b_j = biasKey(to_key);

    CombinedImuFactor imu_factor(from_key, v_i, to_key, v_j, b_i, b_j, pim);

    graph_.add(imu_factor);

    for (size_t id : {from_key, to_key}) {
        if (std::find(pose_ids_.begin(), pose_ids_.end(), id) == pose_ids_.end()) {
            pose_ids_.push_back(id);
        }
    }
}

void FactorGraphOptimizer::addBiasRandomWalkFactor(size_t from_key, size_t to_key,
                                                    double accel_bias_rw_sigma,
                                                    double gyro_bias_rw_sigma) {
    gtsam::Key b_i = biasKey(from_key);
    gtsam::Key b_j = biasKey(to_key);

    auto bias_noise = noiseModel::Diagonal::Sigmas(
        (Vector6() << accel_bias_rw_sigma, accel_bias_rw_sigma, accel_bias_rw_sigma,
                      gyro_bias_rw_sigma, gyro_bias_rw_sigma, gyro_bias_rw_sigma).finished());

    graph_.add(BetweenFactor<imuBias::ConstantBias>(
        b_i, b_j, imuBias::ConstantBias(), bias_noise));
}

bool FactorGraphOptimizer::optimize() {
    if (initial_estimates_.empty()) {
        std::cerr << "[Optimizer] No initial estimates provided" << std::endl;
        return false;
    }

    try {
        initial_error_ = graph_.error(initial_estimates_);

        if (use_isam2_ && isam2_) {
            isam2_->update(graph_, initial_estimates_);
            optimized_estimates_ = isam2_->calculateEstimate();

            for (int i = 0; i < max_iterations_; ++i) {
                isam2_->update();
            }
            optimized_estimates_ = isam2_->calculateEstimate();
        } else if (optimizer_type_ == "levenberg_marquardt") {
            LevenbergMarquardtParams params;
            params.setMaxIterations(max_iterations_);
            params.setVerbosity("SILENT");
            LevenbergMarquardtOptimizer optimizer(graph_, initial_estimates_, params);
            optimized_estimates_ = optimizer.optimize();
        } else if (optimizer_type_ == "gauss_newton") {
            GaussNewtonParams params;
            params.setMaxIterations(max_iterations_);
            params.setVerbosity("SILENT");
            GaussNewtonOptimizer optimizer(graph_, initial_estimates_, params);
            optimized_estimates_ = optimizer.optimize();
        } else if (optimizer_type_ == "dogleg") {
            DoglegParams params;
            params.setMaxIterations(max_iterations_);
            params.setVerbosity("SILENT");
            DoglegOptimizer optimizer(graph_, initial_estimates_, params);
            optimized_estimates_ = optimizer.optimize();
        } else {
            std::cerr << "[Optimizer] Unknown optimizer type: " << optimizer_type_ << std::endl;
            return false;
        }

        final_error_ = graph_.error(optimized_estimates_);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Optimizer] Optimization failed: " << e.what() << std::endl;
        return false;
    }
}

bool FactorGraphOptimizer::incrementalUpdate() {
    if (!use_isam2_ || !isam2_) {
        return optimize();
    }

    try {
        if (!initial_estimates_.empty()) {
            isam2_->update(graph_, initial_estimates_);
            graph_ = NonlinearFactorGraph();
            initial_estimates_.clear();
        } else {
            isam2_->update();
        }

        optimized_estimates_ = isam2_->calculateEstimate();
        final_error_ = isam2_->calculateEstimate().size() > 0 ? 0.0 : 0.0;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Optimizer] ISAM2 update failed: " << e.what() << std::endl;
        return false;
    }
}

Pose3D FactorGraphOptimizer::getOptimizedPose(size_t pose_id) const {
    if (optimized_estimates_.exists(pose_id)) {
        Pose3 gtsam_pose = optimized_estimates_.at<Pose3>(pose_id);
        return fromGtsamPose(gtsam_pose);
    }

    if (initial_estimates_.exists(pose_id)) {
        Pose3 gtsam_pose = initial_estimates_.at<Pose3>(pose_id);
        return fromGtsamPose(gtsam_pose);
    }

    return Pose3D();
}

Eigen::Vector3d FactorGraphOptimizer::getOptimizedVelocity(size_t key) const {
    gtsam::Key v_key = velKey(key);
    if (optimized_estimates_.exists(v_key)) {
        return optimized_estimates_.at<Vector3>(v_key);
    }
    if (initial_estimates_.exists(v_key)) {
        return initial_estimates_.at<Vector3>(v_key);
    }
    return Eigen::Vector3d::Zero();
}

imuBias::ConstantBias FactorGraphOptimizer::getOptimizedIMUBias(size_t key) const {
    gtsam::Key b_key = biasKey(key);
    if (optimized_estimates_.exists(b_key)) {
        return optimized_estimates_.at<imuBias::ConstantBias>(b_key);
    }
    if (initial_estimates_.exists(b_key)) {
        return initial_estimates_.at<imuBias::ConstantBias>(b_key);
    }
    return imuBias::ConstantBias();
}

NavState FactorGraphOptimizer::getOptimizedNavState(size_t key) const {
    NavState ns;
    Pose3D pose = getOptimizedPose(key);
    ns.position = pose.translation;
    ns.rotation = pose.rotation;
    ns.velocity = getOptimizedVelocity(key);

    auto bias = getOptimizedIMUBias(key);
    ns.accel_bias = bias.accelerometer();
    ns.gyro_bias = bias.gyroscope();
    return ns;
}

std::vector<Pose3D> FactorGraphOptimizer::getAllOptimizedPoses() const {
    std::vector<Pose3D> poses;
    poses.reserve(pose_ids_.size());

    std::vector<size_t> sorted_ids = pose_ids_;
    std::sort(sorted_ids.begin(), sorted_ids.end());

    for (size_t id : sorted_ids) {
        poses.push_back(getOptimizedPose(id));
    }
    return poses;
}

size_t FactorGraphOptimizer::getFactorCount() const {
    return graph_.size();
}

gtsam::Key FactorGraphOptimizer::velKey(size_t pose_id) const {
    return Symbol('v', pose_id);
}

gtsam::Key FactorGraphOptimizer::biasKey(size_t pose_id) const {
    return Symbol('b', pose_id);
}

Pose3 FactorGraphOptimizer::toGtsamPose(const Pose3D& pose) {
    Point3 point(pose.translation.x(), pose.translation.y(), pose.translation.z());
    Rot3 rot(pose.rotation.toRotationMatrix());
    return Pose3(rot, point);
}

Pose3D FactorGraphOptimizer::fromGtsamPose(const Pose3& gtsam_pose) {
    Pose3D pose;
    pose.translation.x() = gtsam_pose.translation().x();
    pose.translation.y() = gtsam_pose.translation().y();
    pose.translation.z() = gtsam_pose.translation().z();
    pose.rotation = Eigen::Quaterniond(gtsam_pose.rotation().matrix());
    pose.rotation.normalize();
    return pose;
}

noiseModel::Diagonal::shared_ptr FactorGraphOptimizer::createNoiseModel(
    double nx, double ny, double nz, double nrx, double nry, double nrz) {
    Vector6 sigmas;
    sigmas << nrx, nry, nrz, nx, ny, nz;
    return noiseModel::Diagonal::Sigmas(sigmas);
}

} // namespace mine_slam
