#include "factor_graph_optimizer.h"
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <iostream>
#include <algorithm>

namespace mine_slam {

FactorGraphOptimizer::FactorGraphOptimizer()
    : max_iterations_(100)
    , optimizer_type_("levenberg_marquardt")
    , initial_error_(0.0)
    , final_error_(0.0) {}

void FactorGraphOptimizer::setMaxIterations(int iterations) {
    max_iterations_ = iterations;
}

void FactorGraphOptimizer::setOptimizerType(const std::string& type) {
    optimizer_type_ = type;
}

void FactorGraphOptimizer::reset() {
    graph_ = gtsam::NonlinearFactorGraph();
    initial_estimates_.clear();
    optimized_estimates_.clear();
    pose_ids_.clear();
    initial_error_ = 0.0;
    final_error_ = 0.0;
}

void FactorGraphOptimizer::addPriorFactor(size_t pose_id, const Pose3D& pose,
                                          double noise_x, double noise_y, double noise_z,
                                          double noise_rot) {
    gtsam::Pose3 gtsam_pose = toGtsamPose(pose);
    auto noise = createNoiseModel(noise_x, noise_y, noise_z, noise_rot, noise_rot, noise_rot);
    graph_.add(gtsam::PriorFactor<gtsam::Pose3>(pose_id, gtsam_pose, noise));

    if (std::find(pose_ids_.begin(), pose_ids_.end(), pose_id) == pose_ids_.end()) {
        pose_ids_.push_back(pose_id);
    }
}

void FactorGraphOptimizer::addOdometryFactor(size_t from_id, size_t to_id,
                                             const Pose3D& relative_pose,
                                             double noise_x, double noise_y, double noise_z,
                                             double noise_rot) {
    gtsam::Pose3 rel_pose = toGtsamPose(relative_pose);
    auto noise = createNoiseModel(noise_x, noise_y, noise_z, noise_rot, noise_rot, noise_rot);
    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(from_id, to_id, rel_pose, noise));

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
    gtsam::Pose3 rel_pose = toGtsamPose(relative_pose);
    auto noise = createNoiseModel(noise_x, noise_y, noise_z, noise_rot, noise_rot, noise_rot);
    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(from_id, to_id, rel_pose, noise));

    for (size_t id : {from_id, to_id}) {
        if (std::find(pose_ids_.begin(), pose_ids_.end(), id) == pose_ids_.end()) {
            pose_ids_.push_back(id);
        }
    }
}

void FactorGraphOptimizer::addPoseEstimate(size_t pose_id, const Pose3D& pose) {
    gtsam::Pose3 gtsam_pose = toGtsamPose(pose);
    initial_estimates_.insert(pose_id, gtsam_pose);

    if (std::find(pose_ids_.begin(), pose_ids_.end(), pose_id) == pose_ids_.end()) {
        pose_ids_.push_back(pose_id);
    }
}

bool FactorGraphOptimizer::optimize() {
    if (initial_estimates_.empty()) {
        std::cerr << "[Optimizer] No initial estimates provided" << std::endl;
        return false;
    }

    try {
        initial_error_ = graph_.error(initial_estimates_);

        if (optimizer_type_ == "levenberg_marquardt") {
            gtsam::LevenbergMarquardtParams params;
            params.setMaxIterations(max_iterations_);
            params.setVerbosity("SILENT");

            gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initial_estimates_, params);
            optimized_estimates_ = optimizer.optimize();
        } else if (optimizer_type_ == "gauss_newton") {
            gtsam::GaussNewtonParams params;
            params.setMaxIterations(max_iterations_);
            params.setVerbosity("SILENT");

            gtsam::GaussNewtonOptimizer optimizer(graph_, initial_estimates_, params);
            optimized_estimates_ = optimizer.optimize();
        } else if (optimizer_type_ == "dogleg") {
            gtsam::DoglegParams params;
            params.setMaxIterations(max_iterations_);
            params.setVerbosity("SILENT");

            gtsam::DoglegOptimizer optimizer(graph_, initial_estimates_, params);
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

Pose3D FactorGraphOptimizer::getOptimizedPose(size_t pose_id) const {
    if (optimized_estimates_.exists(pose_id)) {
        gtsam::Pose3 gtsam_pose = optimized_estimates_.at<gtsam::Pose3>(pose_id);
        return fromGtsamPose(gtsam_pose);
    }

    if (initial_estimates_.exists(pose_id)) {
        gtsam::Pose3 gtsam_pose = initial_estimates_.at<gtsam::Pose3>(pose_id);
        return fromGtsamPose(gtsam_pose);
    }

    return Pose3D();
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

gtsam::Pose3 FactorGraphOptimizer::toGtsamPose(const Pose3D& pose) {
    gtsam::Point3 point(pose.translation.x(), pose.translation.y(), pose.translation.z());
    gtsam::Rot3 rot(pose.rotation.toRotationMatrix());
    return gtsam::Pose3(rot, point);
}

Pose3D FactorGraphOptimizer::fromGtsamPose(const gtsam::Pose3& gtsam_pose) {
    Pose3D pose;
    pose.translation.x() = gtsam_pose.translation().x();
    pose.translation.y() = gtsam_pose.translation().y();
    pose.translation.z() = gtsam_pose.translation().z();
    pose.rotation = Eigen::Quaterniond(gtsam_pose.rotation().matrix());
    pose.rotation.normalize();
    return pose;
}

gtsam::noiseModel::Diagonal::shared_ptr FactorGraphOptimizer::createNoiseModel(
    double nx, double ny, double nz, double nrx, double nry, double nrz) {
    gtsam::Vector6 sigmas;
    sigmas << nrx, nry, nrz, nx, ny, nz;
    return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
}

} // namespace mine_slam
