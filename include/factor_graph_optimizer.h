#pragma once

#include "common_types.h"
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/geometry/Rot3.h>
#include <memory>
#include <vector>

namespace mine_slam {

class FactorGraphOptimizer {
public:
    FactorGraphOptimizer();
    ~FactorGraphOptimizer() = default;

    void setMaxIterations(int iterations);
    void setOptimizerType(const std::string& type);

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

    bool optimize();

    Pose3D getOptimizedPose(size_t pose_id) const;
    std::vector<Pose3D> getAllOptimizedPoses() const;

    size_t getPoseCount() const { return pose_ids_.size(); }
    size_t getFactorCount() const { return graph_.size(); }

    double getInitialError() const { return initial_error_; }
    double getFinalError() const { return final_error_; }

private:
    static gtsam::Pose3 toGtsamPose(const Pose3D& pose);
    static Pose3D fromGtsamPose(const gtsam::Pose3& gtsam_pose);
    static gtsam::noiseModel::Diagonal::shared_ptr createNoiseModel(
        double nx, double ny, double nz, double nrx, double nry, double nrz);

    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values initial_estimates_;
    gtsam::Values optimized_estimates_;

    std::vector<size_t> pose_ids_;

    int max_iterations_;
    std::string optimizer_type_;

    double initial_error_;
    double final_error_;
};

using FactorGraphOptimizerPtr = std::shared_ptr<FactorGraphOptimizer>;

} // namespace mine_slam
