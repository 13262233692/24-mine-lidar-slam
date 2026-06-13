#pragma once

#include "common_types.h"
#include "point_cloud_source.h"
#include "voxel_filter.h"
#include "ndt_registration.h"
#include "factor_graph_optimizer.h"
#include "config_parser.h"
#include <memory>
#include <vector>
#include <deque>
#include <fstream>

namespace mine_slam {

class MapBuilder {
public:
    MapBuilder();
    ~MapBuilder();

    bool initialize(const SLAMConfig& config);

    void run();

    void stop();

    bool isRunning() const { return running_; }

    size_t processedFrames() const { return processed_frames_; }
    size_t keyFrameCount() const { return keyframes_.size(); }

    std::vector<Pose3D> getTrajectory() const;
    PointCloudPtr getMap() const;

    void saveTrajectory(const std::string& filename) const;
    void saveMap(const std::string& filename) const;

private:
    bool processFrame(LidarScanPtr scan);

    bool addKeyFrame(LidarScanPtr scan, const Pose3D& pose);

    RegistrationResult registerScanToKeyFrame(
        PointCloudConstPtr scan_cloud,
        KeyFrameConstPtr keyframe,
        const Eigen::Matrix4d& initial_guess);

    bool shouldCreateKeyFrame(const Pose3D& current_pose,
                              const Pose3D& last_keyframe_pose) const;

    bool detectLoopClosure(KeyFramePtr current_kf,
                           size_t& matched_id,
                           Pose3D& relative_pose);

    void updateOptimizedPoses();

    PointCloudPtr buildMapFromKeyFrames() const;

    void printProgress(size_t current, size_t total) const;

    SLAMConfig config_;
    PointCloudSourcePtr data_source_;
    VoxelFilterPtr voxel_filter_;
    NDTRegistrationPtr ndt_registration_;
    FactorGraphOptimizerPtr optimizer_;

    std::vector<KeyFramePtr> keyframes_;
    std::vector<TrajectoryPoint> trajectory_;

    Pose3D current_pose_;
    Pose3D last_keyframe_pose_;

    bool running_;
    size_t processed_frames_;
    size_t loop_closure_count_;

    std::ofstream trajectory_file_;
};

using MapBuilderPtr = std::shared_ptr<MapBuilder>;

} // namespace mine_slam
