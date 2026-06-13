#pragma once

#include "common_types.h"
#include <string>

namespace mine_slam {

struct SLAMConfig {
    DataSourceType source_type;
    std::string input_path;
    std::string point_cloud_topic;
    std::string output_dir;

    double voxel_leaf_size;
    double voxel_centroid_method;

    double ndt_resolution;
    double ndt_step_size;
    int ndt_max_iterations;
    double ndt_transformation_epsilon;

    double keyframe_translation_threshold;
    double keyframe_rotation_threshold;

    int loop_detection_enabled;
    double loop_detection_score_threshold;
    int loop_detection_history_count;

    double odometry_noise_x;
    double odometry_noise_y;
    double odometry_noise_z;
    double odometry_noise_rot;

    double loop_noise_x;
    double loop_noise_y;
    double loop_noise_z;
    double loop_noise_rot;

    int optimizer_max_iterations;

    bool save_map;
    bool save_trajectory;
    int map_downsample_step;

    SLAMConfig()
        : source_type(DataSourceType::PCD_DIR)
        , input_path("")
        , point_cloud_topic("/points_raw")
        , output_dir("./output")
        , voxel_leaf_size(0.5)
        , ndt_resolution(2.0)
        , ndt_step_size(0.1)
        , ndt_max_iterations(35)
        , ndt_transformation_epsilon(0.01)
        , keyframe_translation_threshold(1.0)
        , keyframe_rotation_threshold(0.05)
        , loop_detection_enabled(false)
        , loop_detection_score_threshold(2.0)
        , loop_detection_history_count(20)
        , odometry_noise_x(0.05)
        , odometry_noise_y(0.05)
        , odometry_noise_z(0.05)
        , odometry_noise_rot(0.02)
        , loop_noise_x(0.3)
        , loop_noise_y(0.3)
        , loop_noise_z(0.3)
        , loop_noise_rot(0.05)
        , optimizer_max_iterations(100)
        , save_map(true)
        , save_trajectory(true)
        , map_downsample_step(5) {}
};

class ConfigParser {
public:
    ConfigParser() = default;
    ~ConfigParser() = default;

    bool parseCommandLine(int argc, char** argv, SLAMConfig& config);

    void printConfig(const SLAMConfig& config) const;

private:
    DataSourceType detectSourceType(const std::string& path);
};

} // namespace mine_slam
