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

    double lm_lambda_init;
    double lm_lambda_factor;
    double lm_lambda_min;
    double lm_lambda_max;
    double lm_eigenvalue_threshold;
    double lm_max_translation_step;
    double lm_max_rotation_step;
    double lm_covariance_regularization;
    int lm_min_cell_points;

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

    bool imu_enabled;
    std::string imu_data_path;
    double imu_accel_noise;
    double imu_gyro_noise;
    double imu_accel_bias_rw;
    double imu_gyro_bias_rw;
    double imu_gravity_x;
    double imu_gravity_y;
    double imu_gravity_z;
    double imu_init_velocity_noise;

    bool use_isam2;
    int isam2_relinearize_skip;
    double isam2_relinearize_threshold;

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
        , lm_lambda_init(0.01)
        , lm_lambda_factor(10.0)
        , lm_lambda_min(1e-8)
        , lm_lambda_max(1e6)
        , lm_eigenvalue_threshold(0.01)
        , lm_max_translation_step(1.0)
        , lm_max_rotation_step(0.2)
        , lm_covariance_regularization(0.001)
        , lm_min_cell_points(6)
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
        , imu_enabled(false)
        , imu_data_path("")
        , imu_accel_noise(0.01)
        , imu_gyro_noise(0.001)
        , imu_accel_bias_rw(0.0001)
        , imu_gyro_bias_rw(0.00001)
        , imu_gravity_x(0.0)
        , imu_gravity_y(0.0)
        , imu_gravity_z(-9.81)
        , imu_init_velocity_noise(1.0)
        , use_isam2(false)
        , isam2_relinearize_skip(10)
        , isam2_relinearize_threshold(0.1)
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
