#include "map_builder.h"
#include "timer.h"
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

namespace mine_slam {

MapBuilder::MapBuilder()
    : running_(false)
    , processed_frames_(0)
    , loop_closure_count_(0) {}

MapBuilder::~MapBuilder() {
    stop();
    if (trajectory_file_.is_open()) {
        trajectory_file_.close();
    }
}

bool MapBuilder::initialize(const SLAMConfig& config) {
    config_ = config;

    data_source_ = PointCloudSourceFactory::create(config.source_type);
    if (!data_source_) {
        std::cerr << "[MapBuilder] Failed to create data source" << std::endl;
        return false;
    }

    if (!data_source_->open(config.input_path)) {
        std::cerr << "[MapBuilder] Failed to open data source: "
                  << config.input_path << std::endl;
        return false;
    }

    voxel_filter_ = std::make_shared<VoxelFilter>();
    voxel_filter_->setLeafSize(config.voxel_leaf_size);
    voxel_filter_->setUseOctree(true);

    ndt_registration_ = std::make_shared<NDTRegistration>();
    ndt_registration_->setResolution(config.ndt_resolution);
    ndt_registration_->setStepSize(config.ndt_step_size);
    ndt_registration_->setMaxIterations(config.ndt_max_iterations);
    ndt_registration_->setTransformationEpsilon(config.ndt_transformation_epsilon);

    NDTRegistration::LMParams lm_params;
    lm_params.lambda_init = config.lm_lambda_init;
    lm_params.lambda_factor = config.lm_lambda_factor;
    lm_params.lambda_min = config.lm_lambda_min;
    lm_params.lambda_max = config.lm_lambda_max;
    lm_params.eigenvalue_threshold = config.lm_eigenvalue_threshold;
    lm_params.max_translation_step = config.lm_max_translation_step;
    lm_params.max_rotation_step = config.lm_max_rotation_step;
    lm_params.covariance_regularization = config.lm_covariance_regularization;
    lm_params.min_cell_points = config.lm_min_cell_points;
    lm_params.max_iterations = config.ndt_max_iterations;
    lm_params.convergence_delta = config.ndt_transformation_epsilon;
    ndt_registration_->setLMParams(lm_params);

    optimizer_ = std::make_shared<FactorGraphOptimizer>();
    optimizer_->setMaxIterations(config.optimizer_max_iterations);
    optimizer_->setOptimizerType("levenberg_marquardt");

    current_pose_ = Pose3D();
    last_keyframe_pose_ = Pose3D();
    processed_frames_ = 0;
    loop_closure_count_ = 0;
    keyframes_.clear();
    trajectory_.clear();

    if (config.save_trajectory) {
        std::string traj_path = config.output_dir + "/trajectory.txt";
        trajectory_file_.open(traj_path);
        if (trajectory_file_.is_open()) {
            trajectory_file_ << "# frame_id timestamp x y z qx qy qz qw" << std::endl;
        }
    }

    running_ = true;

    std::cout << "[MapBuilder] Initialized successfully" << std::endl;
    std::cout << "[MapBuilder] Total frames: " << data_source_->totalFrames() << std::endl;

    return true;
}

void MapBuilder::run() {
    if (!running_) {
        std::cerr << "[MapBuilder] Not initialized" << std::endl;
        return;
    }

    std::cout << "[MapBuilder] Starting SLAM processing..." << std::endl;
    Timer total_timer("Total processing");

    size_t total = data_source_->totalFrames();

    while (data_source_->hasNext() && running_) {
        LidarScanPtr scan = data_source_->next();
        if (!scan) {
            std::cerr << "[MapBuilder] Failed to get scan, skipping..." << std::endl;
            continue;
        }

        if (!processFrame(scan)) {
            std::cerr << "[MapBuilder] Failed to process frame "
                      << processed_frames_ << std::endl;
        }

        processed_frames_++;

        if (processed_frames_ % 10 == 0 || processed_frames_ == total) {
            printProgress(processed_frames_, total);
        }
    }

    std::cout << std::endl;

    if (config_.loop_detection_enabled && keyframes_.size() > 2) {
        std::cout << "[MapBuilder] Running final optimization..." << std::endl;
        if (optimizer_->optimize()) {
            std::cout << "[MapBuilder] Optimization completed: "
                      << "initial error = " << optimizer_->getInitialError()
                      << ", final error = " << optimizer_->getFinalError()
                      << std::endl;
            updateOptimizedPoses();
        }
    }

    total_timer.print("Total processing");

    std::cout << "[MapBuilder] Processing complete" << std::endl;
    std::cout << "[MapBuilder] Processed frames: " << processed_frames_ << std::endl;
    std::cout << "[MapBuilder] Keyframes: " << keyframes_.size() << std::endl;
    std::cout << "[MapBuilder] Loop closures: " << loop_closure_count_ << std::endl;

    if (config_.save_trajectory) {
        saveTrajectory(config_.output_dir + "/trajectory.txt");
    }

    if (config_.save_map) {
        saveMap(config_.output_dir + "/map.pcd");
    }

    running_ = false;
}

void MapBuilder::stop() {
    running_ = false;
}

bool MapBuilder::processFrame(LidarScanPtr scan) {
    ScopedTimer timer("Frame processing");

    PointCloudPtr filtered_cloud = voxel_filter_->filter(scan->cloud);

    if (keyframes_.empty()) {
        return addKeyFrame(scan, current_pose_);
    }

    KeyFramePtr last_kf = keyframes_.back();

    Eigen::Matrix4d initial_guess = last_kf->pose.toMatrix();
    RegistrationResult result = registerScanToKeyFrame(
        filtered_cloud, last_kf, initial_guess);

    if (!result.converged) {
        std::cerr << "[MapBuilder] NDT did not converge for frame "
                  << scan->frame_id << std::endl;
        return false;
    }

    if (result.degeneracy.is_degenerate) {
        std::cerr << "[MapBuilder] DEGENERATE frame=" << scan->frame_id
                  << " dims=" << result.degeneracy.degenerate_dimensions
                  << " min_eigenval=" << std::scientific << result.degeneracy.min_eigenvalue
                  << " lambda=" << result.degeneracy.damping_lambda
                  << std::fixed << std::endl;

        if (result.degeneracy.degenerate_dimensions >= 3) {
            std::cerr << "[MapBuilder] Severe degeneracy (" 
                      << result.degeneracy.degenerate_dimensions
                      << " dims), clamping pose to last keyframe + odometry" << std::endl;
            
            Eigen::Vector3d delta_trans = current_pose_.translation - last_keyframe_pose_.translation;
            double clamp_dist = std::min(delta_trans.norm(), config_.keyframe_translation_threshold * 0.5);
            if (delta_trans.norm() > 1e-6) {
                delta_trans = delta_trans.normalized() * clamp_dist;
            }
            
            Pose3D clamped_pose;
            clamped_pose.translation = last_keyframe_pose_.translation + delta_trans;
            clamped_pose.rotation = current_pose_.rotation;
            current_pose_ = clamped_pose;
        } else {
            current_pose_ = Pose3D::fromMatrix(result.transformation);
        }
    } else {
        current_pose_ = Pose3D::fromMatrix(result.transformation);
    }

    TrajectoryPoint traj_point;
    traj_point.frame_id = scan->frame_id;
    traj_point.timestamp = scan->timestamp;
    traj_point.pose = current_pose_;
    trajectory_.push_back(traj_point);

    if (trajectory_file_.is_open()) {
        const auto& p = current_pose_.translation;
        const auto& q = current_pose_.rotation;
        trajectory_file_ << scan->frame_id << " "
                         << scan->timestamp << " "
                         << std::fixed << std::setprecision(6)
                         << p.x() << " " << p.y() << " " << p.z() << " "
                         << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
                         << std::endl;
    }

    if (shouldCreateKeyFrame(current_pose_, last_keyframe_pose_)) {
        if (!addKeyFrame(scan, current_pose_)) {
            return false;
        }

        if (config_.loop_detection_enabled && keyframes_.size() > 3) {
            size_t matched_id;
            Pose3D relative_pose;
            if (detectLoopClosure(keyframes_.back(), matched_id, relative_pose)) {
                loop_closure_count_++;
                std::cout << "[MapBuilder] Loop closure detected: frame "
                          << keyframes_.back()->id << " <-> " << matched_id
                          << std::endl;

                optimizer_->addLoopClosureFactor(
                    keyframes_.back()->id, matched_id,
                    relative_pose,
                    config_.loop_noise_x, config_.loop_noise_y,
                    config_.loop_noise_z, config_.loop_noise_rot);

                if (loop_closure_count_ % 5 == 0) {
                    optimizer_->optimize();
                    updateOptimizedPoses();
                }
            }
        }
    }

    return true;
}

bool MapBuilder::addKeyFrame(LidarScanPtr scan, const Pose3D& pose) {
    auto kf = std::make_shared<KeyFrame>();
    kf->id = keyframes_.size();
    kf->timestamp = scan->timestamp;
    kf->pose = pose;
    kf->cloud = voxel_filter_->filter(scan->cloud);

    keyframes_.push_back(kf);
    last_keyframe_pose_ = pose;

    optimizer_->addPoseEstimate(kf->id, pose);

    if (kf->id == 0) {
        optimizer_->addPriorFactor(
            0, pose,
            0.01, 0.01, 0.01, 0.001);
    } else if (kf->id > 0) {
        const KeyFramePtr prev_kf = keyframes_[kf->id - 1];
        Pose3D relative = prev_kf->pose.inverse() * pose;

        optimizer_->addOdometryFactor(
            prev_kf->id, kf->id, relative,
            config_.odometry_noise_x, config_.odometry_noise_y,
            config_.odometry_noise_z, config_.odometry_noise_rot);
    }

    return true;
}

RegistrationResult MapBuilder::registerScanToKeyFrame(
    PointCloudConstPtr scan_cloud,
    KeyFrameConstPtr keyframe,
    const Eigen::Matrix4d& initial_guess) {

    ndt_registration_->setInputTarget(keyframe->cloud);
    ndt_registration_->setInputSource(scan_cloud);
    return ndt_registration_->align(initial_guess);
}

bool MapBuilder::shouldCreateKeyFrame(const Pose3D& current_pose,
                                      const Pose3D& last_kf_pose) const {
    double trans_dist = (current_pose.translation - last_kf_pose.translation).norm();
    if (trans_dist >= config_.keyframe_translation_threshold) {
        return true;
    }

    Eigen::Quaterniond q_rel = last_kf_pose.rotation.inverse() * current_pose.rotation;
    double angle = 2.0 * std::acos(std::min(1.0, std::abs(q_rel.w())));
    if (angle >= config_.keyframe_rotation_threshold) {
        return true;
    }

    if (keyframes_.empty()) {
        return true;
    }

    return false;
}

bool MapBuilder::detectLoopClosure(KeyFramePtr current_kf,
                                   size_t& matched_id,
                                   Pose3D& relative_pose) {
    int history_count = std::min(
        static_cast<int>(keyframes_.size()) - 3,
        config_.loop_detection_history_count);

    if (history_count <= 0) return false;

    double best_score = std::numeric_limits<double>::max();
    size_t best_id = 0;
    Eigen::Matrix4d best_transform;

    for (int i = 0; i < history_count; ++i) {
        size_t kf_idx = i;
        KeyFramePtr candidate = keyframes_[kf_idx];

        double dist = (current_kf->pose.translation - candidate->pose.translation).norm();
        if (dist > 50.0) continue;

        RegistrationResult result = registerScanToKeyFrame(
            current_kf->cloud, candidate, candidate->pose.toMatrix());

        if (result.converged && result.fitness_score < best_score) {
            best_score = result.fitness_score;
            best_id = kf_idx;
            best_transform = result.transformation;
        }
    }

    if (best_score < config_.loop_detection_score_threshold && best_id < current_kf->id - 2) {
        matched_id = best_id;
        Pose3D absolute_pose = Pose3D::fromMatrix(best_transform);
        relative_pose = current_kf->pose.inverse() * absolute_pose;
        return true;
    }

    return false;
}

void MapBuilder::updateOptimizedPoses() {
    for (auto& kf : keyframes_) {
        kf->pose = optimizer_->getOptimizedPose(kf->id);
    }
}

std::vector<Pose3D> MapBuilder::getTrajectory() const {
    std::vector<Pose3D> poses;
    poses.reserve(trajectory_.size());
    for (const auto& tp : trajectory_) {
        poses.push_back(tp.pose);
    }
    return poses;
}

PointCloudPtr MapBuilder::getMap() const {
    return buildMapFromKeyFrames();
}

PointCloudPtr MapBuilder::buildMapFromKeyFrames() const {
    auto map_cloud = std::make_shared<PointCloud>();

    if (keyframes_.empty()) {
        return map_cloud;
    }

    size_t step = std::max(1, config_.map_downsample_step);

    for (size_t i = 0; i < keyframes_.size(); i += step) {
        const auto& kf = keyframes_[i];
        auto transformed = std::make_shared<PointCloud>();
        pcl::transformPointCloud(*kf->cloud, *transformed, kf->pose.toMatrix().cast<float>());
        *map_cloud += *transformed;
    }

    pcl::VoxelGrid<PointT> voxel_grid;
    voxel_grid.setInputCloud(map_cloud);
    voxel_grid.setLeafSize(
        static_cast<float>(config_.voxel_leaf_size),
        static_cast<float>(config_.voxel_leaf_size),
        static_cast<float>(config_.voxel_leaf_size));

    auto filtered = std::make_shared<PointCloud>();
    voxel_grid.filter(*filtered);

    return filtered;
}

void MapBuilder::saveTrajectory(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[MapBuilder] Failed to open trajectory file: "
                  << filename << std::endl;
        return;
    }

    file << "# frame_id timestamp x y z qx qy qz qw" << std::endl;
    file << "# Total frames: " << trajectory_.size() << std::endl;

    for (const auto& tp : trajectory_) {
        const auto& p = tp.pose.translation;
        const auto& q = tp.pose.rotation;
        file << tp.frame_id << " "
             << tp.timestamp << " "
             << std::fixed << std::setprecision(9)
             << p.x() << " " << p.y() << " " << p.z() << " "
             << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
             << std::endl;
    }

    file.close();
    std::cout << "[MapBuilder] Trajectory saved to: " << filename << std::endl;
    std::cout << "[MapBuilder] Total trajectory points: " << trajectory_.size() << std::endl;
}

void MapBuilder::saveMap(const std::string& filename) const {
    std::cout << "[MapBuilder] Building map from " << keyframes_.size()
              << " keyframes..." << std::endl;

    PointCloudPtr map_cloud = buildMapFromKeyFrames();

    if (map_cloud->empty()) {
        std::cerr << "[MapBuilder] Map is empty, not saving" << std::endl;
        return;
    }

    pcl::io::savePCDFileBinary(filename, *map_cloud);
    std::cout << "[MapBuilder] Map saved to: " << filename << std::endl;
    std::cout << "[MapBuilder] Map points: " << map_cloud->size() << std::endl;
}

void MapBuilder::printProgress(size_t current, size_t total) const {
    if (total == 0) return;

    double progress = static_cast<double>(current) / static_cast<double>(total) * 100.0;

    std::cout << "\r[MapBuilder] Progress: "
              << std::fixed << std::setprecision(1) << progress << "% "
              << "(" << current << "/" << total << ") "
              << "Keyframes: " << keyframes_.size();
    std::cout.flush();
}

} // namespace mine_slam
