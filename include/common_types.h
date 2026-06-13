#pragma once

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <vector>
#include <string>
#include <cstdint>

namespace mine_slam {

using PointT = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<PointT>;
using PointCloudPtr = PointCloud::Ptr;
using PointCloudConstPtr = PointCloud::ConstPtr;

struct DegeneracyInfo {
    bool is_degenerate = false;
    double min_eigenvalue = 0.0;
    double condition_number = 0.0;
    double damping_lambda = 0.0;
    int degenerate_dimensions = 0;
    Eigen::Matrix<double, 6, 1> eigenvalues = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 6> eigenvectors = Eigen::Matrix<double, 6, 6>::Identity();

    void reset() {
        is_degenerate = false;
        min_eigenvalue = 0.0;
        condition_number = 0.0;
        damping_lambda = 0.0;
        degenerate_dimensions = 0;
        eigenvalues.setZero();
        eigenvectors.setIdentity();
    }
};

struct Pose3D {
    Eigen::Vector3d translation;
    Eigen::Quaterniond rotation;

    Pose3D()
        : translation(Eigen::Vector3d::Zero())
        , rotation(Eigen::Quaterniond::Identity()) {}

    Pose3D(const Eigen::Vector3d& t, const Eigen::Quaterniond& r)
        : translation(t)
        , rotation(r) {}

    Eigen::Matrix4d toMatrix() const {
        Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
        mat.block<3, 3>(0, 0) = rotation.toRotationMatrix();
        mat.block<3, 1>(0, 3) = translation;
        return mat;
    }

    static Pose3D fromMatrix(const Eigen::Matrix4d& mat) {
        Pose3D pose;
        pose.translation = mat.block<3, 1>(0, 3);
        pose.rotation = Eigen::Quaterniond(mat.block<3, 3>(0, 0));
        pose.rotation.normalize();
        return pose;
    }

    Pose3D inverse() const {
        Eigen::Quaterniond inv_rot = rotation.inverse();
        Eigen::Vector3d inv_trans = -(inv_rot * translation);
        return Pose3D(inv_trans, inv_rot);
    }

    Pose3D operator*(const Pose3D& other) const {
        Eigen::Vector3d new_trans = translation + rotation * other.translation;
        Eigen::Quaterniond new_rot = rotation * other.rotation;
        new_rot.normalize();
        return Pose3D(new_trans, new_rot);
    }
};

struct LidarScan {
    uint64_t timestamp;
    size_t frame_id;
    PointCloudPtr cloud;

    LidarScan() : timestamp(0), frame_id(0), cloud(new PointCloud()) {}
};

using LidarScanPtr = std::shared_ptr<LidarScan>;
using LidarScanConstPtr = std::shared_ptr<const LidarScan>;

struct RegistrationResult {
    bool converged;
    double fitness_score;
    size_t iterations;
    Eigen::Matrix4d transformation;
    DegeneracyInfo degeneracy;

    RegistrationResult()
        : converged(false)
        , fitness_score(0.0)
        , iterations(0)
        , transformation(Eigen::Matrix4d::Identity()) {}
};

struct KeyFrame {
    size_t id;
    uint64_t timestamp;
    Pose3D pose;
    PointCloudPtr cloud;

    KeyFrame() : id(0), timestamp(0), cloud(new PointCloud()) {}
};

using KeyFramePtr = std::shared_ptr<KeyFrame>;
using KeyFrameConstPtr = std::shared_ptr<const KeyFrame>;

struct TrajectoryPoint {
    size_t frame_id;
    uint64_t timestamp;
    Pose3D pose;
};

enum class DataSourceType {
    PCD_DIR,
    ROS_BAG,
    UNKNOWN
};

} // namespace mine_slam
