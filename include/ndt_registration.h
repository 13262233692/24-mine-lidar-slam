#pragma once

#include "common_types.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <unordered_map>
#include <memory>
#include <vector>

namespace mine_slam {

struct NDTCell {
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d cov_inv = Eigen::Matrix3d::Identity();
    int point_count = 0;
    bool valid = false;

    bool computeDistribution(int min_points = 6, double eigenvalue_threshold = 0.001) {
        if (point_count < min_points) {
            valid = false;
            return false;
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
        if (solver.info() != Eigen::Success) {
            valid = false;
            return false;
        }

        Eigen::Vector3d evals = solver.eigenvalues();
        if (evals.minCoeff() < eigenvalue_threshold) {
            double min_ev = std::max(evals.minCoeff(), eigenvalue_threshold);
            Eigen::Matrix3d evecs = solver.eigenvectors();
            Eigen::Matrix3d diag = Eigen::Matrix3d::Identity();
            diag(0, 0) = std::max(evals(0), min_ev);
            diag(1, 1) = std::max(evals(1), min_ev);
            diag(2, 2) = std::max(evals(2), min_ev);
            cov = evecs * diag * evecs.transpose();
        }

        cov_inv = cov.inverse();
        valid = true;
        return true;
    }
};

struct VoxelKey {
    int x, y, z;

    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHash {
    size_t operator()(const VoxelKey& k) const {
        size_t h1 = std::hash<int>()(k.x);
        size_t h2 = std::hash<int>()(k.y);
        size_t h3 = std::hash<int>()(k.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

using NDTGrid = std::unordered_map<VoxelKey, NDTCell, VoxelKeyHash>;

class NDTRegistration {
public:
    struct LMParams {
        double lambda_init = 0.01;
        double lambda_factor = 10.0;
        double lambda_min = 1e-8;
        double lambda_max = 1e6;
        double eigenvalue_threshold = 0.01;
        double max_translation_step = 1.0;
        double max_rotation_step = 0.2;
        int max_iterations = 35;
        double convergence_delta = 0.01;
        int min_cell_points = 6;
        double covariance_regularization = 0.001;
    };

    NDTRegistration();
    ~NDTRegistration() = default;

    void setResolution(double resolution);
    void setStepSize(double step_size);
    void setMaxIterations(int max_iter);
    void setTransformationEpsilon(double epsilon);
    void setLMParams(const LMParams& params);

    void setInputTarget(PointCloudConstPtr target);
    void setInputSource(PointCloudConstPtr source);

    RegistrationResult align();
    RegistrationResult align(const Eigen::Matrix4d& initial_guess);

    double getFitnessScore() const;
    bool hasConverged() const;
    Eigen::Matrix4d getFinalTransformation() const;
    const DegeneracyInfo& getLastDegeneracyInfo() const;

private:
    void buildNDTGrid();

    double computeScore(const Eigen::Matrix4d& transformation) const;

    void computeGradientAndHessian(
        const Eigen::Matrix4d& transformation,
        Eigen::Matrix<double, 6, 1>& gradient,
        Eigen::Matrix<double, 6, 6>& hessian) const;

    Eigen::Matrix<double, 6, 1> solveLMStep(
        const Eigen::Matrix<double, 6, 1>& gradient,
        const Eigen::Matrix<double, 6, 6>& hessian,
        double lambda) const;

    DegeneracyInfo detectDegeneracy(
        const Eigen::Matrix<double, 6, 6>& hessian) const;

    Eigen::Matrix<double, 6, 6> applyEigenvalueTruncation(
        const Eigen::Matrix<double, 6, 6>& hessian,
        const DegeneracyInfo& info) const;

    Eigen::Matrix<double, 6, 1> constrainStep(
        const Eigen::Matrix<double, 6, 1>& delta) const;

    Eigen::Matrix4d incrementTransformation(
        const Eigen::Matrix4d& current,
        const Eigen::Matrix<double, 6, 1>& delta) const;

    VoxelKey pointToVoxelKey(const Eigen::Vector3d& point) const;

    Eigen::Matrix<double, 3, 6> computePointJacobian(
        const Eigen::Vector3d& transformed_point,
        const Eigen::Matrix3d& rotation) const;

    double transformFitnessScore(const Eigen::Matrix4d& transformation) const;

    NDTGrid ndt_grid_;
    PointCloudConstPtr target_cloud_;
    PointCloudConstPtr source_cloud_;

    double resolution_;
    double step_size_;
    int max_iterations_;
    double transformation_epsilon_;
    LMParams lm_params_;

    bool converged_;
    double fitness_score_;
    Eigen::Matrix4d final_transformation_;
    DegeneracyInfo last_degeneracy_;
};

using NDTRegistrationPtr = std::shared_ptr<NDTRegistration>;

} // namespace mine_slam
