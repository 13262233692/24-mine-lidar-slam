#include "ndt_registration.h"
#include <pcl/common/transforms.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>

namespace mine_slam {

NDTRegistration::NDTRegistration()
    : resolution_(2.0)
    , step_size_(0.1)
    , max_iterations_(35)
    , transformation_epsilon_(0.01)
    , converged_(false)
    , fitness_score_(std::numeric_limits<double>::max())
    , final_transformation_(Eigen::Matrix4d::Identity()) {}

void NDTRegistration::setResolution(double resolution) {
    resolution_ = resolution;
}

void NDTRegistration::setStepSize(double step_size) {
    step_size_ = step_size;
    lm_params_.max_translation_step = step_size * 10.0;
    lm_params_.max_rotation_step = step_size * 2.0;
}

void NDTRegistration::setMaxIterations(int max_iter) {
    max_iterations_ = max_iter;
    lm_params_.max_iterations = max_iter;
}

void NDTRegistration::setTransformationEpsilon(double epsilon) {
    transformation_epsilon_ = epsilon;
    lm_params_.convergence_delta = epsilon;
}

void NDTRegistration::setLMParams(const LMParams& params) {
    lm_params_ = params;
}

void NDTRegistration::setInputTarget(PointCloudConstPtr target) {
    target_cloud_ = target;
    ndt_grid_.clear();
    buildNDTGrid();
}

void NDTRegistration::setInputSource(PointCloudConstPtr source) {
    source_cloud_ = source;
}

void NDTRegistration::buildNDTGrid() {
    if (!target_cloud_ || target_cloud_->empty()) return;

    ndt_grid_.clear();
    double inv_res = 1.0 / resolution_;

    std::unordered_map<VoxelKey, std::vector<Eigen::Vector3d>, VoxelKeyHash> voxel_points;

    for (const auto& pt : target_cloud_->points) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;

        Eigen::Vector3d p(pt.x, pt.y, pt.z);
        VoxelKey key;
        key.x = static_cast<int>(std::floor(p.x() * inv_res));
        key.y = static_cast<int>(std::floor(p.y() * inv_res));
        key.z = static_cast<int>(std::floor(p.z() * inv_res));

        voxel_points[key].push_back(p);
    }

    for (auto& [key, points] : voxel_points) {
        NDTCell cell;
        cell.point_count = static_cast<int>(points.size());

        if (cell.point_count < lm_params_.min_cell_points) {
            cell.valid = false;
            ndt_grid_[key] = cell;
            continue;
        }

        cell.mean.setZero();
        for (const auto& p : points) {
            cell.mean += p;
        }
        cell.mean /= static_cast<double>(cell.point_count);

        cell.cov.setZero();
        for (const auto& p : points) {
            Eigen::Vector3d d = p - cell.mean;
            cell.cov += d * d.transpose();
        }
        cell.cov /= static_cast<double>(cell.point_count - 1);

        cell.cov += lm_params_.covariance_regularization * Eigen::Matrix3d::Identity();

        cell.computeDistribution(lm_params_.min_cell_points,
                                 lm_params_.covariance_regularization);
        ndt_grid_[key] = cell;
    }

    size_t valid_cells = 0;
    for (const auto& [key, cell] : ndt_grid_) {
        if (cell.valid) valid_cells++;
    }
    std::cout << "[NDT] Grid built: " << ndt_grid_.size()
              << " cells, " << valid_cells << " valid" << std::endl;
}

double NDTRegistration::computeScore(const Eigen::Matrix4d& transformation) const {
    double score = 0.0;
    double inv_res = 1.0 / resolution_;
    int matched = 0;

    Eigen::Matrix3d R = transformation.block<3, 3>(0, 0);
    Eigen::Vector3d t = transformation.block<3, 1>(0, 3);

    for (const auto& pt : source_cloud_->points) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;

        Eigen::Vector3d p(pt.x, pt.y, pt.z);
        Eigen::Vector3d y = R * p + t;

        VoxelKey key;
        key.x = static_cast<int>(std::floor(y.x() * inv_res));
        key.y = static_cast<int>(std::floor(y.y() * inv_res));
        key.z = static_cast<int>(std::floor(y.z() * inv_res));

        auto it = ndt_grid_.find(key);
        if (it == ndt_grid_.end() || !it->second.valid) continue;

        const NDTCell& cell = it->second;
        Eigen::Vector3d d = y - cell.mean;
        double mal = d.transpose() * cell.cov_inv * d;

        if (mal < 400.0) {
            score += std::exp(-0.5 * mal);
            matched++;
        }
    }

    return (matched > 0) ? score : -1e10;
}

void NDTRegistration::computeGradientAndHessian(
    const Eigen::Matrix4d& transformation,
    Eigen::Matrix<double, 6, 1>& gradient,
    Eigen::Matrix<double, 6, 6>& hessian) const {

    gradient.setZero();
    hessian.setZero();

    double inv_res = 1.0 / resolution_;

    Eigen::Matrix3d R = transformation.block<3, 3>(0, 0);
    Eigen::Vector3d t = transformation.block<3, 1>(0, 3);

    for (const auto& pt : source_cloud_->points) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;

        Eigen::Vector3d x(pt.x, pt.y, pt.z);
        Eigen::Vector3d y = R * x + t;

        VoxelKey key;
        key.x = static_cast<int>(std::floor(y.x() * inv_res));
        key.y = static_cast<int>(std::floor(y.y() * inv_res));
        key.z = static_cast<int>(std::floor(y.z() * inv_res));

        auto it = ndt_grid_.find(key);
        if (it == ndt_grid_.end() || !it->second.valid) continue;

        const NDTCell& cell = it->second;
        Eigen::Vector3d d = y - cell.mean;
        double mal = d.transpose() * cell.cov_inv * d;

        if (mal > 400.0) continue;

        double p_i = std::exp(-0.5 * mal);

        Eigen::Matrix<double, 3, 6> J = computePointJacobian(y, R);

        Eigen::Vector3d cov_inv_d = cell.cov_inv * d;

        gradient += p_i * J.transpose() * cov_inv_d;

        hessian += p_i * J.transpose() * cell.cov_inv * J;
    }
}

Eigen::Matrix<double, 3, 6> NDTRegistration::computePointJacobian(
    const Eigen::Vector3d& y,
    const Eigen::Matrix3d& R) const {

    Eigen::Matrix<double, 3, 6> J;
    J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

    J(0, 3) = 0.0;    J(1, 3) = -y.z();  J(2, 3) = y.y();
    J(0, 4) = y.z();  J(1, 4) = 0.0;     J(2, 4) = -y.x();
    J(0, 5) = -y.y(); J(1, 5) = y.x();   J(2, 5) = 0.0;

    return J;
}

Eigen::Matrix<double, 6, 1> NDTRegistration::solveLMStep(
    const Eigen::Matrix<double, 6, 1>& gradient,
    const Eigen::Matrix<double, 6, 6>& hessian,
    double lambda) const {

    Eigen::Matrix<double, 6, 6> H_damped = hessian;

    Eigen::Matrix<double, 6, 1> diag = hessian.diagonal();
    for (int i = 0; i < 6; ++i) {
        double d = std::max(std::abs(diag(i)), 1e-6);
        H_damped(i, i) += lambda * d;
    }

    Eigen::Matrix<double, 6, 1> delta;
    delta = H_damped.ldlt().solve(gradient);

    if (!delta.allFinite()) {
        delta = H_damped.fullPivLu().solve(gradient);
        if (!delta.allFinite()) {
            delta.setZero();
        }
    }

    return delta;
}

DegeneracyInfo NDTRegistration::detectDegeneracy(
    const Eigen::Matrix<double, 6, 6>& hessian) const {

    DegeneracyInfo info;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(hessian);
    if (solver.info() != Eigen::Success) {
        info.is_degenerate = true;
        info.degenerate_dimensions = 6;
        info.min_eigenvalue = 0.0;
        info.condition_number = std::numeric_limits<double>::infinity();
        return info;
    }

    info.eigenvalues = solver.eigenvalues();
    info.eigenvectors = solver.eigenvectors();

    info.min_eigenvalue = info.eigenvalues.minCoeff();
    double max_eigenvalue = info.eigenvalues.maxCoeff();

    info.condition_number = (info.min_eigenvalue > 1e-10)
        ? max_eigenvalue / info.min_eigenvalue
        : std::numeric_limits<double>::infinity();

    info.degenerate_dimensions = 0;
    for (int i = 0; i < 6; ++i) {
        if (info.eigenvalues(i) < lm_params_.eigenvalue_threshold) {
            info.degenerate_dimensions++;
        }
    }

    info.is_degenerate = (info.degenerate_dimensions > 0) ||
                         (info.condition_number > 1e6);

    return info;
}

Eigen::Matrix<double, 6, 6> NDTRegistration::applyEigenvalueTruncation(
    const Eigen::Matrix<double, 6, 6>& hessian,
    const DegeneracyInfo& info) const {

    if (!info.is_degenerate) return hessian;

    Eigen::DiagonalMatrix<double, 6> truncated_evals;
    for (int i = 0; i < 6; ++i) {
        truncated_evals.diagonal()(i) = std::max(
            info.eigenvalues(i), lm_params_.eigenvalue_threshold);
    }

    Eigen::Matrix<double, 6, 6> H_truncated =
        info.eigenvectors * truncated_evals.toDenseMatrix() * info.eigenvectors.transpose();

    return H_truncated;
}

Eigen::Matrix<double, 6, 1> NDTRegistration::constrainStep(
    const Eigen::Matrix<double, 6, 1>& delta) const {

    Eigen::Matrix<double, 6, 1> constrained = delta;

    double trans_norm = constrained.head<3>().norm();
    if (trans_norm > lm_params_.max_translation_step) {
        constrained.head<3>() *= lm_params_.max_translation_step / trans_norm;
    }

    double rot_norm = constrained.tail<3>().norm();
    if (rot_norm > lm_params_.max_rotation_step) {
        constrained.tail<3>() *= lm_params_.max_rotation_step / rot_norm;
    }

    if (!constrained.allFinite()) {
        constrained.setZero();
    }

    return constrained;
}

Eigen::Matrix4d NDTRegistration::incrementTransformation(
    const Eigen::Matrix4d& current,
    const Eigen::Matrix<double, 6, 1>& delta) const {

    Eigen::Vector3d dt = delta.head<3>();
    Eigen::Vector3d dw = delta.tail<3>();

    double angle = dw.norm();
    Eigen::Matrix3d dR = Eigen::Matrix3d::Identity();

    if (angle > 1e-10) {
        Eigen::Vector3d axis = dw / angle;
        dR = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    }

    Eigen::Matrix4d delta_T = Eigen::Matrix4d::Identity();
    delta_T.block<3, 3>(0, 0) = dR;
    delta_T.block<3, 1>(0, 3) = dt;

    return delta_T * current;
}

VoxelKey NDTRegistration::pointToVoxelKey(const Eigen::Vector3d& point) const {
    double inv_res = 1.0 / resolution_;
    VoxelKey key;
    key.x = static_cast<int>(std::floor(point.x() * inv_res));
    key.y = static_cast<int>(std::floor(point.y() * inv_res));
    key.z = static_cast<int>(std::floor(point.z() * inv_res));
    return key;
}

double NDTRegistration::transformFitnessScore(const Eigen::Matrix4d& transformation) const {
    double total_dist = 0.0;
    int count = 0;
    double inv_res = 1.0 / resolution_;

    Eigen::Matrix3d R = transformation.block<3, 3>(0, 0);
    Eigen::Vector3d t = transformation.block<3, 1>(0, 3);

    for (const auto& pt : source_cloud_->points) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;

        Eigen::Vector3d p(pt.x, pt.y, pt.z);
        Eigen::Vector3d y = R * p + t;

        VoxelKey key;
        key.x = static_cast<int>(std::floor(y.x() * inv_res));
        key.y = static_cast<int>(std::floor(y.y() * inv_res));
        key.z = static_cast<int>(std::floor(y.z() * inv_res));

        auto it = ndt_grid_.find(key);
        if (it == ndt_grid_.end() || !it->second.valid) continue;

        const NDTCell& cell = it->second;
        Eigen::Vector3d d = y - cell.mean;
        double dist = d.transpose() * cell.cov_inv * d;
        total_dist += dist;
        count++;
    }

    return (count > 0) ? total_dist / static_cast<double>(count)
                        : std::numeric_limits<double>::max();
}

RegistrationResult NDTRegistration::align() {
    return align(Eigen::Matrix4d::Identity());
}

RegistrationResult NDTRegistration::align(const Eigen::Matrix4d& initial_guess) {
    RegistrationResult result;
    last_degeneracy_.reset();

    if (!source_cloud_ || source_cloud_->empty()) {
        std::cerr << "[NDT] Source cloud is empty" << std::endl;
        result.converged = false;
        return result;
    }

    if (ndt_grid_.empty()) {
        std::cerr << "[NDT] NDT grid is empty (no target set)" << std::endl;
        result.converged = false;
        return result;
    }

    Eigen::Matrix4d current_transform = initial_guess;
    double current_score = computeScore(current_transform);

    if (current_score <= -1e9) {
        std::cerr << "[NDT] Initial score is invalid, trying identity" << std::endl;
        current_transform = Eigen::Matrix4d::Identity();
        current_score = computeScore(current_transform);
    }

    double lambda = lm_params_.lambda_init;
    int stagnation_count = 0;
    const int max_stagnation = 5;

    Eigen::Matrix<double, 6, 6> last_hessian = Eigen::Matrix<double, 6, 6>::Identity();

    for (int iter = 0; iter < lm_params_.max_iterations; ++iter) {
        Eigen::Matrix<double, 6, 1> gradient;
        Eigen::Matrix<double, 6, 6> hessian;

        computeGradientAndHessian(current_transform, gradient, hessian);
        last_hessian = hessian;

        DegeneracyInfo degen_info = detectDegeneracy(hessian);

        if (degen_info.is_degenerate) {
            hessian = applyEigenvalueTruncation(hessian, degen_info);

            double degen_lambda = std::max(lambda, lm_params_.eigenvalue_threshold * 10.0);
            lambda = std::max(lambda, degen_lambda);

            degen_info.damping_lambda = lambda;
            last_degeneracy_ = degen_info;

            if (iter % 10 == 0) {
                std::cout << "[NDT] DEGENERATE iter=" << iter
                          << " degeneracy_dims=" << degen_info.degenerate_dimensions
                          << " min_eigenval=" << std::scientific << degen_info.min_eigenvalue
                          << " condition=" << degen_info.condition_number
                          << " lambda=" << lambda
                          << std::fixed << std::endl;
            }
        } else {
            last_degeneracy_ = degen_info;
        }

        double grad_norm = gradient.norm();
        if (grad_norm < 1e-8) {
            converged_ = true;
            break;
        }

        Eigen::Matrix<double, 6, 1> delta = solveLMStep(gradient, hessian, lambda);

        delta = constrainStep(delta);

        if (delta.norm() < 1e-10) {
            converged_ = true;
            break;
        }

        Eigen::Matrix4d new_transform = incrementTransformation(current_transform, delta);

        double new_score = computeScore(new_transform);

        if (new_score > current_score) {
            double score_improvement = std::abs(new_score - current_score)
                / (std::abs(current_score) + 1e-10);

            current_transform = new_transform;
            current_score = new_score;

            lambda = std::max(lambda / lm_params_.lambda_factor, lm_params_.lambda_min);
            stagnation_count = 0;

            if (score_improvement < lm_params_.convergence_delta) {
                converged_ = true;
                break;
            }
        } else {
            lambda = std::min(lambda * lm_params_.lambda_factor, lm_params_.lambda_max);
            stagnation_count++;

            if (stagnation_count >= max_stagnation) {
                if (current_score > 0) {
                    converged_ = true;
                }
                break;
            }
        }
    }

    if (!converged_ && current_score > 0) {
        converged_ = true;
    }

    final_transformation_ = current_transform;
    fitness_score_ = transformFitnessScore(current_transform);

    result.converged = converged_;
    result.fitness_score = fitness_score_;
    result.iterations = static_cast<size_t>(lm_params_.max_iterations);
    result.transformation = final_transformation_;
    result.degeneracy = last_degeneracy_;
    result.degeneracy.damping_lambda = lambda;

    return result;
}

double NDTRegistration::getFitnessScore() const {
    return fitness_score_;
}

bool NDTRegistration::hasConverged() const {
    return converged_;
}

Eigen::Matrix4d NDTRegistration::getFinalTransformation() const {
    return final_transformation_;
}

const DegeneracyInfo& NDTRegistration::getLastDegeneracyInfo() const {
    return last_degeneracy_;
}

} // namespace mine_slam
