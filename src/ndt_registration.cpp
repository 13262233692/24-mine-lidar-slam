#include "ndt_registration.h"
#include <iostream>

namespace mine_slam {

NDTRegistration::NDTRegistration()
    : resolution_(2.0)
    , step_size_(0.1)
    , max_iterations_(35)
    , transformation_epsilon_(0.01) {
    ndt_.setResolution(resolution_);
    ndt_.setStepSize(step_size_);
    ndt_.setMaximumIterations(max_iterations_);
    ndt_.setTransformationEpsilon(transformation_epsilon_);
}

void NDTRegistration::setResolution(double resolution) {
    resolution_ = resolution;
    ndt_.setResolution(resolution_);
}

void NDTRegistration::setStepSize(double step_size) {
    step_size_ = step_size;
    ndt_.setStepSize(step_size_);
}

void NDTRegistration::setMaxIterations(int max_iter) {
    max_iterations_ = max_iter;
    ndt_.setMaximumIterations(max_iterations_);
}

void NDTRegistration::setTransformationEpsilon(double epsilon) {
    transformation_epsilon_ = epsilon;
    ndt_.setTransformationEpsilon(transformation_epsilon_);
}

void NDTRegistration::setInputTarget(PointCloudConstPtr target) {
    target_cloud_ = target;
    ndt_.setInputTarget(target);
}

void NDTRegistration::setInputSource(PointCloudConstPtr source) {
    source_cloud_ = source;
    ndt_.setInputSource(source);
}

RegistrationResult NDTRegistration::align() {
    auto output = std::make_shared<PointCloud>();
    ndt_.align(*output);

    RegistrationResult result;
    result.converged = ndt_.hasConverged();
    result.fitness_score = ndt_.getFitnessScore();
    result.iterations = static_cast<size_t>(ndt_.getFinalNumIteration());
    result.transformation = ndt_.getFinalTransformation().template cast<double>();

    return result;
}

RegistrationResult NDTRegistration::align(const Eigen::Matrix4d& initial_guess) {
    auto output = std::make_shared<PointCloud>();
    Eigen::Matrix4f guess = initial_guess.cast<float>();
    ndt_.align(*output, guess);

    RegistrationResult result;
    result.converged = ndt_.hasConverged();
    result.fitness_score = ndt_.getFitnessScore();
    result.iterations = static_cast<size_t>(ndt_.getFinalNumIteration());
    result.transformation = ndt_.getFinalTransformation().template cast<double>();

    return result;
}

double NDTRegistration::getFitnessScore() const {
    return ndt_.getFitnessScore();
}

bool NDTRegistration::hasConverged() const {
    return ndt_.hasConverged();
}

Eigen::Matrix4d NDTRegistration::getFinalTransformation() const {
    return ndt_.getFinalTransformation().template cast<double>();
}

} // namespace mine_slam
