#pragma once

#include "common_types.h"
#include <pcl/registration/ndt.h>
#include <pcl/search/kdtree.h>
#include <memory>

namespace mine_slam {

class NDTRegistration {
public:
    NDTRegistration();
    ~NDTRegistration() = default;

    void setResolution(double resolution);
    void setStepSize(double step_size);
    void setMaxIterations(int max_iter);
    void setTransformationEpsilon(double epsilon);

    void setInputTarget(PointCloudConstPtr target);
    void setInputSource(PointCloudConstPtr source);

    RegistrationResult align();
    RegistrationResult align(const Eigen::Matrix4d& initial_guess);

    double getFitnessScore() const;
    bool hasConverged() const;

    Eigen::Matrix4d getFinalTransformation() const;

private:
    pcl::NormalDistributionsTransform<PointT, PointT> ndt_;
    PointCloudConstPtr target_cloud_;
    PointCloudConstPtr source_cloud_;
    double resolution_;
    double step_size_;
    int max_iterations_;
    double transformation_epsilon_;
};

using NDTRegistrationPtr = std::shared_ptr<NDTRegistration>;

} // namespace mine_slam
