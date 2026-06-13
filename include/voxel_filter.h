#pragma once

#include "common_types.h"
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/filters/voxel_grid.h>
#include <memory>

namespace mine_slam {

class VoxelFilter {
public:
    VoxelFilter();
    ~VoxelFilter() = default;

    void setLeafSize(double leaf_x, double leaf_y, double leaf_z);
    void setLeafSize(double leaf_size);

    void setUseOctree(bool use_octree);
    bool getUseOctree() const;

    PointCloudPtr filter(PointCloudConstPtr input);

    void filter(PointCloudConstPtr input, PointCloud& output);

    size_t getInputPointCount() const { return input_point_count_; }
    size_t getOutputPointCount() const { return output_point_count_; }
    double getReductionRatio() const;

private:
    PointCloudPtr filterVoxelGrid(PointCloudConstPtr input);
    PointCloudPtr filterOctree(PointCloudConstPtr input);

    double leaf_x_;
    double leaf_y_;
    double leaf_z_;
    bool use_octree_;
    size_t input_point_count_;
    size_t output_point_count_;
};

using VoxelFilterPtr = std::shared_ptr<VoxelFilter>;

} // namespace mine_slam
