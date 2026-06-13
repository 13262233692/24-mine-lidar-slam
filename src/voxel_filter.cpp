#include "voxel_filter.h"
#include <pcl/octree/octree.h>
#include <iostream>

namespace mine_slam {

VoxelFilter::VoxelFilter()
    : leaf_x_(0.5)
    , leaf_y_(0.5)
    , leaf_z_(0.5)
    , use_octree_(true)
    , input_point_count_(0)
    , output_point_count_(0) {}

void VoxelFilter::setLeafSize(double leaf_x, double leaf_y, double leaf_z) {
    leaf_x_ = leaf_x;
    leaf_y_ = leaf_y;
    leaf_z_ = leaf_z;
}

void VoxelFilter::setLeafSize(double leaf_size) {
    leaf_x_ = leaf_size;
    leaf_y_ = leaf_size;
    leaf_z_ = leaf_size;
}

void VoxelFilter::setUseOctree(bool use_octree) {
    use_octree_ = use_octree;
}

bool VoxelFilter::getUseOctree() const {
    return use_octree_;
}

PointCloudPtr VoxelFilter::filter(PointCloudConstPtr input) {
    auto output = std::make_shared<PointCloud>();
    filter(input, *output);
    return output;
}

void VoxelFilter::filter(PointCloudConstPtr input, PointCloud& output) {
    input_point_count_ = input->size();

    if (use_octree_) {
        auto filtered = filterOctree(input);
        output = *filtered;
    } else {
        auto filtered = filterVoxelGrid(input);
        output = *filtered;
    }

    output_point_count_ = output.size();
}

double VoxelFilter::getReductionRatio() const {
    if (input_point_count_ == 0) return 0.0;
    return static_cast<double>(output_point_count_) /
           static_cast<double>(input_point_count_);
}

PointCloudPtr VoxelFilter::filterVoxelGrid(PointCloudConstPtr input) {
    auto output = std::make_shared<PointCloud>();

    pcl::VoxelGrid<PointT> voxel_grid;
    voxel_grid.setInputCloud(input);
    voxel_grid.setLeafSize(leaf_x_, leaf_y_, leaf_z_);
    voxel_grid.filter(*output);

    return output;
}

PointCloudPtr VoxelFilter::filterOctree(PointCloudConstPtr input) {
    auto output = std::make_shared<PointCloud>();

    double resolution = std::min({leaf_x_, leaf_y_, leaf_z_});

    pcl::octree::OctreePointCloudVoxelCentroid<PointT> octree(resolution);
    octree.setInputCloud(input);
    octree.addPointsFromInputCloud();

    output->header = input->header;
    output->width = 0;
    output->height = 1;
    output->is_dense = false;

    for (auto it = octree.leaf_depth_begin(); it != octree.leaf_depth_end(); ++it) {
        const auto& voxel_data = it.getLeafContainer();
        PointT centroid;
        centroid.x = 0;
        centroid.y = 0;
        centroid.z = 0;
        centroid.intensity = 0;

        std::vector<int> voxel_indices;
        voxel_data.getPointIndices(voxel_indices);

        if (voxel_indices.empty()) continue;

        float sum_intensity = 0.0f;
        for (int idx : voxel_indices) {
            const PointT& p = input->points[idx];
            centroid.x += p.x;
            centroid.y += p.y;
            centroid.z += p.z;
            centroid.intensity += p.intensity;
            sum_intensity += p.intensity;
        }

        float count = static_cast<float>(voxel_indices.size());
        centroid.x /= count;
        centroid.y /= count;
        centroid.z /= count;
        centroid.intensity /= count;

        output->points.push_back(centroid);
    }

    output->width = output->points.size();
    output->height = 1;
    output->is_dense = true;

    return output;
}

} // namespace mine_slam
