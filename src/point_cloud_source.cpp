#include "point_cloud_source.h"
#include <pcl/io/pcd_io.h>
#include <filesystem>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace fs = std::filesystem;

namespace mine_slam {

PCDDirectorySource::PCDDirectorySource()
    : current_index_(0)
    , frame_counter_(0) {}

PCDDirectorySource::~PCDDirectorySource() {
    close();
}

bool PCDDirectorySource::open(const std::string& path) {
    if (!fs::is_directory(path)) {
        std::cerr << "[PCDSource] Path is not a directory: " << path << std::endl;
        return false;
    }

    pcd_files_.clear();
    current_index_ = 0;
    frame_counter_ = 0;

    if (!scanDirectory(path)) {
        return false;
    }

    std::sort(pcd_files_.begin(), pcd_files_.end());

    std::cout << "[PCDSource] Found " << pcd_files_.size()
              << " PCD files in " << path << std::endl;

    return !pcd_files_.empty();
}

void PCDDirectorySource::close() {
    pcd_files_.clear();
    current_index_ = 0;
}

bool PCDDirectorySource::hasNext() const {
    return current_index_ < pcd_files_.size();
}

LidarScanPtr PCDDirectorySource::next() {
    if (!hasNext()) {
        return nullptr;
    }

    auto scan = std::make_shared<LidarScan>();
    scan->frame_id = frame_counter_++;
    scan->cloud = std::make_shared<PointCloud>();

    const std::string& filename = pcd_files_[current_index_++];

    if (pcl::io::loadPCDFile<PointT>(filename, *scan->cloud) == -1) {
        std::cerr << "[PCDSource] Failed to load PCD: " << filename << std::endl;
        return nullptr;
    }

    fs::path file_path(filename);
    std::string stem = file_path.stem().string();

    try {
        std::string num_str;
        for (char c : stem) {
            if (std::isdigit(c)) {
                num_str += c;
            }
        }
        if (!num_str.empty()) {
            scan->timestamp = std::stoull(num_str);
        } else {
            scan->timestamp = scan->frame_id * 100000000ULL;
        }
    } catch (...) {
        scan->timestamp = scan->frame_id * 100000000ULL;
    }

    return scan;
}

size_t PCDDirectorySource::totalFrames() const {
    return pcd_files_.size();
}

size_t PCDDirectorySource::currentFrame() const {
    return current_index_;
}

void PCDDirectorySource::reset() {
    current_index_ = 0;
    frame_counter_ = 0;
}

bool PCDDirectorySource::scanDirectory(const std::string& dir_path) {
    try {
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".pcd") {
                    pcd_files_.push_back(entry.path().string());
                }
            }
        }
        return true;
    } catch (const fs::filesystem_error& e) {
        std::cerr << "[PCDSource] Directory scan error: " << e.what() << std::endl;
        return false;
    }
}

PointCloudSourcePtr PointCloudSourceFactory::create(DataSourceType type) {
    switch (type) {
        case DataSourceType::PCD_DIR:
            return std::make_shared<PCDDirectorySource>();
        case DataSourceType::ROS_BAG:
            return nullptr;
        default:
            return nullptr;
    }
}

PointCloudSourcePtr PointCloudSourceFactory::createFromPath(const std::string& path) {
    if (fs::is_directory(path)) {
        return std::make_shared<PCDDirectorySource>();
    }

    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".bag") {
        return nullptr;
    }

    return std::make_shared<PCDDirectorySource>();
}

} // namespace mine_slam
