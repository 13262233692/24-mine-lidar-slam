#pragma once

#include "common_types.h"
#include <memory>
#include <string>
#include <vector>

namespace mine_slam {

class PointCloudSource {
public:
    virtual ~PointCloudSource() = default;

    virtual bool open(const std::string& path) = 0;
    virtual void close() = 0;

    virtual bool hasNext() const = 0;
    virtual LidarScanPtr next() = 0;

    virtual size_t totalFrames() const = 0;
    virtual size_t currentFrame() const = 0;

    virtual void reset() = 0;
};

using PointCloudSourcePtr = std::shared_ptr<PointCloudSource>;

class IMUDataSource {
public:
    virtual ~IMUDataSource() = default;

    virtual bool open(const std::string& path) = 0;
    virtual void close() = 0;

    virtual bool loadAll() = 0;

    virtual std::vector<IMUMeasurement> getMeasurementsBetween(
        uint64_t start_ts, uint64_t end_ts) const = 0;

    virtual const std::vector<IMUMeasurement>& getAllMeasurements() const = 0;

    virtual size_t totalMeasurements() const = 0;

    virtual bool hasTimestamp(uint64_t ts) const = 0;
};

using IMUDataSourcePtr = std::shared_ptr<IMUDataSource>;

class PCDDirectorySource : public PointCloudSource {
public:
    PCDDirectorySource();
    ~PCDDirectorySource() override;

    bool open(const std::string& path) override;
    void close() override;

    bool hasNext() const override;
    LidarScanPtr next() override;

    size_t totalFrames() const override;
    size_t currentFrame() const override;

    void reset() override;

private:
    bool scanDirectory(const std::string& dir_path);

    std::vector<std::string> pcd_files_;
    size_t current_index_;
    size_t frame_counter_;
};

class CSVIMUDataSource : public IMUDataSource {
public:
    CSVIMUDataSource();
    ~CSVIMUDataSource() override;

    bool open(const std::string& path) override;
    void close() override;

    bool loadAll() override;

    std::vector<IMUMeasurement> getMeasurementsBetween(
        uint64_t start_ts, uint64_t end_ts) const override;

    const std::vector<IMUMeasurement>& getAllMeasurements() const override;

    size_t totalMeasurements() const override;

    bool hasTimestamp(uint64_t ts) const override;

private:
    std::vector<IMUMeasurement> measurements_;
    bool loaded_;
};

class PointCloudSourceFactory {
public:
    static PointCloudSourcePtr create(DataSourceType type);
    static PointCloudSourcePtr createFromPath(const std::string& path);
    static IMUDataSourcePtr createIMUSource();
};

} // namespace mine_slam
