#include "config_parser.h"
#include <boost/program_options.hpp>
#include <iostream>
#include <filesystem>
#include <algorithm>

namespace po = boost::program_options;
namespace fs = std::filesystem;

namespace mine_slam {

bool ConfigParser::parseCommandLine(int argc, char** argv, SLAMConfig& config) {
    try {
        po::options_description desc("Mine Lidar SLAM Options");
        desc.add_options()
            ("help,h", "produce help message")
            ("input,i", po::value<std::string>(&config.input_path)->required(),
             "input path (PCD directory or ROS bag file)")
            ("topic,t", po::value<std::string>(&config.point_cloud_topic),
             "point cloud topic name (for ROS bag)")
            ("output,o", po::value<std::string>(&config.output_dir),
             "output directory")
            ("voxel-leaf", po::value<double>(&config.voxel_leaf_size),
             "voxel filter leaf size (meters)")
            ("ndt-res", po::value<double>(&config.ndt_resolution),
             "NDT resolution (meters)")
            ("ndt-step", po::value<double>(&config.ndt_step_size),
             "NDT step size")
            ("ndt-iter", po::value<int>(&config.ndt_max_iterations),
             "NDT max iterations")
            ("lm-lambda", po::value<double>(&config.lm_lambda_init),
             "L-M initial damping lambda")
            ("lm-factor", po::value<double>(&config.lm_lambda_factor),
             "L-M lambda increase/decrease factor")
            ("lm-eigen-thresh", po::value<double>(&config.lm_eigenvalue_threshold),
             "L-M eigenvalue singularity threshold")
            ("lm-max-trans", po::value<double>(&config.lm_max_translation_step),
             "L-M max translation step (meters)")
            ("lm-max-rot", po::value<double>(&config.lm_max_rotation_step),
             "L-M max rotation step (radians)")
            ("kf-trans", po::value<double>(&config.keyframe_translation_threshold),
             "keyframe translation threshold (meters)")
            ("kf-rot", po::value<double>(&config.keyframe_rotation_threshold),
             "keyframe rotation threshold (radians)")
            ("loop-detect", po::value<int>(&config.loop_detection_enabled),
             "enable loop closure detection (0 or 1)")
            ("save-map", po::value<bool>(&config.save_map),
             "save final point cloud map")
            ("save-traj", po::value<bool>(&config.save_trajectory),
             "save trajectory file")
        ;

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return false;
        }

        po::notify(vm);

        config.source_type = detectSourceType(config.input_path);

        if (config.source_type == DataSourceType::UNKNOWN) {
            std::cerr << "Error: Cannot determine data source type from path: "
                      << config.input_path << std::endl;
            return false;
        }

        if (!fs::exists(config.output_dir)) {
            fs::create_directories(config.output_dir);
        }

        return true;
    } catch (const po::error& e) {
        std::cerr << "Command line error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

void ConfigParser::printConfig(const SLAMConfig& config) const {
    std::cout << "========== SLAM Configuration ==========" << std::endl;
    std::cout << "Source type: ";
    switch (config.source_type) {
        case DataSourceType::PCD_DIR: std::cout << "PCD Directory"; break;
        case DataSourceType::ROS_BAG: std::cout << "ROS Bag"; break;
        default: std::cout << "Unknown"; break;
    }
    std::cout << std::endl;
    std::cout << "Input path: " << config.input_path << std::endl;
    std::cout << "Point cloud topic: " << config.point_cloud_topic << std::endl;
    std::cout << "Output directory: " << config.output_dir << std::endl;
    std::cout << std::endl;
    std::cout << "Voxel leaf size: " << config.voxel_leaf_size << " m" << std::endl;
    std::cout << std::endl;
    std::cout << "NDT resolution: " << config.ndt_resolution << " m" << std::endl;
    std::cout << "NDT step size: " << config.ndt_step_size << std::endl;
    std::cout << "NDT max iterations: " << config.ndt_max_iterations << std::endl;
    std::cout << std::endl;
    std::cout << "L-M initial lambda: " << config.lm_lambda_init << std::endl;
    std::cout << "L-M lambda factor: " << config.lm_lambda_factor << std::endl;
    std::cout << "L-M eigenvalue threshold: " << config.lm_eigenvalue_threshold << std::endl;
    std::cout << "L-M max translation step: " << config.lm_max_translation_step << " m" << std::endl;
    std::cout << "L-M max rotation step: " << config.lm_max_rotation_step << " rad" << std::endl;
    std::cout << "L-M covariance regularization: " << config.lm_covariance_regularization << std::endl;
    std::cout << "L-M min cell points: " << config.lm_min_cell_points << std::endl;
    std::cout << std::endl;
    std::cout << "Keyframe translation threshold: "
              << config.keyframe_translation_threshold << " m" << std::endl;
    std::cout << "Keyframe rotation threshold: "
              << config.keyframe_rotation_threshold << " rad" << std::endl;
    std::cout << std::endl;
    std::cout << "Loop detection enabled: "
              << (config.loop_detection_enabled ? "yes" : "no") << std::endl;
    std::cout << "========================================" << std::endl;
}

DataSourceType ConfigParser::detectSourceType(const std::string& path) {
    if (!fs::exists(path)) {
        return DataSourceType::UNKNOWN;
    }

    if (fs::is_directory(path)) {
        std::vector<std::string> pcd_exts = {".pcd", ".PCD"};
        for (const auto& entry : fs::directory_iterator(path)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (std::find(pcd_exts.begin(), pcd_exts.end(), ext) != pcd_exts.end()) {
                    return DataSourceType::PCD_DIR;
                }
            }
        }
        return DataSourceType::UNKNOWN;
    }

    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".bag") {
        return DataSourceType::ROS_BAG;
    }

    return DataSourceType::UNKNOWN;
}

} // namespace mine_slam
