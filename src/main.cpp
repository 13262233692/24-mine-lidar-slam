#include "common_types.h"
#include "config_parser.h"
#include "map_builder.h"
#include "timer.h"
#include <iostream>
#include <memory>
#include <csignal>

using namespace mine_slam;

std::shared_ptr<MapBuilder> g_map_builder = nullptr;

void signalHandler(int signum) {
    std::cout << "\n[Main] Received signal " << signum << ", stopping..." << std::endl;
    if (g_map_builder) {
        g_map_builder->stop();
    }
}

int main(int argc, char** argv) {
    std::cout << "==============================================" << std::endl;
    std::cout << "  Mine Lidar SLAM - High Precision Mapping" << std::endl;
    std::cout << "  Open-pit Mining Autonomous Truck System" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << std::endl;

    signal(SIGINT, signalHandler);
#ifdef SIGTERM
    signal(SIGTERM, signalHandler);
#endif

    SLAMConfig config;
    ConfigParser parser;

    if (!parser.parseCommandLine(argc, argv, config)) {
        return 1;
    }

    parser.printConfig(config);
    std::cout << std::endl;

    g_map_builder = std::make_shared<MapBuilder>();

    Timer init_timer("Initialization");
    if (!g_map_builder->initialize(config)) {
        std::cerr << "[Main] Failed to initialize SLAM system" << std::endl;
        return 1;
    }
    init_timer.print("Initialization");

    std::cout << std::endl;
    std::cout << "[Main] Starting SLAM processing pipeline..." << std::endl;
    std::cout << std::endl;

    try {
        g_map_builder->run();
    } catch (const std::exception& e) {
        std::cerr << "[Main] Exception during SLAM processing: "
                  << e.what() << std::endl;
        return 1;
    }

    std::cout << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "  SLAM Processing Complete" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "  Frames processed: " << g_map_builder->processedFrames() << std::endl;
    std::cout << "  Keyframes created: " << g_map_builder->keyFrameCount() << std::endl;
    std::cout << "  Output directory: " << config.output_dir << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
