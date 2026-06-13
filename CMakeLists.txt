cmake_minimum_required(VERSION 3.14)
project(mine_lidar_slam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_compile_options(/bigobj)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Debug")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
endif()

cmake_policy(SET CMP0167 NEW)

find_package(PCL 1.10 QUIET COMPONENTS
    common
    io
    filters
    registration
    octree
    kdtree
    search
    sample_consensus
)

if(NOT PCL_FOUND)
    message(STATUS "PCL not found. The project requires PCL 1.10+ to build.")
    message(STATUS "Please install PCL and set PCL_DIR or CMAKE_PREFIX_PATH.")
endif()

find_package(GTSAM QUIET)

if(NOT GTSAM_FOUND)
    message(STATUS "GTSAM not found. The project requires GTSAM to build.")
    message(STATUS "Please install GTSAM and set GTSAM_DIR or CMAKE_PREFIX_PATH.")
endif()

find_package(Eigen3 3.3 QUIET)

if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found. The project requires Eigen3 3.3+ to build.")
endif()

find_package(Boost QUIET COMPONENTS
    system
    filesystem
    program_options
    date_time
    thread
)

if(NOT Boost_FOUND)
    message(STATUS "Boost not found. The project requires Boost to build.")
endif()

set(HEADERS
    include/common_types.h
    include/point_cloud_source.h
    include/voxel_filter.h
    include/ndt_registration.h
    include/factor_graph_optimizer.h
    include/imu_preintegration.h
    include/map_builder.h
    include/config_parser.h
    include/timer.h
)

set(SOURCES
    src/point_cloud_source.cpp
    src/voxel_filter.cpp
    src/ndt_registration.cpp
    src/factor_graph_optimizer.cpp
    src/imu_preintegration.cpp
    src/map_builder.cpp
    src/config_parser.cpp
    src/main.cpp
)

if(PCL_FOUND AND GTSAM_FOUND AND Eigen3_FOUND AND Boost_FOUND)
    include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${PCL_INCLUDE_DIRS}
        ${GTSAM_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
    )

    link_directories(
        ${PCL_LIBRARY_DIRS}
        ${Boost_LIBRARY_DIRS}
    )

    add_definitions(${PCL_DEFINITIONS})

    add_executable(mine_lidar_slam ${SOURCES} ${HEADERS})

    target_link_libraries(mine_lidar_slam
        ${PCL_LIBRARIES}
        gtsam
        Eigen3::Eigen
        ${Boost_LIBRARIES}
    )

    if(MSVC)
        target_compile_options(mine_lidar_slam PRIVATE /W3)
    else()
        target_compile_options(mine_lidar_slam PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
            target_link_libraries(mine_lidar_slam stdc++fs)
        endif()
    endif()

    install(TARGETS mine_lidar_slam DESTINATION bin)
    message(STATUS "All dependencies found. Building mine_lidar_slam executable.")
else()
    add_custom_target(mine_lidar_slam SOURCES ${SOURCES} ${HEADERS})
    message(STATUS "============================================================")
    message(STATUS "  Some dependencies missing. Executable target will not be built.")
    message(STATUS "  Please install the following dependencies:")
    if(NOT PCL_FOUND)
        message(STATUS "    - PCL 1.10+ (Point Cloud Library)")
    endif()
    if(NOT GTSAM_FOUND)
        message(STATUS "    - GTSAM (Georgia Tech Smoothing and Mapping)")
    endif()
    if(NOT Eigen3_FOUND)
        message(STATUS "    - Eigen3 3.3+")
    endif()
    if(NOT Boost_FOUND)
        message(STATUS "    - Boost")
    endif()
    message(STATUS "============================================================")
endif()
