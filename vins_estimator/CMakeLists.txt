cmake_minimum_required(VERSION 2.8.3)
project(vins)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g ")

set(ENABLE_BACKWARD false)
set(ENABLE_VWORKS false)
set(ENABLE_CUDA false)


find_package(catkin REQUIRED COMPONENTS
    message_generation
    message_runtime
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    camera_models
    image_transport
    nodelet
    message_filters
    sensor_msgs
)


if(ENABLE_VWORKS)
    find_package(VisionWorks REQUIRED)
    #Set path to visionworks sample here
    # include_directories(
        # /home/xuhao/VisionWorks-1.6-Samples/nvxio/include
    # )
    # link_directories(/home/xuhao/VisionWorks-1.6-Samples/libs/x86_64/linux/release/)
else()
    add_definitions(-DWITHOUT_VWORKS)
endif(ENABLE_VWORKS)

if(ENABLE_BACKWARD)
    add_definitions(-D USE_BACKWARD)
endif()

if(ENABLE_CUDA)
    add_definitions(-D USE_CUDA)
endif()

#SET("OpenCV_DIR"  "/usr/local/share/OpenCV/")
find_package(OpenCV REQUIRED)

find_package(Ceres REQUIRED)


include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

include_directories(
    include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${VisionWorks_INCLUDE_DIRS}
)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    catkin libraries: ${catkin_LIBRARIES}")
#catkin_package()

add_message_files(
    FILES
    VIOKeyframe.msg
    FlattenImages.msg
)

generate_messages(
    DEPENDENCIES
    geometry_msgs
    std_msgs
    sensor_msgs
)

catkin_package()

add_library(stereo_depth
    src/depth_generation/depth_estimator.cpp
    src/depth_generation/stereo_matching.cpp
    src/depth_generation/depth_camera_manager.cpp
    src/depth_generation/color_disparity_graph.cpp
    src/depth_generation/stereo_online_calib.cpp
)

add_library(vins_frontend 
    src/featureTracker/feature_tracker.cpp
    src/featureTracker/vworks_feature_tracker.cpp
)


add_library(vins_factors_lib
    src/factor/pose_local_parameterization.cpp
    src/factor/projectionTwoFrameOneCamFactor.cpp
    src/factor/projectionTwoFrameTwoCamFactor.cpp
    src/factor/projectionOneFrameTwoCamFactor.cpp
)

add_library(vins_lib
    src/estimator/parameters.cpp
    src/estimator/feature_manager.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp

)

add_library(estimator_lib SHARED
    src/estimator/estimator.cpp
)

target_link_libraries(vins_lib ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} dw)
target_link_libraries(stereo_depth ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES}  ${VisionWorks_LIBRARIES} dw)
target_link_libraries(vins_frontend ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES}  ${VisionWorks_LIBRARIES} dw)


add_executable(vins_node src/rosNodeTest.cpp )
target_link_libraries(vins_node vins_lib estimator_lib vins_frontend stereo_depth vins_factors_lib)

add_library(vins_nodelet_lib src/rosNodelet.cpp)
target_link_libraries(vins_nodelet_lib vins_lib estimator_lib vins_frontend stereo_depth vins_factors_lib)

# add_executable(kitti_odom_test src/KITTIOdomTest.cpp)
# target_link_libraries(kitti_odom_test vins_lib estimator_lib)

# add_executable(kitti_gps_test src/KITTIGPSTest.cpp)
# target_link_libraries(kitti_gps_test vins_lib estimator_lib)

