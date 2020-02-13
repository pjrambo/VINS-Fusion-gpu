#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <opencv2/cudastereo.hpp>

#ifndef WITHOUT_VWORKS
#include <NVX/nvx.h>
#include <NVX/nvx_opencv_interop.hpp>
#include "stereo_matching.hpp"
#include "color_disparity_graph.hpp"
#include <NVX/nvx.h>
#include <NVX/nvx_opencv_interop.hpp>
#endif

struct SGMParams {
    bool use_vworks = true;
    int num_disp = 32;
    int block_size = 9;
    int min_disparity = 0;
    int disp12Maxdiff = 28;
    int prefilterCap = 39;
    int prefilterSize = 5;
    int uniquenessRatio = 25;
    int speckleWindowSize = 300;
    int speckleRange = 5;
    int mode = cv::StereoSGBM::MODE_HH;
    int p1 = 8;
    int p2 = 109;
    int ct_win_size = 0;
    int hc_win_size = 1;
    int bt_clip_value = 31;
    int scanlines_mask = 85;
    int flags = 1;
};

class StereoOnlineCalib;

class DepthEstimator {
    cv::Mat cameraMatrix;
    bool show = false;
    cv::Mat _map11, _map12, _map21, _map22;
    cv::cuda::GpuMat map11, map12, map21, map22;
    bool first_init = true;
    cv::Mat R, T, R1, R2, P1, P2, Q;
    double baseline = 0;
    
    SGMParams params;

    bool first_use_vworks = true;
    bool enable_extrinsic_calib = false;

    std::string output_path;
    double extrinsic_calib_rate = 1;
#ifndef WITHOUT_VWORKS
    vx_image vx_img_l;
    vx_image vx_img_r;
    vx_image vx_disparity;
    vx_image vx_disparity_for_color;
    vx_image vx_coloroutput;
    cv::cuda::GpuMat leftRectify_fix, rightRectify_fix, disparity_fix;
    cv::Mat disparity_fix_cpu;
    StereoMatching * stereo;
    ColorDisparityGraph * color;
#endif

    StereoOnlineCalib * online_calib;

    std::vector<cv::Point2f> left_pts;
    std::vector<cv::Point2f> right_pts;

public:
    DepthEstimator(SGMParams _params, Eigen::Vector3d t01, Eigen::Matrix3d R01, cv::Mat camera_mat,
    bool _show, bool _enable_extrinsic_calib, std::string _output_path);

    DepthEstimator(SGMParams _params, std::string Path, cv::Mat camera_mat,
    bool _show, bool _enable_extrinsic_calib, std::string _output_path);

    cv::Mat ComputeDispartiyMap(cv::cuda::GpuMat & left, cv::cuda::GpuMat & right);
    cv::Mat ComputeDepthCloud(cv::cuda::GpuMat & left, cv::cuda::GpuMat & right);
};