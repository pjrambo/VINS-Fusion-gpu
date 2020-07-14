#pragma once
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include "../utility/opencv_cuda.h"
#include <opencv2/core/eigen.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include "stereo_online_calib.hpp"

#include "color_disparity_graph.hpp"
#include "stereo_matching.hpp"
namespace sgm {
    class LibSGMWrapper;
};

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

class DepthEstimator {
    cv::Mat cameraMatrix;
    bool show = false;
    cv::Mat _map11, _map12, _map21, _map22;
#ifdef USE_CUDA
    cv::cuda::GpuMat map11, map12, map21, map22;
    sgm::LibSGMWrapper * sgmp;
#endif
    bool first_init = true;
    cv::Mat R, T, R1, R2, P1, P2, Q;
    double baseline = 0;
    
    SGMParams params;

    bool first_use_vworks = true;
    bool enable_extrinsic_calib = false;

    std::string output_path;
    double extrinsic_calib_rate = 1;
#ifdef WITH_VWORKS
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

    StereoOnlineCalib * online_calib = nullptr;

    std::vector<cv::Point2f> left_pts;
    std::vector<cv::Point2f> right_pts;

public:
    DepthEstimator(SGMParams _params, Eigen::Vector3d t01, Eigen::Matrix3d R01, cv::Mat camera_mat,
    bool _show, bool _enable_extrinsic_calib, std::string _output_path);

    DepthEstimator(SGMParams _params, std::string Path, cv::Mat camera_mat,
    bool _show, bool _enable_extrinsic_calib, std::string _output_path);

    cv::Mat ComputeDispartiyMap(cv::Mat & left, cv::Mat & right);
    cv::Mat ComputeDispartiyMap(cv::cuda::GpuMat & left, cv::cuda::GpuMat & right);

    template<typename cvMat>
    cv::Mat ComputeDepthCloud(cvMat & left, cvMat & right) {
        std::cout << "Computing depth cloud" << std::endl;
        static int count = 0;
        int skip = 10/extrinsic_calib_rate;
        if (skip <= 0) {
            skip = 1;
        }
        if (count ++ % 5 == 0 && enable_extrinsic_calib) {
            if (online_calib == nullptr) {
                online_calib = new StereoOnlineCalib(R, T, cameraMatrix, left.cols, left.rows, show);
            }
            
            bool success = online_calib->calibrate_extrincic(left, right);
            if (success) {
                R = online_calib->get_rotation();
                T = online_calib->get_translation();
                cv::FileStorage fs(output_path, cv::FileStorage::WRITE);
                fs << "R" << R;
                fs << "T" << T;
                fs.release();
                first_init = true;
            }
        }
        
        cv::Mat dispartitymap = ComputeDispartiyMap(left, right);
        int width = left.size().width;
        int height = left.size().height;

        cv::Mat map3d, imgDisparity32F;
        if (params.use_vworks) {
            double min_val = params.min_disparity;
            double max_val = 0;
            // dispartitymap.convertTo(imgDisparity32F, CV_32F, (params.num_disp-params.min_disparity)/255.0);
            // dispartitymap.convertTo(imgDisparity32F, CV_32F, (params.num_disp-params.min_disparity)/255.0);
            dispartitymap.convertTo(imgDisparity32F, CV_32F, 1./16);
            cv::threshold(imgDisparity32F, imgDisparity32F, min_val, 1000, cv::THRESH_TOZERO);
        } else {
            dispartitymap.convertTo(imgDisparity32F, CV_32F, 1./16);
            cv::threshold(imgDisparity32F, imgDisparity32F, params.min_disparity, 1000, cv::THRESH_TOZERO);
        }
        cv::Mat XYZ = cv::Mat::zeros(imgDisparity32F.rows, imgDisparity32F.cols, CV_32FC3);   // Output point cloud
        cv::reprojectImageTo3D(imgDisparity32F, XYZ, Q);    // cv::project

        return XYZ;
    }
};