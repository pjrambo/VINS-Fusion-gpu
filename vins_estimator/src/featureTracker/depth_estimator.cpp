#include "depth_estimator.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/opencv.hpp>
#include "../utility/tic_toc.h"
#include <opencv2/ximgproc/disparity_filter.hpp>

cv::Mat DepthEstimator::ComputeDispartiyMap(cv::cuda::GpuMat & left, cv::cuda::GpuMat & right, double & baseline) {
    // stereoRectify(InputArray cameraMatrix1, InputArray distCoeffs1, 
    // InputArray cameraMatrix2, InputArray distCoeffs2, 
    //Size imageSize, InputArray R, InputArray T, OutputArray R1, OutputArray R2, OutputArray P1, OutputArray P2, 
    //OutputArray Q,
    //  int flags=CALIB_ZERO_DISPARITY, double alpha=-1, 
    // Size newImageSize=Size(), Rect* validPixROI1=0, Rect* validPixROI2=0 )¶
    cv::Size imgSize = left.size();
    cv::Mat R,T, R1, R2, P1, P2, Q;
    cv::eigen2cv(R01, R);
    cv::eigen2cv(t01, T);

    // std::cout << "R init" << R << std::endl << "T" << T;
    cv::stereoRectify(cameraMatrix, cv::Mat(), cameraMatrix, cv::Mat(), imgSize, 
        R, T, R1, R2, P1, P2, Q, 0);
    // std::cout <<"Q:" << Q << std::endl;
    baseline = - 1.0/Q.at<double>(3, 2);
    // ROS_INFO("Stereo Rectify");
    // std::cout << R1 << P1 << std::endl;
    // std::cout << R2 << P2 << std::endl;
    // cv::remap
    cv::Mat _map11, _map12, _map21, _map22;
    initUndistortRectifyMap(cameraMatrix, cv::Mat(), R1, P1, imgSize, CV_32FC1, _map11,
                            _map12);
    initUndistortRectifyMap(cameraMatrix, cv::Mat(), R2, P2, imgSize, CV_32FC1, _map21,
                            _map22);
    TicToc tic;
    // std::cout << "Map Size 11" << _map11.size() << "12" << _map12.size() << std::endl;
    cv::cuda::GpuMat map11(_map11), map12(_map12), map21(_map21), map22(_map22);

    cv::cuda::GpuMat leftRectify, rightRectify, disparity(left.size(), CV_8U);
    cv::cuda::GpuMat leftRectify_Rotate;
    cv::cuda::GpuMat rightRectify_Rotate;

    cv::cuda::remap(left, leftRectify, map11, map12, cv::INTER_LINEAR);
    cv::cuda::remap(right, rightRectify, map21, map22, cv::INTER_LINEAR);

    cv::cuda::transpose(leftRectify, leftRectify_Rotate); 
    cv::cuda::transpose(rightRectify, rightRectify_Rotate);
     
    ROS_INFO("Remap %fms Baseline %f", tic.toc(), baseline);

    TicToc ticbm;
    cv::Ptr<cv::cuda::StereoBM> sbm = cv::cuda::createStereoBM(num_disp, 13);
    sbm->compute(leftRectify_Rotate, rightRectify_Rotate, disparity);
    cv::Mat leftRectifyCPU, rightRectifyCPU, disparityCPU, depthmap;
    leftRectify_Rotate.download(leftRectifyCPU);
    rightRectify_Rotate.download(rightRectifyCPU);
    disparity.download(disparityCPU);
    disparityCPU.convertTo(disparityCPU, CV_16S, 16);


    double min_val, max_val;
    cv::Mat scaled_disp_map;

    auto wls_filter_ = cv::ximgproc::createDisparityWLSFilter(sbm); // left_matcher
    wls_filter_->setLambda(8000.0);
    wls_filter_->setSigmaColor(1.1);
    // wls_filter_->setSigmaColor(0.8);

    auto right_matcher_ = cv::ximgproc::createRightMatcher(sbm);
    cv::Mat raw_right_disparity_map_, filtered_disparity_map_, filtered_disparity_map_8u_;
    right_matcher_->compute(rightRectifyCPU, leftRectifyCPU, raw_right_disparity_map_);

    wls_filter_->filter(disparityCPU,
                      leftRectifyCPU,
                      filtered_disparity_map_,
                      raw_right_disparity_map_);

    ROS_INFO("StereoBM %fms", ticbm.toc());
    cv::transpose(filtered_disparity_map_, filtered_disparity_map_);

    if (show) {
        
        filtered_disparity_map_.convertTo(filtered_disparity_map_8u_, CV_8UC1, 0.0625);
        cv::Mat  raw_disp_map = filtered_disparity_map_8u_.clone();

        cv::minMaxLoc(raw_disp_map, &min_val, &max_val, NULL, NULL);
        raw_disp_map.convertTo(scaled_disp_map, CV_8U, 255/(max_val-min_val), -min_val/(max_val-min_val));
        cv::imshow("Filtered", scaled_disp_map);


        cv::Mat _show;
        cv::transpose(leftRectifyCPU, leftRectifyCPU);
        cv::transpose(rightRectifyCPU, rightRectifyCPU);
        cv::vconcat(leftRectifyCPU, rightRectifyCPU, _show);
        cv::vconcat(_show, scaled_disp_map, _show);
        cv::imshow("Depth", _show);
        cv::waitKey(2);
    }

    return filtered_disparity_map_;
}

cv::Mat DepthEstimator::ComputeDepthImage(cv::cuda::GpuMat & left, cv::cuda::GpuMat & right) {
    double baseline = -1;
    cv::Mat dispartitymap = ComputeDispartiyMap(left, right, baseline);
    int width = left.size().width;
    int height = left.size().height;
    const int border_size = num_disp;
    const int trunc_img_width_end = width - border_size;
    const int trunc_img_height_end = height - border_size;
    cv::Size depth_img_size = left.size();
    depth_img_size.width = depth_img_size.width - border_size*2;
    depth_img_size.height = depth_img_size.height - border_size*2;
    cv::Mat depth_img(depth_img_size, CV_32FC1);
    double baseline_x_fx = cameraMatrix.at<double>(0, 0)*baseline;
    ROS_INFO("Focal length %f baseline %f", cameraMatrix.at<double>(0, 0), baseline);
    for(int v = border_size; v < trunc_img_height_end; ++v)
    {
        for(int u = border_size; u < trunc_img_width_end; ++u)
        {
            float disparity = (float)(dispartitymap.at<uint16_t>(v, u))*0.0625;
            if(disparity >= 1) {
                depth_img.at<float>(v - border_size, u - border_size) = baseline_x_fx / disparity;
            } else {
                depth_img.at<float>(v - border_size, u - border_size) = 0;
            }
            // printf("UV %d %d Depth %f Disp %f\n", u, v, depth_img.at<float>(v, u), disparity);
        }
    }
    return depth_img;
}
