#include "depth_estimator.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/opencv.hpp>
#include "../utility/tic_toc.h"
#include <opencv2/ximgproc/disparity_filter.hpp>
#include "../estimator/parameters.h"
#include <libsgm.h>
#include <NVX/nvx.h>
#include <NVX/nvx_opencv_interop.hpp>
#include "stereo_matching.hpp"
#include <OVX/UtilityOVX.hpp>

ovxio::ContextGuard context;

cv::Mat DepthEstimator::ComputeDispartiyMap(cv::cuda::GpuMat & left, cv::cuda::GpuMat & right) {
    // stereoRectify(InputArray cameraMatrix1, InputArray distCoeffs1, 
    // InputArray cameraMatrix2, InputArray distCoeffs2, 
    //Size imageSize, InputArray R, InputArray T, OutputArray R1, OutputArray R2, OutputArray P1, OutputArray P2, 
    //OutputArray Q,
    //  int flags=CALIB_ZERO_DISPARITY, double alpha=-1, 
    // Size newImageSize=Size(), Rect* validPixROI1=0, Rect* validPixROI2=0 )¶
    TicToc tic;
    if (first_init) {
        cv::Size imgSize = left.size();

        // std::cout << "ImgSize" << imgSize << "\nR" << R << "\nT" << T << std::endl;
        cv::stereoRectify(cameraMatrix, cv::Mat(), cameraMatrix, cv::Mat(), imgSize, 
            R, T, R1, R2, P1, P2, Q, 0);
        // Q.at<double>(3, 2) = -Q.at<double>(3, 2);
        std::cout << Q << std::endl;
        // std::cout << "R1" << R1 << "P1" << P1 << std::endl; 
        initUndistortRectifyMap(cameraMatrix, cv::Mat(), R1, P1, imgSize, CV_32FC1, _map11,
                                _map12);
        initUndistortRectifyMap(cameraMatrix, cv::Mat(), R2, P2, imgSize, CV_32FC1, _map21,
                                _map22);
        map11.upload(_map11);
        map12.upload(_map12);
        map21.upload(_map21);
        map22.upload(_map22);
        Q.convertTo(Q, CV_32F);

        first_init = false;
    } 

    cv::cuda::GpuMat leftRectify, rightRectify, disparity(left.size(), CV_8U);

    cv::cuda::remap(left, leftRectify, map11, map12, cv::INTER_LINEAR);
    cv::cuda::remap(right, rightRectify, map21, map22, cv::INTER_LINEAR);

    if (use_sgbm_cpu) {
        cv::Mat left_rect, right_rect, disparity;
        leftRectify.download(left_rect);
        rightRectify.download(right_rect);
        
        auto sgbm = cv::StereoSGBM::create(min_disparity, num_disp, block_size,
            _p1, _p2, disp12Maxdiff, prefilterCap, uniquenessRatio, speckleWindowSize, 
            speckleRange, mode);

        // sgbm->setBlockSize(block_size);
        // sgbm->setNumDisparities(num_disp);
        // sgbm->set
        // sgbm->setP
        sgbm->compute(left_rect, right_rect, disparity);
        // disparity.convertTo(disparity, CV_16S, 16);

        // std::cout << "DIS size " << disparity.size() << disparity.type() << std::endl;
        ROS_INFO("SGBM time cost %fms", tic.toc());
        if (show) {
            cv::Mat _show;
            cv::Mat raw_disp_map = disparity.clone();
            cv::Mat scaled_disp_map;
            double min_val, max_val;
            cv::minMaxLoc(raw_disp_map, &min_val, &max_val, NULL, NULL);
            raw_disp_map.convertTo(scaled_disp_map, CV_8U, 255/(max_val-min_val), -min_val/(max_val-min_val));
            
            // cv::transpose(left_rect, left_rect);
            // cv::transpose(right_rect, right_rect);
            // cv::transpose(scaled_disp_map, scaled_disp_map);

            cv::hconcat(left_rect, right_rect, _show);
            cv::hconcat(_show, scaled_disp_map, _show);
            // cv::hconcat(left_rect, right_rect, _show);
            // cv::hconcat(_show, scaled_disp_map, _show);
            cv::imshow("raw_disp_map", _show);
        }
        return disparity;
    }
    if (use_vworks) {
        vx_image vx_img_l = nvx_cv::createVXImageFromCVGpuMat(context, leftRectify);
        vx_image vx_img_r = nvx_cv::createVXImageFromCVGpuMat(context, rightRectify);
    } else {
        cv::Mat _disp;

    	sgm::LibSGMWrapper sgm(num_disp, _p1, _p2, uniquenessRatio, true, 
            sgm::PathType::SCAN_4PATH, min_disparity, disp12Maxdiff);
        // sgm::LibSGMWrapper sgm;
		sgm.execute(leftRectify, rightRectify, disparity);
		disparity.download(_disp);

        if (show) {
            cv::Mat _show, left_rect, right_rect;
            leftRectify.download(left_rect);
            rightRectify.download(right_rect);
    
            cv::Mat raw_disp_map = _disp.clone();
            cv::Mat scaled_disp_map;
            double min_val, max_val;
            cv::minMaxLoc(raw_disp_map, &min_val, &max_val, NULL, NULL);
            raw_disp_map.convertTo(scaled_disp_map, CV_8U, 255/(max_val-min_val), -min_val/(max_val-min_val));

            cv::hconcat(left_rect, right_rect, _show);
            cv::hconcat(_show, scaled_disp_map, _show);
            cv::imshow("RAW DISP", _show);
        }            
            
        ROS_INFO("SGBM time cost %fms", tic.toc());

        return _disp;
    }

   
}

cv::Mat DepthEstimator::ComputeDepthCloud(cv::cuda::GpuMat & left, cv::cuda::GpuMat & right) {
    cv::Mat dispartitymap = ComputeDispartiyMap(left, right);
    int width = left.size().width;
    int height = left.size().height;

    cv::Mat map3d, imgDisparity32F;
    dispartitymap.convertTo(imgDisparity32F, CV_32F, 1./16);
    cv::Mat XYZ = cv::Mat::zeros(imgDisparity32F.rows, imgDisparity32F.cols, CV_32FC3);   // Output point cloud
    cv::reprojectImageTo3D(imgDisparity32F, XYZ, Q);    // cv::project

    return XYZ;
}
