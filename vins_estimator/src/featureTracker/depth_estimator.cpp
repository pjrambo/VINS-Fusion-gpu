#include "depth_estimator.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/opencv.hpp>
#include "../utility/tic_toc.h"
#include <opencv2/ximgproc/disparity_filter.hpp>
#include "../estimator/parameters.h"

cv::Mat DepthEstimator::ComputeDispartiyMap(cv::cuda::GpuMat & left, cv::cuda::GpuMat & right) {
    // stereoRectify(InputArray cameraMatrix1, InputArray distCoeffs1, 
    // InputArray cameraMatrix2, InputArray distCoeffs2, 
    //Size imageSize, InputArray R, InputArray T, OutputArray R1, OutputArray R2, OutputArray P1, OutputArray P2, 
    //OutputArray Q,
    //  int flags=CALIB_ZERO_DISPARITY, double alpha=-1, 
    // Size newImageSize=Size(), Rect* validPixROI1=0, Rect* validPixROI2=0 )¶

    if (first_init) {
        cv::Size imgSize = left.size();

        cv::stereoRectify(cameraMatrix, cv::Mat(), cameraMatrix, cv::Mat(), imgSize, 
            R, T, R1, R2, P1, P2, Q, 0);
    
        baseline = - 1.0/Q.at<double>(3, 2);
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
    cv::cuda::GpuMat leftRectify_Rotate;
    cv::cuda::GpuMat rightRectify_Rotate;

    cv::cuda::remap(left, leftRectify, map11, map12, cv::INTER_LINEAR);
    cv::cuda::remap(right, rightRectify, map21, map22, cv::INTER_LINEAR);


    cv::cuda::transpose(leftRectify, leftRectify_Rotate); 
    cv::cuda::transpose(rightRectify, rightRectify_Rotate);

    if (use_sgbm) {
        cv::Mat left_rect, right_rect, disparity;
        leftRectify_Rotate.download(left_rect);
        rightRectify_Rotate.download(right_rect);
        auto sgbm = cv::StereoSGBM::create(0, num_disp);
        sgbm->setPreFilterCap(25);
        sgbm->setUniquenessRatio(10);
        sgbm->setSpeckleWindowSize(100);
        sgbm->setSpeckleRange(2);
        sgbm->setDisp12MaxDiff(1);
        sgbm->setMode(cv::StereoSGBM::MODE_HH);
        // sgbm.fullDP = false;

        sgbm->setP1(p1);
        sgbm->setP2(p2);
        sgbm->compute(left_rect, right_rect, disparity);
        // disparity.convertTo(disparity, CV_16S, 16);

        cv::transpose(disparity, disparity);
        std::cout << "DIS size " << disparity.size() << disparity.type() << std::endl;

        if (show) {
            cv::Mat _show;
            cv::Mat raw_disp_map = disparity.clone();
            cv::Mat scaled_disp_map;
            double min_val, max_val;
            cv::minMaxLoc(raw_disp_map, &min_val, &max_val, NULL, NULL);
            raw_disp_map.convertTo(scaled_disp_map, CV_8U, 255/(max_val-min_val), -min_val/(max_val-min_val));
            cv::transpose(left_rect, left_rect);
            cv::transpose(right_rect, right_rect);
            cv::vconcat(left_rect, right_rect, _show);
            cv::vconcat(_show, scaled_disp_map, _show);
            cv::imshow("raw_disp_map", _show);
        }
        return disparity;
    } else {

        TicToc tic;
        // std::cout << "Map Size 11" << _map11.size() << "12" << _map12.size() << std::endl;

        
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

   
}


cv::Vec3f compute_3D_world_coordinates(int row, int col, const cv::Mat & Q, cv::Mat disparity){

    cv::Mat_<float> vec(4,1);

    vec(0) = col;
    vec(1) = row;
    vec(2) = disparity.at<float>(row,col);

    // Discard points with 0 disparity    
    if(vec(2)==0) return NULL;
    vec(3)=1;              
    vec = Q*vec;
    vec /= vec(3);
    // Discard points that are too far from the camera, and thus are highly
    // unreliable
    if(abs(vec(0))>10 || abs(vec(1))>10 || abs(vec(2))>10) return NULL;

    cv::Vec3f point3f;
    (point3f)[0] = vec(0);
    (point3f)[1] = vec(1);
    (point3f)[2] = vec(2);

    return point3f;
}
cv::Mat DepthEstimator::ComputeDepthImage(cv::cuda::GpuMat & left, cv::cuda::GpuMat & right) {
    cv::Mat dispartitymap = ComputeDispartiyMap(left, right);
    int width = left.size().width;
    int height = left.size().height;
    // const int border_size = num_disp;
    // const int trunc_img_width_end = width;
    // const int trunc_img_height_end = height - border_size;
    // cv::Size depth_img_size = left.size();
    // depth_img_size.width = depth_img_size.width;
    // depth_img_size.height = depth_img_size.height - border_size*2;
    // cv::Mat depth_img(depth_img_size, CV_32FC1);
    // double baseline_x_fx = cameraMatrix.at<double>(0, 0)*baseline;
    // ROS_INFO("Focal length %f baseline %f", cameraMatrix.at<double>(0, 0), baseline);
    // for(int v = border_size; v < trunc_img_height_end; ++v)
    // {
    //     for(int u = 0; u < trunc_img_width_end; ++u)
    //     {   
            
    //         int disparity = 0;
    //         if(use_sgbm ) {
    //             disparity = (float)(dispartitymap.at<uint16_t>(v, u));
    //         } else {
    //             disparity = (float)(dispartitymap.at<uint16_t>(v, u));
    //         }

    //         if(disparity >= 6) {
    //             depth_img.at<float>(v - border_size, u) = baseline_x_fx * 16 / disparity;
    //         } else {
    //             depth_img.at<float>(v - border_size, u) = 0;
    //         }
    //         // printf("UV %d %d Depth %f Disp %f\n", u, v, depth_img.at<float>(v - border_size, u), disparity);
    //     }
    // }
    cv::Mat map3d, imgDisparity32F;
    dispartitymap.convertTo(imgDisparity32F, CV_32F, 1./16);
    cv::Mat_<cv::Vec3f> XYZ(imgDisparity32F.rows ,imgDisparity32F.cols);   // Output point cloud
    cv::Mat_<float> vec_tmp(4,1);
    std::cout << "Q" << Q << std::endl;
    for(int y=0; y<imgDisparity32F.rows; ++y) {
        for(int x=0; x<imgDisparity32F.cols; ++x) {
            vec_tmp(0)=x; vec_tmp(1)=y; vec_tmp(2)=imgDisparity32F.at<float>(y,x); vec_tmp(3)=1;
            vec_tmp = Q*vec_tmp;
            vec_tmp /= vec_tmp(3);
            cv::Vec3f &point = XYZ.at<cv::Vec3f>(y,x);
            point[0] = vec_tmp(0);
            point[1] = vec_tmp(1);
            point[2] = vec_tmp(2);

            // point = compute_3D_world_coordinates(y, x, Q, imgDisparity32F);
        }
    }
    return XYZ;
}
