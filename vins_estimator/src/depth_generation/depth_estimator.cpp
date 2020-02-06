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


// ovxio::ContextGuard context;
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

    if (!params.use_vworks) {
        cv::Mat left_rect, right_rect, disparity;
        leftRectify.download(left_rect);
        rightRectify.download(right_rect);

        
        auto sgbm = cv::StereoSGBM::create(params.min_disparity, params.num_disp, params.block_size,
            params.p1, params.p2, params.disp12Maxdiff, params.prefilterCap, params.uniquenessRatio, params.speckleWindowSize, 
            params.speckleRange, params.mode);

        // sgbm->compute(right_rect, left_rect, disparity);
        sgbm->compute(left_rect, right_rect, disparity);

        ROS_INFO("SGBM time cost %fms", tic.toc());
        if (show) {
            cv::Mat _show;
            cv::Mat raw_disp_map = disparity.clone();
            cv::Mat scaled_disp_map;
            double min_val, max_val;
            cv::minMaxLoc(raw_disp_map, &min_val, &max_val, NULL, NULL);
            raw_disp_map.convertTo(scaled_disp_map, CV_8U, 255/(max_val-min_val), -min_val/(max_val-min_val));
            // cv::cvtColor(raw_disp_map, raw_disp_map, cv::COLOR_GRAY2BGR);
            
            // cv::transpose(left_rect, left_rect);
            // cv::transpose(right_rect, right_rect);
            // cv::transpose(scaled_disp_map, scaled_disp_map);

            cv::hconcat(left_rect, right_rect, _show);
            cv::hconcat(_show, scaled_disp_map, _show);
            // cv::hconcat(left_rect, right_rect, _show);
            // cv::hconcat(_show, scaled_disp_map, _show);
            cv::imshow("raw_disp_map", _show);
            cv::waitKey(2);
        }
        return disparity;
    } else {
        leftRectify.copyTo(leftRectify_fix);
        rightRectify.copyTo(rightRectify_fix);
        if (first_use_vworks) {
            auto lsize = leftRectify_fix.size();
            // disparity_fix = cv::cuda::GpuMat(leftRectify_fix.size(), CV_8U);
            // disparity_fix_cpu = cv::Mat(leftRectify_fix.size(), CV_8U);
            vxDirective(context, VX_DIRECTIVE_ENABLE_PERFORMANCE);

            vxRegisterLogCallback(context, &ovxio::stdoutLogCallback, vx_false_e);

            //
            // Create a NVXIO-based frame source
            //

            StereoMatching::ImplementationType implementationType = StereoMatching::HIGH_LEVEL_API;
            StereoMatching::StereoMatchingParams _params;
            _params.min_disparity = params.min_disparity;
            _params.max_disparity = params.num_disp;
            _params.P1 = params.p1;
            _params.P2 = params.p2;
            _params.uniqueness_ratio = params.uniquenessRatio;
            _params.max_diff = params.disp12Maxdiff;
            _params.bt_clip_value = params.bt_clip_value;
            _params.hc_win_size = params.hc_win_size;
            _params.flags = params.flags;
            _params.sad = params.block_size;
            _params.scanlines_mask = params.scanlines_mask;

            vx_img_l = nvx_cv::createVXImageFromCVGpuMat(context, leftRectify_fix);
            vx_img_r = nvx_cv::createVXImageFromCVGpuMat(context, rightRectify_fix);
            // vx_disparity = nvx_cv::createVXImageFromCVGpuMat(context, disparity_fix);
            vx_disparity = vxCreateImage(context, lsize.width, lsize.height, VX_DF_IMAGE_S16);
            vx_disparity_for_color = vxCreateImage(context, lsize.width, lsize.height, VX_DF_IMAGE_S16);

            vx_coloroutput = vxCreateImage(context, lsize.width, lsize.height, VX_DF_IMAGE_RGB);

            stereo = StereoMatching::createStereoMatching(
                context, _params,
                implementationType,
                vx_img_l, vx_img_r, vx_disparity);
            // stereo = StereoMatching::createStereoMatching(
            //     context, _params,
            //     implementationType,
            //     vx_img_r, vx_img_l, vx_disparity);
            first_use_vworks = false;
            color = new ColorDisparityGraph(context, vx_disparity_for_color, vx_coloroutput, params.num_disp);

        }


        stereo->run();
        cv::Mat cv_disp(leftRectify.size(), CV_8U);

        vx_uint32 plane_index = 0;
        vx_rectangle_t rect = {
            0u, 0u,
            leftRectify.size().width, leftRectify.size().height
        };

        if(show) {
            nvxuCopyImage(context, vx_disparity, vx_disparity_for_color);
        }

        nvx_cv::VXImageToCVMatMapper map(vx_disparity, plane_index, &rect, VX_WRITE_ONLY, NVX_MEMORY_TYPE_CUDA);
        auto _cv_disp_cuda = map.getGpuMat();
        _cv_disp_cuda.download(cv_disp);

        ROS_INFO("DISP %d %d!Time %fms", cv_disp.size().width, cv_disp.size().height, tic.toc());
        if (show) {
            cv::Mat color_disp;
            color->process();
            nvx_cv::VXImageToCVMatMapper map(vx_coloroutput, plane_index, &rect, VX_WRITE_ONLY, NVX_MEMORY_TYPE_CUDA);
            auto cv_disp_cuda = map.getGpuMat();
            cv_disp_cuda.download(color_disp);

            double min_val=0, max_val=0;
            cv::Mat gray_disp;
            cv::minMaxLoc(cv_disp, &min_val, &max_val, NULL, NULL);
            ROS_INFO("Min %f, max %f", min_val, max_val);
            cv_disp.convertTo(gray_disp, CV_8U, 1., 0);
            cv::cvtColor(gray_disp, gray_disp, cv::COLOR_GRAY2BGR);

            cv::Mat _show, left_rect, right_rect;
            leftRectify.download(left_rect);
            rightRectify.download(right_rect);
    
            cv::hconcat(left_rect, right_rect, _show);
            cv::cvtColor(_show, _show, cv::COLOR_GRAY2BGR);
            cv::hconcat(_show, gray_disp, _show);
            cv::hconcat(_show, color_disp, _show);
            cv::imshow("RAW DISP", _show);
            cv::waitKey(2);
        }            
        return cv_disp;

    }

    // if(false)
    // {
    //     cv::Mat _disp;

    // 	sgm::LibSGMWrapper sgm(num_disp, _p1, _p2, uniquenessRatio, true, 
    //         sgm::PathType::SCAN_4PATH, min_disparity, disp12Maxdiff);
    //     // sgm::LibSGMWrapper sgm;
	// 	sgm.execute(leftRectify, rightRectify, disparity);
	// 	disparity.download(_disp);

    //     if (show) {
    //         cv::Mat _show, left_rect, right_rect;
    //         leftRectify.download(left_rect);
    //         rightRectify.download(right_rect);
    
    //         cv::Mat raw_disp_map = _disp.clone();
    //         cv::Mat scaled_disp_map;
    //         double min_val, max_val;
    //         cv::minMaxLoc(raw_disp_map, &min_val, &max_val, NULL, NULL);
    //         raw_disp_map.convertTo(scaled_disp_map, CV_8U, 255/(max_val-min_val), -min_val/(max_val-min_val));

    //         cv::hconcat(left_rect, right_rect, _show);
    //         cv::hconcat(_show, scaled_disp_map, _show);
    //         cv::imshow("RAW DISP", _show);
    //     }            
            
    //     ROS_INFO("SGBM time cost %fms", tic.toc());

    //     return _disp;
    // }

   
}

cv::Mat DepthEstimator::ComputeDepthCloud(cv::cuda::GpuMat & left, cv::cuda::GpuMat & right) {
    cv::Mat dispartitymap = ComputeDispartiyMap(left, right);
    int width = left.size().width;
    int height = left.size().height;

    cv::Mat map3d, imgDisparity32F;
    if (params.use_vworks) {
        double min_val = 0;
        double max_val = 0;
        // dispartitymap.convertTo(imgDisparity32F, CV_32F, (params.num_disp-params.min_disparity)/255.0);
        // dispartitymap.convertTo(imgDisparity32F, CV_32F, (params.num_disp-params.min_disparity)/255.0);
        dispartitymap.convertTo(imgDisparity32F, CV_32F, 1./16);
        // dispartitymap.convertTo(imgDisparity32F, CV_32F, 1.0);
        cv::minMaxLoc(imgDisparity32F, &min_val, &max_val, NULL, NULL);
        ROS_INFO("Disp min %f max %f", min_val, max_val);
    } else {
        dispartitymap.convertTo(imgDisparity32F, CV_32F, 1./16);
    }
    cv::Mat XYZ = cv::Mat::zeros(imgDisparity32F.rows, imgDisparity32F.cols, CV_32FC3);   // Output point cloud
    cv::reprojectImageTo3D(imgDisparity32F, XYZ, Q);    // cv::project

    return XYZ;
}
