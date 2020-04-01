#include "depth_estimator.h"
#include <opencv2/calib3d.hpp>

#ifdef USE_CUDA
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudafeatures2d.hpp>
#endif

#include <opencv2/opencv.hpp>
#include "../utility/tic_toc.h"
#include "../estimator/parameters.h"
#include "stereo_online_calib.hpp"

#ifndef WITHOUT_VWORKS
#ifdef OVX
extern ovxio::ContextGuard context;
#else 
extern vx_context context;
#endif
#endif



DepthEstimator::DepthEstimator(SGMParams _params, Eigen::Vector3d t01, Eigen::Matrix3d R01, cv::Mat camera_mat,
bool _show, bool _enable_extrinsic_calib, std::string _output_path):
    cameraMatrix(camera_mat.clone()),show(_show),params(_params),
    enable_extrinsic_calib(_enable_extrinsic_calib),output_path(_output_path)
{
    cv::eigen2cv(R01, R);
    cv::eigen2cv(t01, T);
}

DepthEstimator::DepthEstimator(SGMParams _params, std::string Path, cv::Mat camera_mat,
bool _show, bool _enable_extrinsic_calib, std::string _output_path):
    cameraMatrix(camera_mat.clone()),show(_show),params(_params),
    enable_extrinsic_calib(_enable_extrinsic_calib),output_path(_output_path)
{
    cv::FileStorage fsSettings(Path, cv::FileStorage::READ);
    ROS_INFO("Stereo read RT from %s", Path.c_str());
    fsSettings["R"] >> R;
    fsSettings["T"] >> T;
    fsSettings.release();
}
    


cv::Mat DepthEstimator::ComputeDispartiyMap(cv::Mat & left, cv::Mat & right) {
    // stereoRectify(InputArray cameraMatrix1, InputArray distCoeffs1, 
    // InputArray cameraMatrix2, InputArray distCoeffs2, 
    //Size imageSize, InputArray R, InputArray T, OutputArray R1, OutputArray R2, OutputArray P1, OutputArray P2, 
    //OutputArray Q,
    //  int flags=CALIB_ZERO_DISPARITY, double alpha=-1, 
    // Size newImageSize=Size(), Rect* validPixROI1=0, Rect* validPixROI2=0 )¶
    TicToc tic;
    if (first_init) {
        cv::Mat _Q;
        cv::Size imgSize = left.size();

        // std::cout << "ImgSize" << imgSize << "\nR" << R << "\nT" << T << std::endl;
        cv::stereoRectify(cameraMatrix, cv::Mat(), cameraMatrix, cv::Mat(), imgSize, 
            R, T, R1, R2, P1, P2, _Q, 0);
        std::cout << Q << std::endl;
        initUndistortRectifyMap(cameraMatrix, cv::Mat(), R1, P1, imgSize, CV_32FC1, _map11,
                                _map12);
        initUndistortRectifyMap(cameraMatrix, cv::Mat(), R2, P2, imgSize, CV_32FC1, _map21,
                                _map22);
#ifdef USE_CUDA
        map11.upload(_map11);
        map12.upload(_map12);
        map21.upload(_map21);
        map22.upload(_map22);
#endif
        _Q.convertTo(Q, CV_32F);

        first_init = false;
    } 


#ifdef USE_CUDA
    cv::cuda::GpuMat leftRectify, rightRectify, disparity(left.size(), CV_8U);
    cv::cuda::remap(left, leftRectify, map11, map12, cv::INTER_LINEAR);
    cv::cuda::remap(right, rightRectify, map21, map22, cv::INTER_LINEAR);
#else
    cv::Mat leftRectify, rightRectify, disparity(left.size(), CV_8U);
    cv::remap(left, leftRectify, _map11, _map12, cv::INTER_LINEAR);
    cv::remap(right, rightRectify, _map21, _map22, cv::INTER_LINEAR);
#endif
    if (!params.use_vworks) {
#ifdef USE_CUDA
        cv::Mat left_rect, right_rect, disparity;
        leftRectify.download(left_rect);
        rightRectify.download(right_rect);
#else
        cv::Mat & left_rect = leftRectify;
        cv::Mat & right_rect = rightRectify;
        cv::Mat disparity;

#endif
        
        auto sgbm = cv::StereoSGBM::create(params.min_disparity, params.num_disp, params.block_size,
            params.p1, params.p2, params.disp12Maxdiff, params.prefilterCap, params.uniquenessRatio, params.speckleWindowSize, 
            params.speckleRange, params.mode);

        // sgbm->compute(right_rect, left_rect, disparity);
        sgbm->compute(left_rect, right_rect, disparity);

        ROS_INFO("CPU SGBM time cost %fms", tic.toc());
        if (show) {
            cv::Mat _show;
            cv::Mat raw_disp_map = disparity.clone();
            cv::Mat scaled_disp_map;
            double min_val, max_val;
            cv::minMaxLoc(raw_disp_map, &min_val, &max_val, NULL, NULL);
            raw_disp_map.convertTo(scaled_disp_map, CV_8U, 255/(max_val-min_val), -min_val/(max_val-min_val));

            cv::hconcat(left_rect, right_rect, _show);
            cv::hconcat(_show, scaled_disp_map, _show);
            // cv::hconcat(left_rect, right_rect, _show);
            // cv::hconcat(_show, scaled_disp_map, _show);
            cv::imshow("raw_disp_map", _show);
            cv::waitKey(2);
        }
        return disparity;
    } else {
#ifdef WITHOUT_VWORKS
        ROS_ERROR("You must set enable_vworks to true or disable vworks in depth config file");
        exit(-1);
#else
        leftRectify.copyTo(leftRectify_fix);
        rightRectify.copyTo(rightRectify_fix);
        if (first_use_vworks) {
            auto lsize = leftRectify_fix.size();
            // disparity_fix = cv::cuda::GpuMat(leftRectify_fix.size(), CV_8U);
            // disparity_fix_cpu = cv::Mat(leftRectify_fix.size(), CV_8U);
#ifdef OVX
            vxDirective(context, VX_DIRECTIVE_ENABLE_PERFORMANCE);
            vxRegisterLogCallback(context, &ovxio::stdoutLogCallback, vx_false_e);
#endif
            //
            // Create a NVXIO-based frame source
            //

            StereoMatching::ImplementationType implementationType = StereoMatching::HIGH_LEVEL_API;
            StereoMatching::StereoMatchingParams _params;
            _params.min_disparity = 0;
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

        ROS_INFO("Visionworks DISP %d %d!Time %fms", cv_disp.size().width, cv_disp.size().height, tic.toc());
        if (show) {
            cv::Mat color_disp;
            color->process();
            nvx_cv::VXImageToCVMatMapper map(vx_coloroutput, plane_index, &rect, VX_WRITE_ONLY, NVX_MEMORY_TYPE_CUDA);
            auto cv_disp_cuda = map.getGpuMat();
            cv_disp_cuda.download(color_disp);

            double min_val=0, max_val=0;
            cv::Mat gray_disp;
            cv::minMaxLoc(cv_disp, &min_val, &max_val, NULL, NULL);
            // ROS_INFO("Min %f, max %f", min_val, max_val);
            cv_disp.convertTo(gray_disp, CV_8U, 1., 0);
            cv::cvtColor(gray_disp, gray_disp, cv::COLOR_GRAY2BGR);

            cv::Mat _show, left_rect, right_rect;
            leftRectify.download(left_rect);
            rightRectify.download(right_rect);
    
            cv::hconcat(left_rect, right_rect, _show);
            cv::cvtColor(_show, _show, cv::COLOR_GRAY2BGR);
            cv::hconcat(_show, gray_disp, _show);
            cv::hconcat(_show, color_disp, _show);
            char win_name[50] = {0};
            // sprintf(win_name, "RAW_DISP %f %f %f", T.at<double>(0, 0), T.at<double>(1, 0), T.at<double>(2, 0));
            cv::imshow("Disparity", _show);
            cv::waitKey(2);
        }            
        return cv_disp;
#endif  
    }
}

cv::Mat DepthEstimator::ComputeDepthCloud(cv::Mat & left, cv::Mat & right) {
    static int count = 0;
    int skip = 10/extrinsic_calib_rate;
    if (skip <= 0) {
        skip = 1;
    }
    if (count ++ % 5 == 0) {
        if(enable_extrinsic_calib) {

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

