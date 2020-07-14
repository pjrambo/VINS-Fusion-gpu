#include "depth_estimator.h"
#include <opencv2/calib3d.hpp>

#include "../utility/opencv_cuda.h"

#include <opencv2/opencv.hpp>
#include "../utility/tic_toc.h"
#include "../estimator/parameters.h"

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

#ifdef USE_CUDA
    if (!params.use_vworks) {
    	sgmp = new sgm::LibSGMWrapper(params.num_disp, params.p1, params.p2, params.uniquenessRatio, true, 
            sgm::PathType::SCAN_8PATH, params.min_disparity, params.disp12Maxdiff);
    }
#endif
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
    

cv::Mat DepthEstimator::ComputeDispartiyMap(cv::cuda::GpuMat & left, cv::cuda::GpuMat & right) {
#ifdef USE_CUDA
    std::cout << "Computing disp for cuda" << std::endl;
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
        map11.upload(_map11);
        map12.upload(_map12);
        map21.upload(_map21);
        map22.upload(_map22);

        _Q.convertTo(Q, CV_32F);

        first_init = false;
    } 


    cv::cuda::GpuMat leftRectify, rightRectify;
    cv::cuda::remap(left, leftRectify, map11, map12, cv::INTER_LINEAR);
    cv::cuda::remap(right, rightRectify, map21, map22, cv::INTER_LINEAR);

    cv::cuda::GpuMat disparity(leftRectify.size(), CV_8U);
    if (!params.use_vworks) {
        cv::Mat disparity;
        cv::cuda::GpuMat d_disparity;

		sgmp->execute(leftRectify, rightRectify, d_disparity);
        d_disparity.download(disparity);
        
        cv::Mat mask = disparity == sgmp->getInvalidDisparity();


        if (show) {
            cv::Mat disparity_color, disp;
    	    disparity.convertTo(disp, CV_8U, 255. / params.num_disp);

            cv::applyColorMap(disp, disparity_color, cv::COLORMAP_JET);
	        disparity.setTo(0, mask);
	        disparity_color.setTo(cv::Scalar(0, 0, 0), mask);

            cv::Mat _show, left_rect, right_rect;
            leftRectify.download(left_rect);
            rightRectify.download(right_rect);

            std::cout << "Size L" << left_rect.size() << " R" << right_rect.size() << "D" << disparity_color.size() << std::endl;

            cv::hconcat(left_rect, right_rect, _show);
            cv::cvtColor(_show, _show, cv::COLOR_GRAY2BGR);
            cv::hconcat(_show, disparity_color, _show);

            cv::imshow("RAW DISP", _show);
            cv::waitKey(2);
        }            
            
        ROS_INFO("SGBM time cost %fms", tic.toc());

        return disparity;

    } else {
#ifdef WITHOUT_VWORKS
        ROS_ERROR("You must set enable_vworks to true or disable vworks in depth config file");
        exit(-1);
        return cv::Mat();
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
#else
    printf("Must set USE_CUDA on in CMake to enable cuda!!!\n");
    exit(-1);
#endif
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
        _Q.convertTo(Q, CV_32F);

        first_init = false;
    } 


    cv::Mat leftRectify, rightRectify, disparity(left.size(), CV_8U);
    cv::remap(left, leftRectify, _map11, _map12, cv::INTER_LINEAR);
    cv::remap(right, rightRectify, _map21, _map22, cv::INTER_LINEAR);

    if (!params.use_vworks) {
        cv::Mat & left_rect = leftRectify;
        cv::Mat & right_rect = rightRectify;
        cv::Mat disparity;
        
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
    }
}



