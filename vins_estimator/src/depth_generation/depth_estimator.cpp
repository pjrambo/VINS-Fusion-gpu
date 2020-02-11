#include "depth_estimator.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/opencv.hpp>
#include "../utility/tic_toc.h"
#include <opencv2/cudafeatures2d.hpp>
#include "../estimator/parameters.h"

#ifndef WITHOUT_VWORKS
ovxio::ContextGuard context;
#endif

#define MINIUM_ESSENTIALMAT_SIZE 10
#define GOOD_RT_THRES 0.1
#define GOOD_RT_THRES_T 0.1

bool DepthEstimator::calibrate_extrincic(cv::cuda::GpuMat & left, cv::cuda::GpuMat & right) {

    std::vector<cv::Point2f> Pts1;
    std::vector<cv::Point2f> Pts2;
    TicToc tic1;
    find_corresponding_pts(left, right, Pts1, Pts2, true);

    if (Pts1.size() < MINIUM_ESSENTIALMAT_SIZE) {
        return false;
    }

    left_pts.insert( left_pts.end(), Pts1.begin(), Pts1.end() );
    right_pts.insert( right_pts.end(), Pts2.begin(), Pts2.end() );

    ROS_INFO("All pts for stereo calib %d; Find1 use %fms", left_pts.size(), tic1.toc());

    if (left_pts.size() < 50) {
        return false;
    }
    TicToc tic2;
    vector<uchar> status;
    cv::Mat essentialMat = cv::findEssentialMat(left_pts, right_pts, cameraMatrix, cv::RANSAC, 0.999, 1.0, status);
    ROS_INFO("Find2 use %fms", left_pts.size(), tic2.toc());

    double scale = norm(T);

    cv::Mat R1, R2, t;
    decomposeEssentialMat(essentialMat, R1, R2, t);
    if (t.at<double>(0, 0) > 0) {
        t = -t;
    }


    double dis1 = norm(R - R1);
    double dis2 = norm(R - R2);
    double dis3 = norm(t - T/scale);

    std::cout << "R0" << R << std::endl;
    std::cout << "T0" << T << std::endl;
    
    std::cout << "Essential Matrix" << essentialMat << std::endl;
    std::cout << "R1" << R1 << "DIS" << norm(R - R1) << std::endl;
    std::cout << "R2" << R2 << "DIS" << norm(R - R2) << std::endl;
    std::cout << "T1" << t*scale << "DIS" << dis3 << std::endl;

    if (dis1 < dis2) {
        if (dis1 < GOOD_RT_THRES && dis3 < GOOD_RT_THRES_T) {
            R = R1;
            T = t*scale;
            return true;
        }
    } else {
        if (dis2 < GOOD_RT_THRES && dis3 < GOOD_RT_THRES_T) {
            R = R2;
            T = t*scale;
            return true;
        }
    }
    return false;
}

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
            vxDirective(context, VX_DIRECTIVE_ENABLE_PERFORMANCE);

            vxRegisterLogCallback(context, &ovxio::stdoutLogCallback, vx_false_e);

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
            sprintf(win_name, "RAW_DISP %f %f %f", T.at<double>(0, 0), T.at<double>(1, 0), T.at<double>(2, 0));
            cv::imshow(win_name, _show);
            cv::waitKey(2);
        }            
        return cv_disp;
#endif  
    }
}

cv::Mat DepthEstimator::ComputeDepthCloud(cv::cuda::GpuMat & left, cv::cuda::GpuMat & right) {
    static int count = 0;
    if (count ++ % 10 == 0) {
        if(enable_extrinsic_calib) {
            bool success = calibrate_extrincic(left, right);
            if (success) {
                cv::FileStorage fs(output_path, cv::FileStorage::WRITE);
                fs << "R" << R;
                fs << "T" << T;
                fs.release();
            }
        }
    }
    
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
        cv::threshold(imgDisparity32F, imgDisparity32F, min_val, 1000, cv::THRESH_TOZERO);
        // dispartitymap.convertTo(imgDisparity32F, CV_32F, 1.0);
        cv::minMaxLoc(imgDisparity32F, &min_val, &max_val, NULL, NULL);
    } else {
        dispartitymap.convertTo(imgDisparity32F, CV_32F, 1./16);
    }
    cv::Mat XYZ = cv::Mat::zeros(imgDisparity32F.rows, imgDisparity32F.cols, CV_32FC3);   // Output point cloud
    cv::reprojectImageTo3D(imgDisparity32F, XYZ, Q);    // cv::project

    return XYZ;
}

#define ORB_HAMMING_DISTANCE 40 //Max hamming
#define ORB_UV_DISTANCE 1.5 //UV distance bigger than mid*this will be removed


std::vector<cv::KeyPoint> detect_orb_by_region(cv::Mat _img, int features, int cols = 4, int rows = 3) {
    int small_width = _img.cols / cols;
    int small_height = _img.rows / rows;
    printf("Cut to W %d H %d for FAST\n", small_width, small_height);
    
    auto _orb = cv::ORB::create(features/(cols*rows));
    std::vector<cv::KeyPoint> ret;
    for (int i = 0; i < cols; i ++) {
        for (int j = 0; j < rows; j ++) {
            std::vector<cv::KeyPoint> kpts;
            _orb->detect(_img(cv::Rect(small_width*i, small_width*j, small_width, small_height)), kpts);
            printf("Detect %ld feature in reigion %d %d\n", kpts.size(), i, j);

            for (auto kp : kpts) {
                kp.pt.x = kp.pt.x + small_width*i;
                kp.pt.y = kp.pt.y + small_width*j;
                ret.push_back(kp);
            }
        }
    }

    return ret;
}

std::vector<cv::DMatch> filter_by_duv(const std::vector<cv::DMatch> & matches, 
    std::vector<cv::KeyPoint> query_pts, 
    std::vector<cv::KeyPoint> train_pts) {
    std::vector<cv::DMatch> good_matches;
    std::vector<float> uv_dis;
    for (auto gm : matches) {
        if (gm.queryIdx >= query_pts.size() || gm.trainIdx >= train_pts.size()) {
            ROS_ERROR("out of size");
            exit(-1);
        } 
        uv_dis.push_back(cv::norm(query_pts[gm.queryIdx].pt - train_pts[gm.trainIdx].pt));
    }

    std::sort(uv_dis.begin(), uv_dis.end());
    
    // printf("MIN UV DIS %f, MID %f END %f\n", uv_dis[0], uv_dis[uv_dis.size()/2], uv_dis[uv_dis.size() - 1]);

    double mid_dis = uv_dis[uv_dis.size()/2];

    for (auto gm: matches) {
        if (gm.distance < mid_dis*ORB_UV_DISTANCE) {
            good_matches.push_back(gm);
        }
    }

    return good_matches;
}

std::vector<cv::DMatch> filter_by_x(const std::vector<cv::DMatch> & matches, 
    std::vector<cv::KeyPoint> query_pts, 
    std::vector<cv::KeyPoint> train_pts, double OUTLIER_XY_PRECENT) {
    std::vector<cv::DMatch> good_matches;
    std::vector<float> dxs;
    for (auto gm : matches) {
        dxs.push_back(query_pts[gm.queryIdx].pt.x - train_pts[gm.trainIdx].pt.x);
    }

    std::sort(dxs.begin(), dxs.end());

    int num = dxs.size();
    int l = num*OUTLIER_XY_PRECENT;
    if (l == 0) {
        l = 1;
    }
    int r = num*(1-OUTLIER_XY_PRECENT);
    if (r >= num - 1) {
        r = num - 2;
    }

    if (r <= l ) {
        return good_matches;
    }

    // printf("MIN DX DIS:%f, l:%f m:%f r:%f END:%f\n", dxs[0], dxs[l], dxs[num/2], dxs[r], dxs[dxs.size() - 1]);

    double lv = dxs[l];
    double rv = dxs[r];

    for (auto gm: matches) {
        if (query_pts[gm.queryIdx].pt.x - train_pts[gm.trainIdx].pt.x > lv && query_pts[gm.queryIdx].pt.x - train_pts[gm.trainIdx].pt.x < rv) {
            good_matches.push_back(gm);
        }
    }

    return good_matches;
}

std::vector<cv::DMatch> filter_by_y(const std::vector<cv::DMatch> & matches, 
    std::vector<cv::KeyPoint> query_pts, 
    std::vector<cv::KeyPoint> train_pts, double OUTLIER_XY_PRECENT) {
    std::vector<cv::DMatch> good_matches;
    std::vector<float> dys;
    for (auto gm : matches) {
        dys.push_back(query_pts[gm.queryIdx].pt.y - train_pts[gm.trainIdx].pt.y);
    }

    std::sort(dys.begin(), dys.end());

    int num = dys.size();
    int l = num*OUTLIER_XY_PRECENT;
    if (l == 0) {
        l = 1;
    }
    int r = num*(1-OUTLIER_XY_PRECENT);
    if (r >= num - 1) {
        r = num - 2;
    }

    if (r <= l ) {
        return good_matches;
    }

    // printf("MIN DX DIS:%f, l:%f m:%f r:%f END:%f\n", dys[0], dys[l], dys[num/2], dys[r], dys[dys.size() - 1]);

    double lv = dys[l];
    double rv = dys[r];

    for (auto gm: matches) {
        if (query_pts[gm.queryIdx].pt.y - train_pts[gm.trainIdx].pt.y > lv && query_pts[gm.queryIdx].pt.y - train_pts[gm.trainIdx].pt.y < rv) {
            good_matches.push_back(gm);
        }
    }

    return good_matches;
}

std::vector<cv::DMatch> filter_by_hamming(const std::vector<cv::DMatch> & matches) {
    std::vector<cv::DMatch> good_matches;
    std::vector<float> dys;
    for (auto gm : matches) {
        dys.push_back(gm.distance);
    }

    std::sort(dys.begin(), dys.end());

    // printf("MIN DX DIS:%f, 2min %fm ax %f\n", dys[0], 2*dys[0], dys[dys.size() - 1]);

    double max_hamming = 2*dys[0];
    if (max_hamming < ORB_HAMMING_DISTANCE) {
        max_hamming = ORB_HAMMING_DISTANCE;
    }
    for (auto gm: matches) {
        if (gm.distance < max_hamming) {
            good_matches.push_back(gm);
        }
    }

    return good_matches;
}


void DepthEstimator::find_corresponding_pts(cv::cuda::GpuMat & img1, cv::cuda::GpuMat & img2, std::vector<cv::Point2f> & Pts1, std::vector<cv::Point2f> & Pts2, bool visualize) {
    TicToc tic;
    std::vector<cv::KeyPoint> kps1, kps2;
    std::vector<cv::DMatch> good_matches;
    // bool use_surf = false;

    // auto _orb = cv::ORB::create(1000, 1.2f, 8, 31, 0, 4, cv::ORB::HARRIS_SCORE, 31, 20);
    std::cout << img1.size() << std::endl;
    // auto _orb = cv::cuda::ORB::create(1000, 1.2f, 8, 31, 0, 4, cv::ORB::HARRIS_SCORE, 31, 20);
    // cv::Mat _mask(img1.size(), CV_8UC1, cv::Scalar(255));
    // cv::cuda::GpuMat mask(_mask);
    
    cv::Mat desc1, desc2;
    cv::Mat _img1, _img2, mask;
    
    img1.download(_img1);
    img2.download(_img2);

    auto _orb = cv::ORB::create(1000, 1.2f, 8, 31, 0, 4, cv::ORB::HARRIS_SCORE, 31, 20);
    _orb->detectAndCompute(_img1, mask, kps1, desc1);
    _orb->detectAndCompute(_img2, mask, kps2, desc2);

    size_t j = 0;

    cv::BFMatcher bfmatcher(cv::NORM_HAMMING2, true);
    std::vector<cv::DMatch> matches;
    bfmatcher.match(desc2, desc1, matches);
    // printf("ORIGIN MATCHES %ld\n", matches.size());
    matches = filter_by_hamming(matches);
    // printf("AFTER HAMMING X MATCHES %ld\n", matches.size());
    
    // matches = filter_by_duv(matches, kps2, kps1);
    // printf("AFTER DUV MATCHES %ld\n", matches.size());

    double thres = 0.05;
    
    matches = filter_by_x(matches, kps2, kps1, thres);
    matches = filter_by_y(matches, kps2, kps1, thres);

    vector<cv::Point2f> _pts1, _pts2;
    vector<uchar> status;
    for (auto gm : matches) {
        auto _id1 = gm.trainIdx;
        auto _id2 = gm.queryIdx;
        _pts1.push_back(kps1[_id1].pt);
        _pts2.push_back(kps2[_id2].pt);
    }

    // cv::findEssentialMat(_pts1, _pts2, cv::RANSAC, 0.99, 1.0, status);
    cv::findEssentialMat(_pts1, _pts2, cameraMatrix, cv::RANSAC, 0.99, 1.0, status);

    for(int i = 0; i < _pts1.size(); i ++) {
        if (i < status.size() && status[i]) {
            Pts1.push_back(_pts1[i]);
            Pts2.push_back(_pts2[i]);
            good_matches.push_back(matches[i]);
        }
    }

    ROS_INFO("Find correponding cost %fms", tic.toc());
    
    if (visualize) {
        cv::Mat img1_cpu, img2_cpu, _show;
        // img1.download(_img1);
        // img2.download(_img2);
        cv::drawMatches(_img2, kps2, _img1, kps1, good_matches, _show);
        // cv::resize(_show, _show, cv::Size(), VISUALIZE_SCALE, VISUALIZE_SCALE);
        cv::imshow("KNNMatch", _show);
        cv::waitKey(2);
    }
}
