#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <opencv2/cudastereo.hpp>
#include "../utility/tic_toc.h"


#define ORB_HAMMING_DISTANCE 40 //Max hamming
#define ORB_UV_DISTANCE 1.5 //UV distance bigger than mid*this will be removed
#define MINIUM_ESSENTIALMAT_SIZE 10
#define GOOD_R_THRES 0.1
#define GOOD_T_THRES 0.1

using namespace std;

class StereoOnlineCalib {
    cv::Mat cameraMatrix;
    cv::Mat R, T;
    Eigen::Vector3d T_eig;
    Eigen::Matrix3d R_eig;
    bool show;
    std::vector<cv::Point2f> left_pts, right_pts;
public:
    StereoOnlineCalib(cv::Mat _R, cv::Mat _T, cv::Mat _cameraMatrix, bool _show):
        R(_R), T(_T), cameraMatrix(_cameraMatrix), show(_show)
    {
    }

    cv::Mat get_rotation() {
        return R;
    }

    cv::Mat get_translation() {
        return T;
    }
    
    void find_corresponding_pts(cv::cuda::GpuMat & img1, cv::cuda::GpuMat & img2, std::vector<cv::Point2f> & Pts1, std::vector<cv::Point2f> & Pts2);
    bool calibrate_extrincic(cv::cuda::GpuMat & left, cv::cuda::GpuMat & right);
    static std::vector<cv::KeyPoint> detect_orb_by_region(cv::Mat _img, int features, int cols = 4, int rows = 3);
    bool calibrate_extrinsic_opencv(std::vector<cv::Point2f> left_pts, std::vector<cv::Point2f> right_pts);
};

bool StereoOnlineCalib::calibrate_extrinsic_opencv(std::vector<cv::Point2f> left_pts, std::vector<cv::Point2f> right_pts) {
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

    // std::cout << "R0" << R << std::endl;
    // std::cout << "T0" << T << std::endl;
    
    // std::cout << "Essential Matrix" << essentialMat << std::endl;
    // std::cout << "R1" << R1 << "DIS" << norm(R - R1) << std::endl;
    // std::cout << "R2" << R2 << "DIS" << norm(R - R2) << std::endl;
    // std::cout << "T1" << t*scale << "DIS" << dis3 << std::endl;

    if (dis1 < dis2) {
        if (dis1 < GOOD_R_THRES && dis3 < GOOD_T_THRES) {
            R = R1;
            T = t*scale;
            
            ROS_INFO("Update R T");
            std::cout << "R" << R << std::endl;
            std::cout << "T" << T << std::endl;
            return true;
        }
    } else {
        if (dis2 < GOOD_R_THRES && dis3 < GOOD_T_THRES) {
            R = R2;
            T = t*scale;

            ROS_INFO("Update R T");
            std::cout << "R" << R << std::endl;
            std::cout << "T" << T << std::endl;
            return true;
        }
    }
    return false;
}

bool StereoOnlineCalib::calibrate_extrincic(cv::cuda::GpuMat & left, cv::cuda::GpuMat & right) {

    std::vector<cv::Point2f> Pts1;
    std::vector<cv::Point2f> Pts2;
    TicToc tic1;
    find_corresponding_pts(left, right, Pts1, Pts2);

    if (Pts1.size() < MINIUM_ESSENTIALMAT_SIZE) {
        return false;
    }

    left_pts.insert( left_pts.end(), Pts1.begin(), Pts1.end() );
    right_pts.insert( right_pts.end(), Pts2.begin(), Pts2.end() );

    ROS_INFO("All pts for stereo calib %d; Find1 use %fms", left_pts.size(), tic1.toc());

    return calibrate_extrinsic_opencv(left_pts, right_pts);
}

std::vector<cv::KeyPoint> StereoOnlineCalib::detect_orb_by_region(cv::Mat _img, int features, int cols, int rows) {
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


void StereoOnlineCalib::find_corresponding_pts(cv::cuda::GpuMat & img1, cv::cuda::GpuMat & img2, std::vector<cv::Point2f> & Pts1, std::vector<cv::Point2f> & Pts2) {
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

    if (_pts1.size() > MINIUM_ESSENTIALMAT_SIZE) {
        cv::findEssentialMat(_pts1, _pts2, cameraMatrix, cv::RANSAC, 0.99, 1.0, status);
    }

    for(int i = 0; i < _pts1.size(); i ++) {
        if (i < status.size() && status[i]) {
            Pts1.push_back(_pts1[i]);
            Pts2.push_back(_pts2[i]);
            good_matches.push_back(matches[i]);
        }
    }

    ROS_INFO("Find correponding cost %fms", tic.toc());
    
    if (show) {
        cv::Mat img1_cpu, img2_cpu, _show;
        // img1.download(_img1);
        // img2.download(_img2);
        cv::drawMatches(_img2, kps2, _img1, kps1, good_matches, _show);
        // cv::resize(_show, _show, cv::Size(), VISUALIZE_SCALE, VISUALIZE_SCALE);
        cv::imshow("KNNMatch", _show);
        cv::waitKey(2);
    }
}
