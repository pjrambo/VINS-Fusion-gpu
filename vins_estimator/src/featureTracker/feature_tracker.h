/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"
#include "fisheye_undist.hpp"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
public:
    FeatureTracker();
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage_fisheye(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    
    void setMask();
    void setMaskFisheye();
    cv::Mat setMaskFisheye(cv::Size shape, vector<cv::Point2f> & cur_pts, vector<int> & track_cnt, vector<int> & ids);
    void addPoints();
    void addPointsFisheye();
    void readIntrinsicParameter(const vector<string> &calib_file);
    void showUndistortion(const string &name);
    void rejectWithF();
    void undistortedPoints();
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);
    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2, 
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                                   vector<int> &curLeftIds,
                                   vector<cv::Point2f> &curLeftPts, 
                                   vector<cv::Point2f> &curRightPts,
                                   map<int, cv::Point2f> &prevLeftPtsMap);
    
    featureFrame setup_feature_frame();
    
    void drawTrackFisheye(cv::cuda::GpuMat & imUpTop,
                            cv::cuda::GpuMat & imDownTop,
                            cv::cuda::GpuMat & imUpSide, 
                            cv::cuda::GpuMat & imDownSide);
    
    void drawTrackImage(cv::Mat & img, vector<cv::Point2f> pts, vector<int> ids, map<int, cv::Point2f> prev_pts);

    void setPrediction(map<int, Eigen::Vector3d> &predictPts);
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    void removeOutliers(set<int> &removePtsIds);
    cv::Mat getTrackImage();
    bool inBorder(const cv::Point2f &pt);

    void detectPoints(const cv::cuda::GpuMat & img, const cv::Mat & mask, vector<cv::Point2f> & n_pts, vector<cv::Point2f> & cur_pts, int require_pts);

    int row, col;
    cv::Mat imTrack;
    cv::Mat mask;

    cv::Mat mask_up_top, mask_down_top, mask_up_side;
    cv::Size top_size;
    cv::Size side_size;
    
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img;
    cv::cuda::GpuMat prev_gpu_img, cur_gpu_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> n_pts_up_top, n_pts_down_top, n_pts_up_side;
    int sum_n;

    vector<cv::Point2f> predict_pts;
    vector<cv::Point2f> predict_pts_debug;
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
    vector<cv::Point2f> pts_velocity, right_pts_velocity;
    vector<int> ids, ids_right;
    vector<int> pts_img_id, pts_img_id_right;


    vector<cv::Point2f> predict_up_side, predict_pts_left_top, predict_pts_right_top, predict_pts_down_side;
    vector<cv::Point2f> prev_up_top_pts, cur_up_top_pts, prev_up_side_pts, cur_up_side_pts;
    vector<cv::Point2f> cur_down_top_pts, cur_down_side_pts;
    vector<int> ids_up_top, ids_up_side, ids_down_top, ids_down_side;
    map<int, cv::Point2f> up_top_prevLeftPtsMap;
    map<int, cv::Point2f> down_top_prevLeftPtsMap;
    map<int, cv::Point2f> up_side_prevLeftPtsMap;
    map<int, cv::Point2f> down_side_prevLeftPtsMap;


    // vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
    // vector<cv::Point2f> pts_velocity, right_pts_velocity;
    

    vector<int> track_cnt;

    vector<int> track_up_top_cnt;
    vector<int> track_down_top_cnt;
    vector<int> track_up_side_cnt;
    map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
    map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
    map<int, cv::Point2f> prevLeftPtsMap;
    vector<camodocal::CameraPtr> m_camera;
    vector<FisheyeUndist> fisheys_undists;
    double cur_time;
    double prev_time;
    bool stereo_cam;
    int n_id;
    bool hasPrediction;
};
