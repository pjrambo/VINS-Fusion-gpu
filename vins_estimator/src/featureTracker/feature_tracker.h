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

typedef Eigen::Matrix<double, 8, 1> TrackFeatureNoId;
typedef pair<int, TrackFeatureNoId> TrackFeature;
typedef vector<TrackFeature> FeatureFramenoId;
typedef map<int, FeatureFramenoId> FeatureFrame;
class Estimator;
class FeatureTracker
{
public:
    Estimator * estimator = nullptr;
    FeatureTracker();
    FeatureFrame trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    FeatureFrame trackImage_fisheye(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1, 
        const Eigen::Vector3d & tic0, const Eigen::Matrix3d & ric0,
        const Eigen::Vector3d & tic1, const Eigen::Matrix3d & ric1,
        cv::Mat & depthmap
    );

    vector<cv::Point2f> opticalflow_track(cv::cuda::GpuMat & cur_img, 
                        cv::cuda::GpuMat & prev_img, vector<cv::Point2f> & prev_pts, 
                        vector<int> & ids, vector<int> & track_cnt,
                        bool flow_back, vector<cv::Point2f> prediction_points = vector<cv::Point2f>());
    
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
    vector<cv::Point3f> undistortedPtsTop(vector<cv::Point2f> &pts, FisheyeUndist & fisheye);
    vector<cv::Point3f> undistortedPtsSide(vector<cv::Point2f> &pts, FisheyeUndist & fisheye, bool is_downward);

    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);

    vector<cv::Point3f> ptsVelocity3D(vector<int> &ids, vector<cv::Point3f> &pts, 
                                    map<int, cv::Point3f> &cur_id_pts, map<int, cv::Point3f> &prev_id_pts);

    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2, 
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                    vector<int> &curLeftIds,
                    vector<cv::Point2f> &curLeftPts, 
                    vector<cv::Point2f> &curRightPts,
                    map<int, cv::Point2f> &prevLeftPtsMap);
    
    void setup_feature_frame(FeatureFrame & ff, vector<int> ids, vector<cv::Point2f> cur_pts, vector<cv::Point3f> cur_un_pts, vector<cv::Point3f> cur_pts_vel, int camera_id);
    FeatureFrame setup_feature_frame();
    
    void drawTrackFisheye(const cv::Mat & img_up, const cv::Mat & img_down, 
                            cv::cuda::GpuMat & imUpTop,
                            cv::cuda::GpuMat & imDownTop,
                            cv::cuda::GpuMat & imUpSide, 
                            cv::cuda::GpuMat & imDownSide);
    
    void drawTrackImage(cv::Mat & img, vector<cv::Point2f> pts, vector<int> ids, vector<int> track_cnt, map<int, cv::Point2f> prev_pts);

    void setPrediction(map<int, Eigen::Vector3d> &predictPts);
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    void removeOutliers(set<int> &removePtsIds);
    cv::Mat getTrackImage();
    bool inBorder(const cv::Point2f &pt);
    bool inBorder(const cv::Point2f &pt, cv::Size shape);

    void detectPoints(const cv::cuda::GpuMat & img, const cv::Mat & mask, vector<cv::Point2f> & n_pts, vector<cv::Point2f> & cur_pts, int require_pts);

    void setFeatureStatus(int feature_id, int status) {
        this->pts_status[feature_id] = status;
        if (status < 0) {
            removed_pts.insert(feature_id);
        }
    }

    int row, col;
    cv::Mat imTrack;
    cv::Mat mask;

    cv::Mat mask_up_top, mask_down_top, mask_up_side;
    cv::Size top_size;
    cv::Size side_size;
    
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img;
    cv::cuda::GpuMat prev_gpu_img, cur_gpu_img;

    cv::cuda::GpuMat prev_up_top_img, prev_down_top_img, prev_up_side_img;

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
    map<int, int> pts_status;


    vector<cv::Point2f> predict_up_side, predict_pts_left_top, predict_pts_right_top, predict_pts_down_side;
    vector<cv::Point2f> prev_up_top_pts, cur_up_top_pts, prev_up_side_pts, cur_up_side_pts, prev_down_top_pts, prev_down_side_pts;
    
    vector<cv::Point3f> prev_up_top_un_pts,  prev_up_side_un_pts, prev_down_top_un_pts, prev_down_side_un_pts;
    vector<cv::Point2f> cur_down_top_pts, cur_down_side_pts;

    vector<cv::Point3f> up_top_vel, up_side_vel, down_top_vel, down_side_vel;
    vector<cv::Point3f> cur_up_top_un_pts, cur_up_side_un_pts, cur_down_top_un_pts, cur_down_side_un_pts;

    vector<int> ids_up_top, ids_up_side, ids_down_top, ids_down_side;
    map<int, cv::Point2f> up_top_prevLeftPtsMap;
    map<int, cv::Point2f> down_top_prevLeftPtsMap;
    map<int, cv::Point2f> up_side_prevLeftPtsMap;
    map<int, cv::Point2f> down_side_prevLeftPtsMap;
    set<int> removed_pts;


    // vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
    // vector<cv::Point2f> pts_velocity, right_pts_velocity;
    

    vector<int> track_cnt;

    vector<int> track_up_top_cnt;
    vector<int> track_down_top_cnt;
    vector<int> track_up_side_cnt;
    vector<int> track_down_side_cnt;
    map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
    map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;

    map<int, cv::Point3f> cur_up_top_un_pts_map, prev_up_top_un_pts_map;
    map<int, cv::Point3f> cur_down_top_un_pts_map, prev_down_top_un_pts_map;
    map<int, cv::Point3f> cur_up_side_un_pts_map, prev_up_side_un_pts_map;
    map<int, cv::Point3f> cur_down_side_un_pts_map, prev_down_side_un_pts_map;

    map<int, cv::Point2f> prevLeftPtsMap;
    vector<camodocal::CameraPtr> m_camera;
    vector<FisheyeUndist> fisheys_undists;
    double cur_time;
    double prev_time;
    bool stereo_cam;
    int n_id;
    bool hasPrediction;
};
