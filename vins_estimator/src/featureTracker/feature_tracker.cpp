/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com), Xu Hao (xuhao3e8@gmail.com)
 *******************************************************/

#include "feature_tracker.h"
#include "../estimator/estimator.h"
#include "fisheye_undist.hpp"

#ifdef WITH_VWORKS
#include "vworks_feature_tracker.hpp"
#ifdef OVX
ovxio::ContextGuard context;
#else 
vx_context context;
#endif
#endif
// #define PERF_OUTPUT
Eigen::Quaterniond t1(Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(1, 0, 0)));
Eigen::Quaterniond t2 = t1 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
Eigen::Quaterniond t3 = t2 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
Eigen::Quaterniond t4 = t3 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
Eigen::Quaterniond t_down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1, 0, 0)));

#define PYR_LEVEL 3
#define WIN_SIZE cv::Size(21, 21)

bool FeatureTracker::inBorder(const cv::Point2f &pt, cv::Size shape) const
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < shape.width - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < shape.height - BORDER_SIZE;
}

bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

double FeatureTracker::distance(cv::Point2f pt1, cv::Point2f pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
    stereo_cam = 0;
    n_id = 0;
    hasPrediction = false;
    sum_n = 0;
}

cv::Mat FeatureTracker::setMaskFisheye(cv::Size shape, vector<cv::Point2f> & cur_pts,
    vector<int> & track_cnt, vector<int> & ids) {
    mask = cv::Mat(shape, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        // if (removed_pts.find(it.second.second) == removed_pts.end()) {
            if (mask.at<uchar>(it.second.first) == 255)
            {
                cur_pts.push_back(it.second.first);
                ids.push_back(it.second.second);
                track_cnt.push_back(it.first);
                cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
            }
        // }
    }

    return mask;
}


void FeatureTracker::setMaskFisheye() {
    //TODO:Set mask for fisheye
    if(enable_up_top) {
        mask_up_top = setMaskFisheye(top_size, cur_up_top_pts, track_up_top_cnt, ids_up_top);
    }

    if(enable_down_top) {
        mask_down_top = setMaskFisheye(top_size, cur_down_top_pts, track_down_top_cnt, ids_down_top);
    }

    if(enable_up_side) {
        mask_up_side = setMaskFisheye(side_size, cur_up_side_pts, track_up_side_cnt, ids_up_side);
    }
}

void FeatureTracker::setMask()
{
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (removed_pts.find(it.second.second) == removed_pts.end()) {
            if (mask.at<uchar>(it.second.first) == 255)
            {
                cur_pts.push_back(it.second.first);
                ids.push_back(it.second.second);
                track_cnt.push_back(it.first);
                cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
            }
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        cur_pts.push_back(p);
        ids.push_back(n_id++);
        track_cnt.push_back(1);
    }
}

double distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

void FeatureTracker::drawTrackImage(cv::Mat & img, vector<cv::Point2f> pts, vector<int> ids, map<int, cv::Point2f> prev_pts) {
    char idtext[10] = {0};
    for (size_t j = 0; j < pts.size(); j++) {
        //Not tri
        //Not solving
        //Just New point yellow
        cv::Scalar color = cv::Scalar(0, 255, 255);
        if (pts_status.find(ids[j]) != pts_status.end()) {
            int status = pts_status[ids[j]];
            if (status < 0) {
                //Removed points
                color = cv::Scalar(0, 0, 0);
            }

            if (status == 1) {
                //Good pt; But not used for solving; Blue 
                color = cv::Scalar(255, 0, 0);
            }

            if (status == 2) {
                //Bad pt; Red
                color = cv::Scalar(0, 0, 255);
            }

            if (status == 3) {
                //Good pt for solving; Green
                color = cv::Scalar(0, 255, 0);
            }

        }

        cv::circle(img, pts[j], 1, color, 2);

        sprintf(idtext, "%d", ids[j]);
	    cv::putText(img, idtext, pts[j] - cv::Point2f(5, 0), cv::FONT_HERSHEY_SIMPLEX, 1, color, 3);

    }

    for (size_t i = 0; i < ids.size(); i++)
    {
        int id = ids[i];
        auto mapIt = prev_pts.find(id);
        if(mapIt != prev_pts.end()) {
            cv::arrowedLine(img, pts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }
}

#ifdef USE_CUDA
void FeatureTracker::drawTrackFisheye(const cv::Mat & img_up,
    const cv::Mat & img_down,
    cv::cuda::GpuMat imUpTop,
    cv::cuda::GpuMat imDownTop,
    cv::cuda::GpuMat imUpSide_cuda, 
    cv::cuda::GpuMat imDownSide_cuda) {
    cv::Mat a, b, c, d;
    imUpTop.download(a);
    imDownTop.download(b);
    imUpSide_cuda.download(c);
    imDownSide_cuda.download(d);
    drawTrackFisheye(img_up, img_down, a, b, c, d);
}
#endif

void FeatureTracker::drawTrackFisheye(const cv::Mat & img_up,
    const cv::Mat & img_down,
    cv::Mat imUpTop,
    cv::Mat imDownTop,
    cv::Mat imUpSide, 
    cv::Mat imDownSide)
{
    // ROS_INFO("Up image %d, down %d", imUp.size(), imDown.size());
    cv::Mat imTrack;
    cv::Mat fisheye_up;
    cv::Mat fisheye_down;
    
    int side_height = imUpSide.size().height;
    int width = imUpTop.size().width;
    //128
    if (img_up.size().width == 1024) {
        fisheye_up = img_up(cv::Rect(190, 62, 900, 900));
        fisheye_down = img_down(cv::Rect(190, 62, 900, 900));
    } else {
        fisheye_up = cv::Mat(cv::Size(900, 900), CV_8UC3, cv::Scalar(0, 0, 0)); 
        fisheye_down = cv::Mat(cv::Size(900, 900), CV_8UC3, cv::Scalar(0, 0, 0)); 
    }

    cv::resize(fisheye_up, fisheye_up, cv::Size(width, width));
    cv::resize(fisheye_down, fisheye_down, cv::Size(width, width));
    if (fisheye_up.channels() != 3) {
        cv::cvtColor(fisheye_up,   fisheye_up,   cv::COLOR_GRAY2BGR);
        cv::cvtColor(fisheye_down, fisheye_down, cv::COLOR_GRAY2BGR);
    }

    if (imUpTop.channels() != 3) {
        if (!imUpTop.empty()) {
            cv::cvtColor(imUpTop, imUpTop, cv::COLOR_GRAY2BGR);
        }
    
        if(!imDownTop.empty()) {
            cv::cvtColor(imDownTop, imDownTop, cv::COLOR_GRAY2BGR);
        }
        
        if(!imUpSide.empty()) {
            cv::cvtColor(imUpSide, imUpSide, cv::COLOR_GRAY2BGR);
        }

        if(!imDownSide.empty()) {
            cv::cvtColor(imDownSide, imDownSide, cv::COLOR_GRAY2BGR);
        }
    }

    if(enable_up_top) {
        drawTrackImage(imUpTop, cur_up_top_pts, ids_up_top, up_top_prevLeftPtsMap);
    }

    if(enable_down_top) {
        drawTrackImage(imDownTop, cur_down_top_pts, ids_down_top, down_top_prevLeftPtsMap);
    }

    if(enable_up_side) {
        drawTrackImage(imUpSide, cur_up_side_pts, ids_up_side, up_side_prevLeftPtsMap);
    }

    if(enable_down_side) {
        drawTrackImage(imDownSide, cur_down_side_pts, ids_down_side, down_side_prevLeftPtsMap);
    }

    //Show images
    int side_count = 3;
    if (enable_rear_side) {
        side_count = 4;
    }

    for (int i = 1; i < side_count + 1; i ++) {
        cv::line(imUpSide, cv::Point2d(i*width, 0), cv::Point2d(i*width, side_height), cv::Scalar(255, 0, 0), 1);
        cv::line(imDownSide, cv::Point2d(i*width, 0), cv::Point2d(i*width, side_height), cv::Scalar(255, 0, 0), 1);
    }

    cv::vconcat(imUpSide, imDownSide, imTrack);

    cv::Mat top_cam;


    cv::hconcat(imUpTop, imDownTop, top_cam);
    cv::hconcat(fisheye_up, top_cam, top_cam);
    cv::hconcat(top_cam, fisheye_down, top_cam); 
    // ROS_INFO("Imtrack width %d", imUpSide.size().width);
    cv::resize(top_cam, top_cam, cv::Size(imUpSide.size().width, imUpSide.size().width/4));
    
    cv::vconcat(top_cam, imTrack, imTrack);
    
    double fx = ((double)SHOW_WIDTH) / ((double) imUpSide.size().width);
    cv::resize(imTrack, imTrack, cv::Size(), fx, fx);
    cv::imshow("tracking", imTrack);
    cv::waitKey(2);
}


std::vector<cv::Mat> convertCPUMat(const std::vector<cv::cuda::GpuMat> & arr) {
    std::vector<cv::Mat> ret;
    for (const auto & mat:arr) {
        cv::Mat matcpu;
        mat.download(matcpu);
        cv::cvtColor(matcpu, matcpu, cv::COLOR_GRAY2BGR);
        ret.push_back(matcpu);
    }

    return ret;
}

#ifdef USE_CUDA
cv::cuda::GpuMat concat_side(const std::vector<cv::cuda::GpuMat> & arr) {
    int cols = arr[1].cols;
    int rows = arr[1].rows;
    if (enable_rear_side) {
        cv::cuda::GpuMat NewImg(rows, cols*4, arr[1].type()); 
        for (int i = 1; i < 5; i ++) {
            arr[i].copyTo(NewImg(cv::Rect(cols * (i-1), 0, cols, rows)));
        }
        return NewImg;
    } else {
        cv::cuda::GpuMat NewImg(rows, cols*3, arr[1].type()); 
        for (int i = 1; i < 4; i ++) {
            arr[i].copyTo(NewImg(cv::Rect(cols * (i-1), 0, cols, rows)));
        }
        return NewImg;
    }
}
#endif

cv::Mat concat_side(const std::vector<cv::Mat> & arr) {
    int cols = arr[1].cols;
    int rows = arr[1].rows;
    if (enable_rear_side) {
        cv::Mat NewImg(rows, cols*4, arr[1].type()); 
        for (int i = 1; i < 5; i ++) {
            arr[i].copyTo(NewImg(cv::Rect(cols * (i-1), 0, cols, rows)));
        }
        return NewImg;
    } else {
        cv::Mat NewImg(rows, cols*3, arr[1].type()); 
        for (int i = 1; i < 4; i ++) {
            arr[i].copyTo(NewImg(cv::Rect(cols * (i-1), 0, cols, rows)));
        }
        return NewImg;
    }
}


void FeatureTracker::addPointsFisheye()
{
    // ROS_INFO("Up top new pts %d", n_pts_up_top.size());
    for (auto &p : n_pts_up_top)
    {
        cur_up_top_pts.push_back(p);
        ids_up_top.push_back(n_id++);
        track_up_top_cnt.push_back(1);
    }

    for (auto &p : n_pts_down_top)
    {
        cur_down_top_pts.push_back(p);
        ids_down_top.push_back(n_id++);
        track_down_top_cnt.push_back(1);
    }

    for (auto &p : n_pts_up_side)
    {
        cur_up_side_pts.push_back(p);
        ids_up_side.push_back(n_id++);
        track_up_side_cnt.push_back(1);
    }
}

#ifdef USE_CUDA
void FeatureTracker::detectPoints(const cv::cuda::GpuMat & img, const cv::Mat & mask, 
    vector<cv::Point2f> & n_pts, vector<cv::Point2f> & cur_pts, int require_pts) {
    int lack_up_top_pts = require_pts - static_cast<int>(cur_pts.size());

    //Add Points Top
    TicToc tic;
    ROS_INFO("Lack %d pts; Require %d will detect %d", lack_up_top_pts, require_pts, lack_up_top_pts > require_pts/4);
    if (lack_up_top_pts > require_pts/4) {
        if(mask.empty())
            cout << "mask is empty " << endl;
        if (mask.type() != CV_8UC1)
            cout << "mask type wrong " << endl;
        
        //Detect top img
        cv::Ptr<cv::cuda::CornersDetector> detector = cv::cuda::createGoodFeaturesToTrackDetector(
            img.type(), lack_up_top_pts, 0.01, MIN_DIST);
        cv::cuda::GpuMat d_prevPts;
        cv::cuda::GpuMat gpu_mask(mask);
        detector->detect(img, d_prevPts, gpu_mask);
        // std::cout << "d_prevPts size: "<< d_prevPts.size()<<std::endl;
        if(!d_prevPts.empty()) {
            n_pts = cv::Mat_<cv::Point2f>(cv::Mat(d_prevPts));
        }
        else {
            n_pts.clear();
        }
    }
    else {
        n_pts.clear();
    }
#ifdef PERF_OUTPUT
    ROS_INFO("Detected %ld npts %fms", n_pts.size(), tic.toc());
#endif

 }

#endif

std::vector<cv::Point2f> detect_orb_by_region(const cv::Mat & _img, const cv::Mat & _mask, int features, int cols = 4, int rows = 4) {
    int small_width = _img.cols / cols;
    int small_height = _img.rows / rows;
    
    auto _orb = cv::ORB::create(10);
    std::vector<cv::Point2f> ret;
    for (int i = 0; i < cols; i ++) {
        for (int j = 0; j < rows; j ++) {
            std::vector<cv::KeyPoint> kpts;
            cv::Rect roi(small_width*i, small_height*j, small_width, small_height);
            std::cout << "ROI " << roi << "Img " << _img.size() << std::endl;
            _orb->detect(_img(roi), kpts, _mask(roi));
            printf("Detected %ld features in reigion (%d, %d)\n", kpts.size(), i, j);

            for (auto kp : kpts) {
                kp.pt.x = kp.pt.x + small_width*i;
                kp.pt.y = kp.pt.y + small_width*j;
                ret.push_back(kp.pt);
            }
        }
    }

    return ret;
}


void FeatureTracker::detectPoints(const cv::Mat & img, const cv::Mat & mask, vector<cv::Point2f> & n_pts, vector<cv::Point2f> & cur_pts, int require_pts) {
    int lack_up_top_pts = require_pts - static_cast<int>(cur_pts.size());

    //Add Points Top
    TicToc tic;
    ROS_INFO("Lost %d pts; Require %d will detect %d", lack_up_top_pts, require_pts, lack_up_top_pts > require_pts/4);
    if (lack_up_top_pts > require_pts/4) {
        if(mask.empty())
            cout << "mask is empty " << endl;
        if (mask.type() != CV_8UC1)
            cout << "mask type wrong " << endl;
        
        if (!USE_ORB) {
            cv::Mat d_prevPts;
            cv::goodFeaturesToTrack(img, d_prevPts, lack_up_top_pts, 0.01, MIN_DIST, mask);
            if(!d_prevPts.empty()) {
                n_pts = cv::Mat_<cv::Point2f>(cv::Mat(d_prevPts));
            }
            else {
                n_pts.clear();
            }
        } else {
            if (img.cols == img.rows) {
                n_pts = detect_orb_by_region(img, mask, lack_up_top_pts, 4, 4);
            } else {
                n_pts = detect_orb_by_region(img, mask, lack_up_top_pts, 4, 1);
            }
        }

    }
    else {
        n_pts.clear();
    }
#ifdef PERF_OUTPUT
    ROS_INFO("Detected %ld npts %fms", n_pts.size(), tic.toc());
#endif

 }

void FeatureTracker::setup_feature_frame(FeatureFrame & ff, vector<int> ids, vector<cv::Point2f> cur_pts, vector<cv::Point3f> cur_un_pts, vector<cv::Point3f> cur_pts_vel, int camera_id) {
    // ROS_INFO("Setup feature frame pts %ld un pts %ld vel %ld on Camera %d", cur_pts.size(), cur_un_pts.size(), cur_pts_vel.size(), camera_id);
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i];
        double x, y ,z;
        x = cur_un_pts[i].x;
        y = cur_un_pts[i].y;
        z = cur_un_pts[i].z;
        double p_u, p_v;
        p_u = cur_pts[i].x;
        p_v = cur_pts[i].y;
        double velocity_x, velocity_y, velocity_z;
        velocity_x = cur_pts_vel[i].x;
        velocity_y = cur_pts_vel[i].y;
        velocity_z = cur_pts_vel[i].z;

        TrackFeatureNoId xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y, velocity_z;

        // ROS_INFO("FeaturePts Id %d; Cam %d; pos %f, %f, %f uv %f, %f, vel %f, %f, %f", feature_id, camera_id,
            // x, y, z, p_u, p_v, velocity_x, velocity_y, velocity_z);
        ff[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }
 }


FeatureFrame FeatureTracker::setup_feature_frame() {
    FeatureFrame ff;
    setup_feature_frame(ff, ids_up_top, cur_up_top_pts, cur_up_top_un_pts, up_top_vel, 0);   
    setup_feature_frame(ff, ids_up_side, cur_up_side_pts, cur_up_side_un_pts, up_side_vel, 0);
    setup_feature_frame(ff, ids_down_top, cur_down_top_pts, cur_down_top_un_pts, down_top_vel, 1);
    setup_feature_frame(ff, ids_down_side, cur_down_side_pts, cur_down_side_un_pts, down_side_vel, 1);

    return ff;
}


vector<cv::Point3f> FeatureTracker::undistortedPtsTop(vector<cv::Point2f> &pts, FisheyeUndist & fisheye) {
    auto & cam = fisheye.cam_top;
    vector<cv::Point3f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);
        b.normalize();
#ifdef UNIT_SPHERE_ERROR
        un_pts.push_back(cv::Point3f(b.x(), b.y(), b.z()));
#else
        un_pts.push_back(cv::Point3f(b.x() / b.z(), b.y() / b.z(), 1));
#endif
    }
    return un_pts;
}


vector<cv::Point3f> FeatureTracker::undistortedPtsSide(vector<cv::Point2f> &pts, FisheyeUndist & fisheye, bool is_downward) {
    auto & cam = fisheye.cam_side;
    vector<cv::Point3f> un_pts;
    //Need to rotate pts
    //Side pos 1,2,3,4 is left front right
    //For downward camera, additational rotate 180 deg on x is required


    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        if(ENABLE_DOWNSAMPLE) {
            a = a*2;
        }

        int side_pos_id = floor(a.x() / top_size.width) + 1;

        a.x() = a.x() - floor(a.x() / top_size.width)*top_size.width;

        cam->liftProjective(a, b);

        // ROS_INFO("Pts x is %f, is at %d direction width %d", a.x(), side_pos_id, top_size.width);
        
        if (side_pos_id == 1) {
            b = t1 * b;
        } else if(side_pos_id == 2) {
            b = t2 * b;
        } else if (side_pos_id == 3) {
            b = t3 * b;
        } else if (side_pos_id == 4) {
            b = t4 * b;
        } else {
            ROS_ERROR("Err pts img pos id %d!! x %f width %d", side_pos_id, a.x(), top_size.width);
            exit(-1);
        }

        if (is_downward) {
            b = t_down * b;
        }

        b.normalize();
#ifdef UNIT_SPHERE_ERROR
        un_pts.push_back(cv::Point3f(b.x(), b.y(), b.z()));
#else
        if (fabs(b.z()) < 1e-3) {
            b.z() = 1e-3;
        }
        
        if (b.z() < - 1e-2) {
            //Is under plane, z is -1
            un_pts.push_back(cv::Point3f(b.x() / b.z(), b.y() / b.z(), -1));
        } else if (b.z() > 1e-2) {
            //Is up plane, z is 1
            un_pts.push_back(cv::Point3f(b.x() / b.z(), b.y() / b.z(), 1));
        }
#endif
    }
    return un_pts;
}


vector<cv::Point2f> FeatureTracker::opticalflow_track(vector<cv::Mat> * cur_pyr, 
                        vector<cv::Mat> * prev_pyr, vector<cv::Point2f> & prev_pts, 
                        vector<int> & ids, vector<int> & track_cnt, vector<cv::Point2f> prediction_points) const {
    if (prev_pts.size() == 0) {
        return vector<cv::Point2f>();
    }
    TicToc tic;
    vector<uchar> status;

    for (size_t i = 0; i < ids.size(); i ++) {
        int _id = ids[i];
        if (removed_pts.find(_id) == removed_pts.end()) {
            status.push_back(1);
        } else {
            status.push_back(0);
        }
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    
    if (prev_pts.size() == 0) {
        return vector<cv::Point2f>();
    }

    vector<cv::Point2f> cur_pts;
    TicToc t_og;
    status.clear();
    vector<float> err;
    cv::calcOpticalFlowPyrLK(*prev_pyr, *cur_pyr, prev_pts, cur_pts, status, err, WIN_SIZE, PYR_LEVEL);
    std::cout << "Prev pts" << prev_pts.size() << std::endl;    
    if(FLOW_BACK)
    {
        vector<cv::Point2f> reverse_pts;
        vector<uchar> reverse_status;
        cv::calcOpticalFlowPyrLK(*cur_pyr, *prev_pyr, cur_pts, reverse_pts, reverse_status, err, WIN_SIZE, PYR_LEVEL);

        for(size_t i = 0; i < status.size(); i++)
        {
            if(status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
            {
                status[i] = 1;
            }
            else
                status[i] = 0;
        }
    }
    // printf("gpu temporal optical flow costs: %f ms\n",t_og.toc());

    for (int i = 0; i < int(cur_pts.size()); i++) {
        if (status[i] && !inBorder(cur_pts[i], cur_img.size())) {
            status[i] = 0;
        }
    }            

    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(ids, status);
    if(track_cnt.size() > 0) {
        reduceVector(track_cnt, status);
    }

#ifdef PERF_OUTPUT
    ROS_INFO("Optical flow costs: %fms Pts %ld", t_og.toc(), ids.size());
#endif

    //printf("track cnt %d\n", (int)ids.size());

    for (auto &n : track_cnt)
        n++;

    return cur_pts;
}

vector<cv::Point2f> FeatureTracker::opticalflow_track(cv::Mat & cur_img, vector<cv::Mat> * cur_pyr, 
                        cv::Mat & prev_img, vector<cv::Mat> * prev_pyr, vector<cv::Point2f> & prev_pts, 
                        vector<int> & ids, vector<int> & track_cnt, vector<cv::Point2f> prediction_points) const {
    if (prev_pts.size() == 0) {
        return vector<cv::Point2f>();
    }
    TicToc tic;
    vector<uchar> status;

    for (size_t i = 0; i < ids.size(); i ++) {
        int _id = ids[i];
        if (removed_pts.find(_id) == removed_pts.end()) {
            status.push_back(1);
        } else {
            status.push_back(0);
        }
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    
    if (prev_pts.size() == 0) {
        return vector<cv::Point2f>();
    }

    vector<cv::Point2f> cur_pts;
    TicToc t_og;
    status.clear();
    vector<float> err;
    
    TicToc t_build;

    TicToc t_calc;
    cv::calcOpticalFlowPyrLK(*prev_pyr, *cur_pyr, prev_pts, cur_pts, status, err, WIN_SIZE, PYR_LEVEL);
    // cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, WIN_SIZE, PYR_LEVEL);
    // std::cout << "Track img Prev pts" << prev_pts.size() << " TS " << t_calc.toc() << std::endl;    
    if(FLOW_BACK)
    {
        vector<cv::Point2f> reverse_pts;
        vector<uchar> reverse_status;
        cv::calcOpticalFlowPyrLK(*cur_pyr, *prev_pyr, cur_pts, reverse_pts, reverse_status, err, WIN_SIZE, PYR_LEVEL);
        // cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, WIN_SIZE, PYR_LEVEL);

        for(size_t i = 0; i < status.size(); i++)
        {
            if(status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
            {
                status[i] = 1;
            }
            else
                status[i] = 0;
        }
    }
    // printf("gpu temporal optical flow costs: %f ms\n",t_og.toc());

    for (int i = 0; i < int(cur_pts.size()); i++) {
        if (status[i] && !inBorder(cur_pts[i], cur_img.size())) {
            status[i] = 0;
        }
    }            

    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(ids, status);
    if(track_cnt.size() > 0) {
        reduceVector(track_cnt, status);
    }

    // std::cout << "Cur pts" << cur_pts.size() << std::endl;


#ifdef PERF_OUTPUT
    ROS_INFO("Optical flow costs: %fms Pts %ld", t_og.toc(), ids.size());
#endif

    //printf("track cnt %d\n", (int)ids.size());

    for (auto &n : track_cnt)
        n++;

    return cur_pts;
} 
#ifdef USE_CUDA
vector<cv::Point2f> FeatureTracker::opticalflow_track(cv::cuda::GpuMat & cur_img, 
                        cv::cuda::GpuMat & prev_img, vector<cv::Point2f> & prev_pts, 
                        vector<int> & ids, vector<int> & track_cnt,
                        bool is_lr_track, vector<cv::Point2f> prediction_points){
    if (prev_pts.size() == 0) {
        return vector<cv::Point2f>();
    }
    TicToc tic;
    vector<uchar> status;

    for (size_t i = 0; i < ids.size(); i ++) {
        int _id = ids[i];
        if (removed_pts.find(_id) == removed_pts.end()) {
            status.push_back(1);
        } else {
            status.push_back(0);
        }
    }

    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(ids, status);
    
    if (prev_pts.size() == 0) {
        return vector<cv::Point2f>();
    }

    vector<cv::Point2f> cur_pts;
    TicToc t_og;
    cv::cuda::GpuMat prev_gpu_pts(prev_pts);
    cv::cuda::GpuMat cur_gpu_pts(cur_pts);
    cv::cuda::GpuMat gpu_status;
    status.clear();

    //Assume No Prediction Need to add later
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
        cv::Size(21, 21), 3, 30, false);
    d_pyrLK_sparse->calc(prev_img, cur_img, prev_gpu_pts, cur_gpu_pts, gpu_status);
    
    // std::cout << "Prev gpu pts" << prev_gpu_pts.size() << std::endl;    
    // std::cout << "Cur gpu pts" << cur_gpu_pts.size() << std::endl;
    cur_gpu_pts.download(cur_pts);

    gpu_status.download(status);
    if(FLOW_BACK)
    {
        // ROS_INFO("Is flow back");
        cv::cuda::GpuMat reverse_gpu_status;
        cv::cuda::GpuMat reverse_gpu_pts = prev_gpu_pts;
        cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
        cv::Size(21, 21), 1, 30, true);
        d_pyrLK_sparse->calc(cur_img, prev_img, cur_gpu_pts, reverse_gpu_pts, reverse_gpu_status);

        vector<cv::Point2f> reverse_pts(reverse_gpu_pts.cols);
        reverse_gpu_pts.download(reverse_pts);

        vector<uchar> reverse_status(reverse_gpu_status.cols);
        reverse_gpu_status.download(reverse_status);

        for(size_t i = 0; i < status.size(); i++)
        {
            if(status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
            {
                status[i] = 1;
            }
            else
                status[i] = 0;
        }
    }
    // printf("gpu temporal optical flow costs: %f ms\n",t_og.toc());

    for (int i = 0; i < int(cur_pts.size()); i++){
        if (status[i] && !inBorder(cur_pts[i], cur_img.size())) {
            status[i] = 0;
        }
    }            

    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(ids, status);
    if(track_cnt.size() > 0) {
        reduceVector(track_cnt, status);
    }

#ifdef PERF_OUTPUT
    ROS_INFO("Optical flow costs: %fms Pts %ld", t_og.toc(), ids.size());
#endif

    //printf("track cnt %d\n", (int)ids.size());

    for (auto &n : track_cnt)
        n++;

    return cur_pts;
}
#endif

#ifdef WITH_VWORKS

pair<vector<cv::Point2f>, vector<int>> vxarray2cv_pts(vx_array fVx, bool output=false) {
    std::vector<cv::Point2f> fPts;
    vector<int> status;
    vx_size numItems = 0;
    vxQueryArray(fVx, VX_ARRAY_ATTRIBUTE_NUMITEMS, &numItems, sizeof(numItems));
    vx_size stride = sizeof(vx_size);
    void *base = NULL;
    vxAccessArrayRange(fVx, 0, numItems, &stride, &base, VX_READ_ONLY);

    //For tracker status
    // Holds tracking status. Zero indicates a lost point. Initialized to 1 by corner detectors.

    //The primitive uses tracking_status information for input points (VX_TYPE_KEYPOINT, NVX_TYPE_KEYPOINTF) and updates only points with non-zero tracking_status. 
    // The points with tracking_status == 0 gets copied to the output array as is.
    // The VisionWorks corner detectors (FastCorners, HarrisCorners, FAST Track, Harris Track) 
    // initialize the tracking_status field of detected points to 1.
    for (vx_size i = 0; i < numItems; i++)
    {
        nvx_keypointf_t* points = (nvx_keypointf_t*)base;
        vx_float32 error = points[i].error;
        vx_float32 orientation = points[i].orientation;
        vx_float32 scale = points[i].scale;
        vx_float32 strength = points[i].strength;
        vx_int32 trackingStatus = points[i].tracking_status;
        vx_float32 x = points[i].x;
        vx_float32 y = points[i].y;
        if (output) {
            std::cout << "index: " << i
                    // << ":: error:          " << error << std::endl
                    // << ":: orientation:    " << orientation << std::endl
                    // << ":: scale:          " << scale << std::endl
                    // << ":: strength:       " << strength << std::endl
                    << ":: status: " << trackingStatus
                    << ":: x:   " << x
                    << ":: y:   " << y << std::endl;
        }
        fPts.push_back(cv::Point2f(x, y));
        status.push_back((int)trackingStatus);
    }
    return pair<vector<cv::Point2f>, vector<int>>(fPts, status);
}


// tracker_up_top->printPerfs();


// //In cur pts 255 is keep tracking point
// //0 is the new pts
// ROS_INFO("PREV PTS");
// auto cv_prev_pts = vxarray2cv_pts(prev_pts);
// ROS_INFO("CUR PTS");
// auto cv_cur_pts = vxarray2cv_pts(cur_pts);
// ROS_INFO("VWorks track cost %fms cv pts %ld", tic.toc(), cv_cur_pts.first.size());
// cv::cuda::GpuMat up_top_img_Debug;
// cv::Mat uptop_debug;
// up_side_img.copyTo(up_top_img_Debug);
// up_top_img_Debug.download(uptop_debug);

int to_pt_pos_id(const cv::Point2f & pt) {
    return floor(pt.x * 100000) + floor(pt.y*100);
}

void FeatureTracker::process_vworks_tracking(nvx::FeatureTracker* _tracker, vector<int> & _ids, vector<cv::Point2f> & prev_pts, vector<cv::Point2f> & cur_pts, 
        vector<int> &track, vector<cv::Point2f> & n_pts, map<int, int> & _id_by_index, bool debug_output) {
    auto prev_ids = _ids;
    map<int, int> new_id_by_index;
    map<int, int> _track;
    for (unsigned int i = 0; i < track.size(); i ++) {
        _track[_ids[i]] = track[i];
    }

    _ids.clear();
    prev_ids.clear();
    prev_pts.clear();
    cur_pts.clear();

    auto vx_prev_pts_ = _tracker->getPrevFeatures();
    auto vx_cur_pts_ = _tracker->getCurrFeatures();

    auto cv_cur_pts_flag = vxarray2cv_pts(vx_cur_pts_, false);
    auto cv_prev_pts_flag = vxarray2cv_pts(vx_prev_pts_, false);
    auto cv_cur_pts = cv_cur_pts_flag.first;
    auto cv_cur_flags = cv_cur_pts_flag.second;
    bool first_frame = _id_by_index.empty();


    //For new point; prev is 1 cur is 255
    //For old point; prev and cur is 255
    //1 is create by FAST
    //255 is track by opticalflow
    //Now we always use tracked 2 frame point instead of full
    //This is because the vworks tracker
    if (!first_frame) {
        for (unsigned int i = 0; i < cv_cur_pts.size(); i ++) {
            if (cv_cur_flags[i] == 0) {
                //This is failed point
                continue;
            }
            int prev_pos_id = to_pt_pos_id(cv_prev_pts_flag.first[i]);
            int cur_pos_id = to_pt_pos_id(cv_cur_pts[i]);
            if (_id_by_index.find(prev_pos_id) != _id_by_index.end()) {
                //This is keep tracking point
                int _id = _id_by_index[prev_pos_id];
                new_id_by_index[cur_pos_id] = _id;

                _ids.push_back(_id);
                prev_pts.push_back(cv_prev_pts_flag.first[i]);
                cur_pts.push_back(cv_cur_pts[i]);
                if (debug_output) {
                    ROS_INFO("Index %d ID %d POSID %d PrevID %d PT %f %f ->  %f %f FLAG %d from %d",
                        i,
                        _ids.back(),
                        cur_pos_id,
                        prev_pos_id,
                        prev_pts.back().x, prev_pts.back().y,
                        cur_pts.back().x, cur_pts.back().y,
                        cv_cur_flags[i], cv_prev_pts_flag.second[i]
                    );
                }
                _track[_id] ++;
            }
        }
    }

    for (unsigned int i = 0; i < cv_cur_pts.size(); i ++) {
        if (cv_cur_flags[i] == 0) {
            //This is failed point
            continue;
        }
        int prev_pos_id = to_pt_pos_id(cv_prev_pts_flag.first[i]);
        //This create new points
        if (_id_by_index.find(prev_pos_id) == _id_by_index.end()) {
            //This is new create points
            int cur_pos_id = to_pt_pos_id(cv_cur_pts[i]);
            int prev_pos_id = to_pt_pos_id(cv_prev_pts_flag.first[i]);
            cur_pts.push_back(cv_cur_pts[i]);
            _ids.push_back(n_id++);
            new_id_by_index[cur_pos_id] = _ids.back();
            _track[_ids.back()] = 1;
            if (debug_output) {
                ROS_INFO("New ID %d pos_id %d  prev_id %d PT %f %f CUR %d PREV %d", _ids.back(), 
                    cur_pos_id, prev_pos_id, cv_cur_pts[i].x, cv_cur_pts[i].y, cv_cur_flags[i], cv_prev_pts_flag.second[i]);
            }
        }
    }

    track.clear();
    for (unsigned int i = 0; i < _ids.size(); i ++) {
        int cur_pos_id = to_pt_pos_id(cv_cur_pts[i]);
        int _id = new_id_by_index[cur_pos_id];
        track.push_back(_track[_id]);

        if (debug_output) {
            ROS_INFO("ID %d POSID %d Pos %f %f",
                _id, cur_pos_id, cv_cur_pts[i].x, cv_cur_pts[i].y);
        }
    }
    
    _id_by_index = new_id_by_index;
}

void FeatureTracker::init_vworks_tracker(cv::cuda::GpuMat & up_top_img, cv::cuda::GpuMat & down_top_img, cv::cuda::GpuMat & up_side_img, cv::cuda::GpuMat & down_side_img) {
    context = VX_API_CALL(vxCreateContext());

    if (enable_up_top) {
        vx_up_top_image = nvx_cv::createVXImageFromCVGpuMat(context, up_top_img_fix);
        vx_up_top_mask = nvx_cv::createVXImageFromCVGpuMat(context, mask_up_top_fix); 
    }

    if(enable_down_top) {
        vx_down_top_image = nvx_cv::createVXImageFromCVGpuMat(context, down_top_img_fix);
        vx_down_top_mask = nvx_cv::createVXImageFromCVGpuMat(context, mask_down_top_fix); 
    }

    if(enable_up_side) {
        vx_up_side_image = nvx_cv::createVXImageFromCVGpuMat(context, up_side_img_fix);
        vx_up_side_mask = nvx_cv::createVXImageFromCVGpuMat(context, mask_up_side_fix); 
    }

    if(enable_down_side) {
        vx_down_side_image = nvx_cv::createVXImageFromCVGpuMat(context, down_side_img_fix);
    }

    
    nvx::FeatureTracker::Params params;
    params.use_rgb = RGB_DEPTH_CLOUD;
    params.use_harris_detector = false;
    // params.use_harris_detector = true;
    // params.harris_k = 0.04;
    // params.harris_thresh = 10;
    params.array_capacity = TOP_PTS_CNT;
    params.fast_thresh = 10;

    params.lk_win_size = 21;
    params.detector_cell_size = MIN_DIST;
    if (enable_up_top) {
        tracker_up_top = nvx::FeatureTracker::create(context, params);
        tracker_up_top->init(vx_up_top_image, vx_up_top_mask);
    }
    if(enable_down_top) {
        tracker_down_top = nvx::FeatureTracker::create(context, params);
        tracker_down_top->init(vx_down_top_image, vx_down_top_mask);
    }
    params.detector_cell_size = MIN_DIST;
    params.array_capacity = SIDE_PTS_CNT;

    if(enable_up_side) {
        tracker_up_side = nvx::FeatureTracker::create(context, params);
        tracker_up_side->init(vx_up_side_image, vx_up_side_mask);
    }
}

#endif


map<int, cv::Point2f> pts_map(vector<int> ids, vector<cv::Point2f> cur_pts) {
    map<int, cv::Point2f> prevMap;
    for (unsigned int i = 0; i < ids.size(); i ++) {
        prevMap[ids[i]] = cur_pts[i];
    }
    return prevMap;
}

FeatureFrame FeatureTracker::trackImage_fisheye(double _cur_time, const std::vector<cv::Mat> & fisheye_imgs_up, const std::vector<cv::Mat> & fisheye_imgs_down) {
    // ROS_INFO("tracking fisheye cpu %ld:%ld", fisheye_imgs_up.size(), fisheye_imgs_down.size());
    cur_time = _cur_time;
    static double count = 0;
    count += 1;

    TicToc t_r;

    cv::Mat up_side_img = concat_side(fisheye_imgs_up);
    cv::Mat down_side_img = concat_side(fisheye_imgs_down);
    cv::Mat up_top_img = fisheye_imgs_up[0];
    cv::Mat down_top_img = fisheye_imgs_down[0];

    std::vector<cv::Mat> * up_top_pyr = nullptr, * down_top_pyr = nullptr, * up_side_pyr = nullptr, * down_side_pyr = nullptr;
    double concat_cost = t_r.toc();

    top_size = up_top_img.size();
    side_size = up_side_img.size();

    //Clear All current pts
    cur_up_top_pts.clear();
    cur_up_side_pts.clear();
    cur_down_top_pts.clear();
    cur_down_side_pts.clear();

    cur_up_top_un_pts.clear();
    cur_up_side_un_pts.clear();
    cur_down_top_un_pts.clear();
    cur_down_side_un_pts.clear();


    TicToc t_pyr;
    #pragma omp parallel sections 
    {
        #pragma omp section 
        {
            if(enable_up_top) {
                // printf("Building up top pyr\n");
                up_top_pyr = new std::vector<cv::Mat>();
                cv::buildOpticalFlowPyramid(up_top_img, *up_top_pyr, WIN_SIZE, PYR_LEVEL, true);//, cv::BORDER_REFLECT101, cv::BORDER_CONSTANT, false);
            }
        }
        
        #pragma omp section 
        {
            if(enable_down_top) {
                // printf("Building down top pyr\n");
                down_top_pyr = new std::vector<cv::Mat>();
                cv::buildOpticalFlowPyramid(down_top_img, *down_top_pyr, WIN_SIZE, PYR_LEVEL, true);
            }
        }
        
        #pragma omp section 
        {
            if(enable_up_side) {
                // printf("Building up side pyr\n");
                up_side_pyr = new std::vector<cv::Mat>();
                cv::buildOpticalFlowPyramid(up_side_img, *up_side_pyr, WIN_SIZE, PYR_LEVEL, true);
            }
        }
        
        #pragma omp section 
        {
            if(enable_down_side) {
                // printf("Building downn side pyr\n");
                down_side_pyr = new std::vector<cv::Mat>();
                cv::buildOpticalFlowPyramid(down_side_img, *down_side_pyr, WIN_SIZE, PYR_LEVEL, true);
            }
        }
    }

    static double pyr_sum = 0;
    pyr_sum += t_pyr.toc();

    TicToc t_t;
    #pragma omp parallel sections
    {
        #pragma omp section 
        {
            //If has predict;
            if (enable_up_top) {
                // printf("Start track up top\n");
                cur_up_top_pts = opticalflow_track(up_top_img, up_top_pyr, prev_up_top_img_cpu, prev_up_top_pyr, prev_up_top_pts, ids_up_top, track_up_top_cnt);
                // printf("End track up top\n");
            }
        }

        #pragma omp section 
        {
            if (enable_up_side) {
                // printf("Start track up side\n");
                cur_up_side_pts = opticalflow_track(up_side_img, up_side_pyr, prev_up_side_img_cpu, prev_up_side_pyr, prev_up_side_pts, ids_up_side, track_up_side_cnt);
                // printf("End track up side\n");
            }
        }

        #pragma omp section 
        {
            if (enable_down_top) {
                // printf("Start track down top\n");
                cur_down_top_pts = opticalflow_track(down_top_img, down_top_pyr, prev_down_top_img_cpu, prev_down_top_pyr, prev_down_top_pts, ids_down_top, track_down_top_cnt);
                // printf("End track down top\n");
            }
        }

        
       
    }
    

    static double lk_sum = 0;
    lk_sum += t_t.toc();

    TicToc t_d;

    setMaskFisheye();

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (enable_up_top) {
                detectPoints(up_top_img, mask_up_top, n_pts_up_top, cur_up_top_pts, TOP_PTS_CNT);
            }
        }

        #pragma omp section
        {
            if (enable_down_top) {
                detectPoints(down_top_img, mask_down_top, n_pts_down_top, cur_down_top_pts, TOP_PTS_CNT);
            }
        }

        #pragma omp section
        {
            if (enable_up_side) {
                detectPoints(up_side_img, mask_up_side, n_pts_up_side, cur_up_side_pts, SIDE_PTS_CNT);
            }
        }
    }

    ROS_INFO("Detect cost %fms", t_d.toc());

    static double detect_sum = 0;

    detect_sum = detect_sum + t_d.toc();

    addPointsFisheye();
    
    TicToc t_tk;
    {
        if (enable_down_side) {
            ids_down_side = ids_up_side;
            std::vector<cv::Point2f> down_side_init_pts = cur_up_side_pts;
            if (down_side_init_pts.size() > 0) {
                cur_down_side_pts = opticalflow_track(down_side_img, down_side_pyr, up_side_img, up_side_pyr, down_side_init_pts, ids_down_side, track_down_side_cnt);
            }
        }
    }

    ROS_INFO("Tracker 2 cost %fms", t_tk.toc());

    //Undist points
    cur_up_top_un_pts = undistortedPtsTop(cur_up_top_pts, fisheys_undists[0]);
    cur_down_top_un_pts = undistortedPtsTop(cur_down_top_pts, fisheys_undists[1]);

    cur_up_side_un_pts = undistortedPtsSide(cur_up_side_pts, fisheys_undists[0], false);
    cur_down_side_un_pts = undistortedPtsSide(cur_down_side_pts, fisheys_undists[1], true);

    //Calculate Velocitys
    up_top_vel = ptsVelocity3D(ids_up_top, cur_up_top_un_pts, cur_up_top_un_pts_map, prev_up_top_un_pts_map);
    down_top_vel = ptsVelocity3D(ids_down_top, cur_down_top_un_pts, cur_down_top_un_pts_map, prev_down_top_un_pts_map);

    up_side_vel = ptsVelocity3D(ids_up_side, cur_up_side_un_pts, cur_up_side_un_pts_map, prev_up_side_un_pts_map);
    down_side_vel = ptsVelocity3D(ids_down_side, cur_down_side_un_pts, cur_down_side_un_pts_map, prev_down_side_un_pts_map);

    // ROS_INFO("Up top VEL %ld", up_top_vel.size());
    double tcost_all = t_r.toc();
    if (SHOW_TRACK) {
        drawTrackFisheye(cv::Mat(), cv::Mat(), up_top_img, down_top_img, up_side_img, down_side_img);
    }

        
    prev_up_top_img_cpu = up_top_img;
    prev_down_top_img_cpu = down_top_img;
    prev_up_side_img_cpu = up_side_img;

    if(prev_down_top_pyr != nullptr) {
        delete prev_down_top_pyr;
    }

    if(prev_up_top_pyr != nullptr) {
        delete prev_up_top_pyr;
    }

    if (prev_up_side_pyr!=nullptr) {
        delete prev_up_side_pyr;
    }

    if (down_side_pyr!=nullptr) {
        delete down_side_pyr;
    }

    prev_down_top_pyr = down_top_pyr;
    prev_up_top_pyr = up_top_pyr;
    prev_up_side_pyr = up_side_pyr;

    prev_up_top_pts = cur_up_top_pts;
    prev_down_top_pts = cur_down_top_pts;
    prev_up_side_pts = cur_up_side_pts;
    prev_down_side_pts = cur_down_side_pts;

    prev_up_top_un_pts = cur_up_top_un_pts;
    prev_down_top_un_pts = cur_down_top_un_pts;
    prev_up_side_un_pts = cur_up_side_un_pts;
    prev_down_side_un_pts = cur_down_side_un_pts;

    prev_up_top_un_pts_map = cur_up_top_un_pts_map;
    prev_down_top_un_pts_map = cur_down_top_un_pts_map;
    prev_up_side_un_pts_map = cur_up_side_un_pts_map;
    prev_down_side_un_pts_map = cur_up_side_un_pts_map;
    prev_time = cur_time;

    up_top_prevLeftPtsMap = pts_map(ids_up_top, cur_up_top_pts);
    down_top_prevLeftPtsMap = pts_map(ids_down_top, cur_down_top_pts);
    up_side_prevLeftPtsMap = pts_map(ids_up_side, cur_up_side_pts);
    down_side_prevLeftPtsMap = pts_map(ids_down_side, cur_down_side_pts);

    // hasPrediction = false;
    auto ff = setup_feature_frame();
    
    static double whole_sum = 0.0;

    whole_sum += t_r.toc();

    printf("FT Whole %fms; AVG %fms\n DetectAVG %fms PYRAvg %fms LKAvg %fms Concat %fms PTS %ld T\n", 
        t_r.toc(), whole_sum/count, detect_sum/count, pyr_sum/count, lk_sum/count, concat_cost, ff.size());
    return ff;
}


#ifdef USE_CUDA
FeatureFrame FeatureTracker::trackImage_fisheye(double _cur_time,   
        const std::vector<cv::cuda::GpuMat> & fisheye_imgs_up,
        const std::vector<cv::cuda::GpuMat> & fisheye_imgs_down) {
    cur_time = _cur_time;

    TicToc t_r;
    cv::cuda::GpuMat up_side_img = concat_side(fisheye_imgs_up);
    cv::cuda::GpuMat down_side_img = concat_side(fisheye_imgs_down);
    cv::cuda::GpuMat up_top_img = fisheye_imgs_up[0];
    cv::cuda::GpuMat down_top_img = fisheye_imgs_down[0];
    double concat_cost = t_r.toc();

    top_size = up_top_img.size();
    side_size = up_side_img.size();

    //Clear All current pts
    cur_up_top_pts.clear();
    cur_up_side_pts.clear();
    cur_down_top_pts.clear();
    cur_down_side_pts.clear();

    cur_up_top_un_pts.clear();
    cur_up_side_un_pts.clear();
    cur_down_top_un_pts.clear();
    cur_down_side_un_pts.clear();

    if(USE_VXWORKS) {
#ifndef WITH_VWORKS
        ROS_ERROR("You must set enable_vworks to true or disable vworks in VINS config file");
        exit(-1);
#else
        TicToc tic;
        //TODO: simpified this to make no copy
        if (enable_up_top) {
            up_top_img.copyTo(up_top_img_fix);
        }

        if(enable_down_top) {
            down_top_img.copyTo(down_top_img_fix);
        }

        if(enable_up_side) {
            up_side_img.copyTo(up_side_img_fix);
        }

        if(enable_down_side) {
            down_side_img.copyTo(down_side_img_fix);
        }

        ROS_INFO("Copy Image cost %fms", tic.toc());
        if(first_frame) {
            setMaskFisheye();
            if (enable_up_top) {
                mask_up_top_fix.upload(mask_up_top);
            }
    
            if(enable_down_top) {
                mask_down_top_fix.upload(mask_down_top);
            }

            if(enable_up_side) {
                mask_up_side_fix.upload(mask_up_side);
            }
            ROS_INFO("setFisheyeMask Image cost %fms", tic.toc());

            init_vworks_tracker(up_top_img_fix, down_top_img_fix, up_side_img_fix, down_side_img_fix);
            first_frame = false;
        } else {
            if (enable_up_top) {
                tracker_up_top->track(vx_up_top_image, vx_up_top_mask);
            }

            if(enable_down_top) {
                tracker_down_top->track(vx_down_top_image, vx_down_top_mask);
            }

            if(enable_up_side) {
                tracker_up_side->track(vx_up_side_image, vx_up_side_mask);
            }
        }
        
        ROS_INFO("Track only cost %fms", tic.toc());

        if (enable_up_top) {
            process_vworks_tracking(tracker_up_top,  ids_up_top, prev_up_top_pts, cur_up_top_pts, 
                track_up_top_cnt, n_pts_up_top, up_top_id_by_index);
        }

        if(enable_down_top) {
            process_vworks_tracking(tracker_down_top,  ids_down_top, prev_down_top_pts, cur_down_top_pts, 
                track_down_top_cnt, n_pts_down_top, down_top_id_by_index);
        }

        if(enable_up_side) {
            process_vworks_tracking(tracker_up_side,  ids_up_side, prev_up_side_pts, cur_up_side_pts, 
                track_up_side_cnt, n_pts_up_side, up_side_id_by_index);
        }
        
        ROS_INFO("Visionworks cost %fms", tic.toc());

        if (enable_down_side) {
            ids_down_side = ids_up_side;
            vector<cv::Point2f> down_side_init_pts = cur_up_side_pts;
            cur_down_side_pts = opticalflow_track(down_side_img, up_side_img, down_side_init_pts, ids_down_side, track_down_side_cnt, FLOW_BACK);
            // ROS_INFO("Down side try to track %ld pts; gives %ld:%ld", cur_up_side_pts.size(), cur_down_side_pts.size(), ids_down_side.size());
        }
#endif
    } else {
        if (up_top_img.channels() == 3) {
            cv::cuda::cvtColor(up_top_img, up_top_img, cv::COLOR_BGR2GRAY);
            cv::cuda::cvtColor(down_top_img, down_top_img, cv::COLOR_BGR2GRAY);
            cv::cuda::cvtColor(up_side_img, up_side_img, cv::COLOR_BGR2GRAY);
            cv::cuda::cvtColor(down_side_img, down_side_img, cv::COLOR_BGR2GRAY);
        }

        ROS_INFO("CVT Color %fms", t_r.toc());
        
        //If has predict;
        if (enable_up_top) {
            // ROS_INFO("Tracking top");
            cur_up_top_pts = opticalflow_track(up_top_img, prev_up_top_img, prev_up_top_pts, ids_up_top, track_up_top_cnt, false);
        }
        if (enable_up_side) {
            cur_up_side_pts = opticalflow_track(up_side_img, prev_up_side_img, prev_up_side_pts, ids_up_side, track_up_side_cnt, false);
        }

        if (enable_down_top) {
            cur_down_top_pts = opticalflow_track(down_top_img, prev_down_top_img, prev_down_top_pts, ids_down_top, track_down_top_cnt, false);
        }
        
        ROS_INFO("FT %fms", t_r.toc());

        setMaskFisheye();

        ROS_INFO("SetMaskFisheye %fms", t_r.toc());
        
        TicToc t_d;
        if (enable_up_top) {
            // ROS_INFO("Detecting top");
            detectPoints(up_top_img, mask_up_top, n_pts_up_top, cur_up_top_pts, TOP_PTS_CNT);
        }
        if (enable_down_top) {
            detectPoints(down_top_img, mask_down_top, n_pts_down_top, cur_down_top_pts, TOP_PTS_CNT);
        }

        if (enable_up_side) {
            detectPoints(up_side_img, mask_up_side, n_pts_up_side, cur_up_side_pts, SIDE_PTS_CNT);
        }

        ROS_INFO("DetectPoints %fms", t_d.toc());

        addPointsFisheye();

        if (enable_down_side) {
            ids_down_side = ids_up_side;
            std::vector<cv::Point2f> down_side_init_pts = cur_up_side_pts;
            cur_down_side_pts = opticalflow_track(down_side_img, up_side_img, down_side_init_pts, ids_down_side, track_down_side_cnt, true);
            // ROS_INFO("Down side try to track %ld pts; gives %ld:%ld", cur_up_side_pts.size(), cur_down_side_pts.size(), ids_down_side.size());
        }
    }

    //Undist points
    cur_up_top_un_pts = undistortedPtsTop(cur_up_top_pts, fisheys_undists[0]);
    cur_down_top_un_pts = undistortedPtsTop(cur_down_top_pts, fisheys_undists[1]);

    cur_up_side_un_pts = undistortedPtsSide(cur_up_side_pts, fisheys_undists[0], false);
    cur_down_side_un_pts = undistortedPtsSide(cur_down_side_pts, fisheys_undists[1], true);

    //Calculate Velocitys
    up_top_vel = ptsVelocity3D(ids_up_top, cur_up_top_un_pts, cur_up_top_un_pts_map, prev_up_top_un_pts_map);
    down_top_vel = ptsVelocity3D(ids_down_top, cur_down_top_un_pts, cur_down_top_un_pts_map, prev_down_top_un_pts_map);

    up_side_vel = ptsVelocity3D(ids_up_side, cur_up_side_un_pts, cur_up_side_un_pts_map, prev_up_side_un_pts_map);
    down_side_vel = ptsVelocity3D(ids_down_side, cur_down_side_un_pts, cur_down_side_un_pts_map, prev_down_side_un_pts_map);

    // ROS_INFO("Up top VEL %ld", up_top_vel.size());
    double tcost_all = t_r.toc();
    if (SHOW_TRACK) {
        drawTrackFisheye(cv::Mat(), cv::Mat(), up_top_img, down_top_img, up_side_img, down_side_img);
    }
        
    prev_up_top_img = up_top_img;
    prev_down_top_img = down_top_img;
    prev_up_side_img = up_side_img;

    prev_up_top_pts = cur_up_top_pts;
    prev_down_top_pts = cur_down_top_pts;
    prev_up_side_pts = cur_up_side_pts;
    prev_down_side_pts = cur_down_side_pts;

    prev_up_top_un_pts = cur_up_top_un_pts;
    prev_down_top_un_pts = cur_down_top_un_pts;
    prev_up_side_un_pts = cur_up_side_un_pts;
    prev_down_side_un_pts = cur_down_side_un_pts;

    prev_up_top_un_pts_map = cur_up_top_un_pts_map;
    prev_down_top_un_pts_map = cur_down_top_un_pts_map;
    prev_up_side_un_pts_map = cur_up_side_un_pts_map;
    prev_down_side_un_pts_map = cur_up_side_un_pts_map;
    prev_time = cur_time;

    up_top_prevLeftPtsMap = pts_map(ids_up_top, cur_up_top_pts);
    down_top_prevLeftPtsMap = pts_map(ids_down_top, cur_down_top_pts);
    up_side_prevLeftPtsMap = pts_map(ids_up_side, cur_up_side_pts);
    down_side_prevLeftPtsMap = pts_map(ids_down_side, cur_down_side_pts);

    // hasPrediction = false;
    auto ff = setup_feature_frame();

    printf("FT Whole %fms; MainProcess %fms concat %fms PTS %ld T\n", t_r.toc(), tcost_all, concat_cost, ff.size());
    return ff;
}

#endif

FeatureFrame FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1)
{
    TicToc t_r;
    cur_time = _cur_time;
    cv::Mat rightImg;
    cv::cuda::GpuMat right_gpu_img;

    if (USE_GPU) {
#ifdef USE_CUDA
        TicToc t_g;
        cur_gpu_img = cv::cuda::GpuMat(_img);
        right_gpu_img = cv::cuda::GpuMat(_img1);

        printf("gpumat cost: %fms\n",t_g.toc());

        row = _img.rows;
        col = _img.cols;
        if(SHOW_TRACK) {
            cur_img = _img;
            rightImg = _img1;
        }
        if(ENABLE_DOWNSAMPLE) {
            cv::cuda::resize(cur_gpu_img, cur_gpu_img, cv::Size(), 0.5, 0.5);
            cv::cuda::resize(right_gpu_img, right_gpu_img, cv::Size(), 0.5, 0.5);
            row = _img.rows/2;
            col = _img.cols/2;
        }
#endif
    } else {
        if(ENABLE_DOWNSAMPLE) {
            cv::resize(_img, cur_img, cv::Size(), 0.5, 0.5);
            cv::resize(_img1, rightImg, cv::Size(), 0.5, 0.5);
            row = _img.rows/2;
            col = _img.cols/2;
        } else {
            cur_img = _img;
            rightImg = _img1;
            row = _img.rows;
            col = _img.cols;
        }
    }

    /*
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(cur_img, cur_img);
        if(!rightImg.empty())
            clahe->apply(rightImg, rightImg);
    }
    */
    cur_pts.clear();

    if (prev_pts.size() > 0)
    {
        vector<uchar> status;
        if(!USE_GPU)
        {
            TicToc t_o;
            
            vector<float> err;
            if(hasPrediction)
            {
                cur_pts = predict_pts;
                cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 1, 
                cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
                
                int succ_num = 0;
                for (size_t i = 0; i < status.size(); i++)
                {
                    if (status[i])
                        succ_num++;
                }
                if (succ_num < 10)
                cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
            }
            else
                cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
            // reverse check
            if(FLOW_BACK)
            {
                vector<uchar> reverse_status;
                vector<cv::Point2f> reverse_pts = prev_pts;
                cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1, 
                cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
                //cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3); 
                for(size_t i = 0; i < status.size(); i++)
                {
                    if(status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
                    {
                        status[i] = 1;
                    }
                    else
                        status[i] = 0;
                }
            }
            // printf("temporal optical flow costs: %fms\n", t_o.toc());
        }
        else
        {
#ifdef USE_CUDA
            TicToc t_og;
            cv::cuda::GpuMat prev_gpu_pts(prev_pts);
            cv::cuda::GpuMat cur_gpu_pts(cur_pts);
            cv::cuda::GpuMat gpu_status;
            if(hasPrediction)
            {
                cur_gpu_pts = cv::cuda::GpuMat(predict_pts);
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                cv::Size(21, 21), 1, 30, true);
                d_pyrLK_sparse->calc(prev_gpu_img, cur_gpu_img, prev_gpu_pts, cur_gpu_pts, gpu_status);
                
                vector<cv::Point2f> tmp_cur_pts(cur_gpu_pts.cols);
                cur_gpu_pts.download(tmp_cur_pts);
                cur_pts = tmp_cur_pts;

                vector<uchar> tmp_status(gpu_status.cols);
                gpu_status.download(tmp_status);
                status = tmp_status;

                int succ_num = 0;
                for (size_t i = 0; i < tmp_status.size(); i++)
                {
                    if (tmp_status[i])
                        succ_num++;
                }
                if (succ_num < 10)
                {
                    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                    cv::Size(21, 21), 3, 30, false);
                    d_pyrLK_sparse->calc(prev_gpu_img, cur_gpu_img, prev_gpu_pts, cur_gpu_pts, gpu_status);

                    vector<cv::Point2f> tmp1_cur_pts(cur_gpu_pts.cols);
                    cur_gpu_pts.download(tmp1_cur_pts);
                    cur_pts = tmp1_cur_pts;

                    vector<uchar> tmp1_status(gpu_status.cols);
                    gpu_status.download(tmp1_status);
                    status = tmp1_status;
                }
            }
            else
            {
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                cv::Size(21, 21), 3, 30, false);
                d_pyrLK_sparse->calc(prev_gpu_img, cur_gpu_img, prev_gpu_pts, cur_gpu_pts, gpu_status);

                vector<cv::Point2f> tmp1_cur_pts(cur_gpu_pts.cols);
                cur_gpu_pts.download(tmp1_cur_pts);
                cur_pts = tmp1_cur_pts;

                vector<uchar> tmp1_status(gpu_status.cols);
                gpu_status.download(tmp1_status);
                status = tmp1_status;
            }
            if(FLOW_BACK)
            {
                cv::cuda::GpuMat reverse_gpu_status;
                cv::cuda::GpuMat reverse_gpu_pts = prev_gpu_pts;
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                cv::Size(21, 21), 1, 30, true);
                d_pyrLK_sparse->calc(cur_gpu_img, prev_gpu_img, cur_gpu_pts, reverse_gpu_pts, reverse_gpu_status);

                vector<cv::Point2f> reverse_pts(reverse_gpu_pts.cols);
                reverse_gpu_pts.download(reverse_pts);

                vector<uchar> reverse_status(reverse_gpu_status.cols);
                reverse_gpu_status.download(reverse_status);

                for(size_t i = 0; i < status.size(); i++)
                {
                    if(status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
                    {
                        status[i] = 1;
                    }
                    else
                        status[i] = 0;
                }
            }
            // printf("gpu temporal optical flow costs: %f ms\n",t_og.toc());
#endif
        }
    
        for (int i = 0; i < int(cur_pts.size()); i++)
            if (status[i] && !inBorder(cur_pts[i]))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        // ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        
        //printf("track cnt %d\n", (int)ids.size());
    }

    for (auto &n : track_cnt)
        n++;

    //rejectWithF();
    ROS_DEBUG("set mask begins");
    TicToc t_m;
    setMask();
    // ROS_DEBUG("set mask costs %fms", t_m.toc());
    // printf("set mask costs %fms\n", t_m.toc());
    ROS_DEBUG("detect feature begins");
    
    int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
    if(!USE_GPU)
    {
        if (n_max_cnt > MAX_CNT/4)
        {
            TicToc t_t;
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            cv::goodFeaturesToTrack(cur_img, n_pts, MAX_CNT - cur_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        addPoints();
    }
    
    // ROS_DEBUG("detect feature costs: %fms", t_t.toc());
    // printf("good feature to track costs: %fms\n", t_t.toc());
    else
    {
#ifdef USE_CUDA
        if (n_max_cnt > MAX_CNT/4)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
          
            cv::Ptr<cv::cuda::CornersDetector> detector = cv::cuda::createGoodFeaturesToTrackDetector(cur_gpu_img.type(), MAX_CNT - cur_pts.size(), 0.01, MIN_DIST);
            cv::cuda::GpuMat d_prevPts;
            cv::cuda::GpuMat gpu_mask(mask);
            detector->detect(cur_gpu_img, d_prevPts, gpu_mask);
            // std::cout << "d_prevPts size: "<< d_prevPts.size()<<std::endl;
            if(!d_prevPts.empty()) {
                n_pts = cv::Mat_<cv::Point2f>(cv::Mat(d_prevPts));
            }
            else {
                n_pts.clear();
            }

            // sum_n += n_pts.size();
            // printf("total point from gpu: %d\n",sum_n);
            // printf("gpu good feature to track cost: %fms\n", t_g.toc());
        }
        else 
            n_pts.clear();
#endif
        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        // ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
        // printf("selectFeature costs: %fms\n", t_a.toc());
    }

    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    if(!_img1.empty() && stereo_cam)
    {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        if(!cur_pts.empty())
        {
            //printf("stereo image; track feature on right image\n");
            
            vector<cv::Point2f> reverseLeftPts;
            vector<uchar> status, statusRightLeft;
            if(!USE_GPU)
            {
                TicToc t_check;
                vector<float> err;
                // cur left ---- cur right
                cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_pts, cur_right_pts, status, err, cv::Size(21, 21), 3);
                // reverse check cur right ---- cur left
                if(FLOW_BACK)
                {
                    cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_pts, reverseLeftPts, statusRightLeft, err, cv::Size(21, 21), 3);
                    for(size_t i = 0; i < status.size(); i++)
                    {
                        if(status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
                            status[i] = 1;
                        else
                            status[i] = 0;
                    }
                }
                // printf("left right optical flow cost %fms\n",t_check.toc());
            }
            else
            {
#ifdef USE_CUDA
                TicToc t_og1;
                cv::cuda::GpuMat cur_gpu_pts(cur_pts);
                cv::cuda::GpuMat cur_right_gpu_pts;
                cv::cuda::GpuMat gpu_status;
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                cv::Size(21, 21), 3, 30, false);
                d_pyrLK_sparse->calc(cur_gpu_img, right_gpu_img, cur_gpu_pts, cur_right_gpu_pts, gpu_status);

                vector<cv::Point2f> tmp_cur_right_pts(cur_right_gpu_pts.cols);
                cur_right_gpu_pts.download(tmp_cur_right_pts);
                cur_right_pts = tmp_cur_right_pts;

                vector<uchar> tmp_status(gpu_status.cols);
                gpu_status.download(tmp_status);
                status = tmp_status;

                if(FLOW_BACK)
                {   
                    cv::cuda::GpuMat reverseLeft_gpu_Pts;
                    cv::cuda::GpuMat status_gpu_RightLeft;
                    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                    cv::Size(21, 21), 3, 30, false);
                    d_pyrLK_sparse->calc(right_gpu_img, cur_gpu_img, cur_right_gpu_pts, reverseLeft_gpu_Pts, status_gpu_RightLeft);

                    vector<cv::Point2f> tmp_reverseLeft_Pts(reverseLeft_gpu_Pts.cols);
                    reverseLeft_gpu_Pts.download(tmp_reverseLeft_Pts);
                    reverseLeftPts = tmp_reverseLeft_Pts;

                    vector<uchar> tmp1_status(status_gpu_RightLeft.cols);
                    status_gpu_RightLeft.download(tmp1_status);
                    statusRightLeft = tmp1_status;
                    for(size_t i = 0; i < status.size(); i++)
                    {
                        if(status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
                            status[i] = 1;
                        else
                            status[i] = 0;
                    }
                }
                // printf("gpu left right optical flow cost %fms\n",t_og1.toc());
#endif
            }
            ids_right = ids;
            reduceVector(cur_right_pts, status);
            reduceVector(ids_right, status);
            // only keep left-right pts
            // reduceVector(cur_pts, status);
            // reduceVector(ids, status);
            // reduceVector(track_cnt, status);
            // reduceVector(cur_un_pts, status);
            // reduceVector(pts_velocity, status);
            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
            
        }
        prev_un_right_pts_map = cur_un_right_pts_map;
    }
    if(SHOW_TRACK)
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);

    prev_img = cur_img;
#ifdef USE_CUDA
    prev_gpu_img = cur_gpu_img;
#endif
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;
    hasPrediction = false;

    prevLeftPtsMap.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    FeatureFrame featureFrame;
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i];
        double x, y ,z;
        x = cur_un_pts[i].x;
        y = cur_un_pts[i].y;
        z = 1;

#ifdef UNIT_SPHERE_ERROR
        Eigen::Vector3d un_pt(x, y, z);
        un_pt.normalize();
        x = un_pt.x();
        y = un_pt.y();
        z = un_pt.z();
#endif

        double p_u, p_v;
        p_u = cur_pts[i].x;
        p_v = cur_pts[i].y;
        int camera_id = 0;
        double velocity_x, velocity_y;
        velocity_x = pts_velocity[i].x;
        velocity_y = pts_velocity[i].y;

        TrackFeatureNoId xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y, 0;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }

    if (!_img1.empty() && stereo_cam)
    {
        for (size_t i = 0; i < ids_right.size(); i++)
        {
            int feature_id = ids_right[i];
            double x, y ,z;
            x = cur_un_right_pts[i].x;
            y = cur_un_right_pts[i].y;
            z = 1;

#ifdef UNIT_SPHERE_ERROR
            Eigen::Vector3d un_pt(x, y, z);
            un_pt.normalize();
            x = un_pt.x();
            y = un_pt.y();
            z = un_pt.z();
#endif
            double p_u, p_v;
            p_u = cur_right_pts[i].x;
            p_v = cur_right_pts[i].y;
            int camera_id = 1;
            double velocity_x, velocity_y;
            velocity_x = right_pts_velocity[i].x;
            velocity_y = right_pts_velocity[i].y;

            TrackFeatureNoId xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y, 0;
            featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
        }
    }

    printf("feature track whole time %f PTS %ld\n", t_r.toc(), cur_un_pts.size());
    return featureFrame;
}

void FeatureTracker::rejectWithF()
{
    if (cur_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);

        if (FISHEYE) {
            ROS_INFO("Use as fisheye %s", calib_file[i].c_str());
            FisheyeUndist un(calib_file[i].c_str(), i, FISHEYE_FOV, true, COL);
            fisheys_undists.push_back(un);
        }

    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name.c_str(), undistortedImg);
    cv::waitKey(0);
}

vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        if(ENABLE_DOWNSAMPLE) {
            a = a*2;
        }
        cam->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}

vector<cv::Point3f> FeatureTracker::ptsVelocity3D(vector<int> &ids, vector<cv::Point3f> &cur_pts, 
                                            map<int, cv::Point3f> &cur_id_pts, map<int, cv::Point3f> &prev_id_pts)
{
    // ROS_INFO("Pts %ld Prev pts %ld IDS %ld", cur_pts.size(), prev_id_pts.size(), ids.size());
    vector<cv::Point3f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], cur_pts[i]));
    }

    // caculate points velocity
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;
        
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            std::map<int, cv::Point3f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end())
            {
                double v_x = (cur_pts[i].x - it->second.x) / dt;
                double v_y = (cur_pts[i].y - it->second.y) / dt;
                double v_z = (cur_pts[i].z - it->second.z) / dt;
                pts_velocity.push_back(cv::Point3f(v_x, v_y, v_z));
                // ROS_INFO("Dt %f, vel %f %f %f", v_x, v_y, v_z);

            }
            else
                pts_velocity.push_back(cv::Point3f(0, 0, 0));

        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point3f(0, 0, 0));
        }
    }
    return pts_velocity;
}

vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                            map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // caculate points velocity
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;
        
        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end())
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0));

        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}

void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts, 
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap)
{
    //int rows = imLeft.rows;
    int cols = imLeft.cols;
    if (!imRight.empty() && stereo_cam)
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();

    // cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);

    drawTrackImage(imTrack, curLeftPts, curLeftIds, prevLeftPtsMap);

    // for (size_t j = 0; j < curLeftPts.size(); j++)
    // {
    //     double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
    //     if(ENABLE_DOWNSAMPLE) {
    //         cv::circle(imTrack, curLeftPts[j]*2, 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    //     } else {
    //         cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    //     }
    // }
    if (!imRight.empty() && stereo_cam)
    {
        for (size_t i = 0; i < curRightPts.size(); i++)
        {
            if(ENABLE_DOWNSAMPLE) {
                cv::Point2f rightPt = curRightPts[i]*2;
                rightPt.x += cols;
                cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            } else {
                cv::Point2f rightPt = curRightPts[i];
                rightPt.x += cols;
                cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            }
            //cv::Point2f leftPt = curLeftPtsTrackRight[i];
            //cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }
    
    // map<int, cv::Point2f>::iterator mapIt;
    // for (size_t i = 0; i < curLeftIds.size(); i++)
    // {
    //     int id = curLeftIds[i];
    //     mapIt = prevLeftPtsMap.find(id);
    //     if(mapIt != prevLeftPtsMap.end())
    //     {
    //         if(ENABLE_DOWNSAMPLE) {
    //             cv::arrowedLine(imTrack, curLeftPts[i]*2, mapIt->second*2, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    //         } else {
    //             cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    //         }
    //     }
    // }



    cv::imshow("Track", imTrack);
    cv::waitKey(2);
}


void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    hasPrediction = true;
    predict_pts.clear();
    predict_pts_debug.clear();
    map<int, Eigen::Vector3d>::iterator itPredict;
    for (size_t i = 0; i < ids.size(); i++)
    {
        //printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids[i];
        itPredict = predictPts.find(id);
        if (itPredict != predictPts.end())
        {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
            predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
            predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        }
        else
            predict_pts.push_back(prev_pts[i]);
    }
}


void FeatureTracker::removeOutliers(set<int> &removePtsIds)
{
    std::set<int>::iterator itSet;
    vector<uchar> status;
    for (size_t i = 0; i < ids.size(); i++)
    {
        itSet = removePtsIds.find(ids[i]);
        if(itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
}


cv::Mat FeatureTracker::getTrackImage()
{
    return imTrack;
}
