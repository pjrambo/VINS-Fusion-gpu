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

#include "feature_tracker.h"

#define BACKWARD_HAS_DW 1
#include <backward.hpp>

namespace backward
{
    backward::SignalHandling sh;
}

bool FeatureTracker::inBorder(const cv::Point2f &pt, cv::Size shape)
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

double distance(cv::Point2f pt1, cv::Point2f pt2)
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
        if (mask.at<uchar>(it.second.first) == 255)
        {
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }

    return mask;
}


void FeatureTracker::setMaskFisheye() {
    //TODO:Set mask for fisheye
    mask_up_top = setMaskFisheye(top_size, cur_up_top_pts, track_up_top_cnt, ids_up_top);
    mask_down_top = setMaskFisheye(top_size, cur_down_top_pts, track_down_top_cnt, ids_down_top);
    mask_up_side = setMaskFisheye(side_size, cur_up_side_pts, track_up_side_cnt, ids_up_side);
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
        if (mask.at<uchar>(it.second.first) == 255)
        {
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
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

double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

void FeatureTracker::drawTrackImage(cv::Mat & img, vector<cv::Point2f> pts, vector<int> ids, vector<int> track_cnt, map<int, cv::Point2f> prev_pts) {
    char idtext[10] = {0};
    for (int j = 0; j < pts.size(); j++) {
        double len = 0;
        if (track_cnt.size() > 0) {
            len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        }
        cv::circle(img, pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
        sprintf(idtext, "%d", ids[j]);
	    cv::putText(img, idtext, pts[j] - cv::Point2f(5, 0), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(252, 255, 240), 3);

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

void FeatureTracker::drawTrackFisheye(const cv::Mat & img_up,
    const cv::Mat & img_down,
    cv::cuda::GpuMat & imUpTop,
    cv::cuda::GpuMat &imDownTop,
    cv::cuda::GpuMat &imUpSide_cuda, 
    cv::cuda::GpuMat &imDownSide_cuda)
{
    // ROS_INFO("Up image %d, down %d", imUp.size(), imDown.size());
    cv::Mat up_camera;
    cv::Mat down_camera;
    cv::Mat imTrack;
    cv::Mat imUpSide;
    cv::Mat imDownSide;
    cv::Mat fisheye_up;
    cv::Mat fisheye_down;
    
    int side_height = imUpSide_cuda.size().height;
    int width = imUpTop.size().width;

    //128
    fisheye_up = img_up(cv::Rect(190, 62, 900, 900));
    fisheye_down = img_down(cv::Rect(190, 62, 900, 900));

    cv::resize(fisheye_up, fisheye_up, cv::Size(width, width));
    cv::resize(fisheye_down, fisheye_down, cv::Size(width, width));
    
    cv::cvtColor(fisheye_up,   fisheye_up,   cv::COLOR_GRAY2BGR);
    cv::cvtColor(fisheye_down, fisheye_down, cv::COLOR_GRAY2BGR);

    imUpTop.download(up_camera);
    cv::cvtColor(up_camera, up_camera, cv::COLOR_GRAY2BGR);
    imDownTop.download(down_camera);
    cv::cvtColor(down_camera, down_camera, cv::COLOR_GRAY2BGR);


    imUpSide_cuda.download(imUpSide);
    cv::cvtColor(imUpSide, imUpSide, cv::COLOR_GRAY2BGR);

    imDownSide_cuda.download(imDownSide);
    cv::cvtColor(imDownSide, imDownSide, cv::COLOR_GRAY2BGR);

    drawTrackImage(up_camera, cur_up_top_pts, ids_up_top, track_up_top_cnt, up_top_prevLeftPtsMap);
    drawTrackImage(down_camera, cur_down_top_pts, ids_down_top, track_down_top_cnt, down_top_prevLeftPtsMap);
    drawTrackImage(imUpSide, cur_up_side_pts, ids_up_side, track_up_side_cnt, up_side_prevLeftPtsMap);
    drawTrackImage(imDownSide, cur_down_side_pts, ids_down_side, vector<int>(), down_side_prevLeftPtsMap);

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
    cv::hconcat(up_camera, down_camera, top_cam);
    cv::hconcat(fisheye_up, top_cam, top_cam);
    cv::hconcat(top_cam, fisheye_down, top_cam); 
    // ROS_INFO("Imtrack width %d", imUpSide.size().width);
    cv::resize(top_cam, top_cam, cv::Size(imUpSide.size().width, imUpSide.size().width/4));
    
    cv::vconcat(top_cam, imTrack, imTrack);
    
    // cv::resize(up_camera, up_camera, cv::Size(side_height, side_height));
    // cv::resize(down_camera, down_camera, cv::Size(side_height, side_height));

    // cv::hconcat(up_camera, imUpSide, up_camera);
    // cv::hconcat(down_camera, imDownSide, down_camera);

    // cv::hconcat(up_camera, down_camera, imTrack);
    double fx = ((double)SHOW_WIDTH) / ((double) imUpSide.size().width);
    cv::resize(imTrack, imTrack, cv::Size(), fx, fx);
    cv::imshow("tracking", imTrack);
    // cv::imshow("tracking_top", top_cam);
    cv::waitKey(10);
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


void FeatureTracker::detectPoints(const cv::cuda::GpuMat & img, const cv::Mat & mask, 
    vector<cv::Point2f> & n_pts, vector<cv::Point2f> & cur_pts, int require_pts) {
    int lack_up_top_pts = require_pts - static_cast<int>(cur_pts.size());

    //Add Points Top
    ROS_DEBUG("Lack %d pts; Require %d will detect %d", lack_up_top_pts, require_pts, lack_up_top_pts > require_pts/4);
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

    ROS_DEBUG("Detected %ld npts", n_pts.size());
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
    Eigen::Quaterniond t1(Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(1, 0, 0)));
    Eigen::Quaterniond t2 = t1 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
    Eigen::Quaterniond t3 = t2 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
    Eigen::Quaterniond t4 = t3 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
    Eigen::Quaterniond t_down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1, 0, 0)));

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

vector<cv::Point2f> FeatureTracker::opticalflow_track(cv::cuda::GpuMat & cur_img, 
                        cv::cuda::GpuMat & prev_img, vector<cv::Point2f> & prev_pts, 
                        vector<int> & ids, vector<int> & track_cnt,
                        bool flow_back, vector<cv::Point2f> prediction_points){
    if (prev_pts.size() == 0) {
        return vector<cv::Point2f>();
    }
    vector<cv::Point2f> cur_pts;
    TicToc t_og;
    cv::cuda::GpuMat prev_gpu_pts(prev_pts);
    cv::cuda::GpuMat cur_gpu_pts(cur_pts);
    cv::cuda::GpuMat gpu_status;
    vector<uchar> status;

    //Assume No Prediction Need to add later
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
        cv::Size(21, 21), 3, 30, false);
    d_pyrLK_sparse->calc(prev_img, cur_img, prev_gpu_pts, cur_gpu_pts, gpu_status);

    cur_gpu_pts.download(cur_pts);

    gpu_status.download(status);
    if(FLOW_BACK)
    {
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

    // ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    
    //printf("track cnt %d\n", (int)ids.size());

    for (auto &n : track_cnt)
        n++;

    return cur_pts;
}


map<int, cv::Point2f> pts_map(vector<int> ids, vector<cv::Point2f> cur_pts) {
    map<int, cv::Point2f> prevMap;
    for (unsigned int i = 0; i < ids.size(); i ++) {
        prevMap[ids[i]] = cur_pts[i];
    }
    return prevMap;
}

FeatureFrame FeatureTracker::trackImage_fisheye(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1) {
    TicToc t_r;
    cur_time = _cur_time;

    auto fisheye_imgs_up = fisheys_undists[0].undist_all_cuda(_img);
    auto fisheye_imgs_down = fisheys_undists[1].undist_all_cuda(_img1);

    auto up_side_img = concat_side(fisheye_imgs_up);
    auto down_side_img = concat_side(fisheye_imgs_down);
    auto & up_top_img = fisheye_imgs_up[0];
    auto & down_top_img = fisheye_imgs_down[0];

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

    //If has predict;
    if (enable_up_top) {
        cur_up_top_pts = opticalflow_track(up_top_img, prev_up_top_img, prev_up_top_pts, ids_up_top, track_up_top_cnt, FLOW_BACK);
    }
    if (enable_up_side) {
        cur_up_side_pts = opticalflow_track(up_side_img, prev_up_side_img, prev_up_side_pts, ids_up_side, track_up_side_cnt, FLOW_BACK);
    }

    if (enable_down_top) {
        cur_down_top_pts = opticalflow_track(down_top_img, prev_down_top_img, prev_down_top_pts, ids_down_top, track_down_top_cnt, FLOW_BACK);
    }
    

    setMaskFisheye();
    if (enable_up_top) {
        detectPoints(up_top_img, mask_up_top, n_pts_up_top, cur_up_top_pts, TOP_PTS_CNT);
    }
    if (enable_down_top) {
        detectPoints(down_top_img, mask_down_top, n_pts_down_top, cur_down_top_pts, TOP_PTS_CNT);
    }

    if (enable_up_side) {
        detectPoints(up_side_img, mask_up_side, n_pts_up_side, cur_up_side_pts, SIDE_PTS_CNT);
    }

    addPointsFisheye();


    if (enable_down_side) {
        ids_down_side = ids_up_side;
        auto down_side_init_pts = cur_up_side_pts;
        cur_down_side_pts = opticalflow_track(down_side_img, up_side_img, down_side_init_pts, ids_down_side, track_down_side_cnt, FLOW_BACK);
        // ROS_INFO("Down side try to track %ld pts; gives %ld", cur_up_side_pts.size(), cur_down_side_pts.size());
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
    if (SHOW_TRACK) {
        drawTrackFisheye(_img, _img1,up_top_img, down_top_img, up_side_img, down_side_img);
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

    printf("feature track whole time %f PTS %ld\n", t_r.toc(), cur_un_pts.size());
    return ff;
}

FeatureFrame FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1)
{
    TicToc t_r;
    cur_time = _cur_time;
    cv::Mat rightImg;
    cv::cuda::GpuMat right_gpu_img;

    if (USE_GPU_ACC_FLOW) {
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
        if(!USE_GPU_ACC_FLOW)
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
            if(!USE_GPU_ACC_FLOW)
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
    prev_gpu_img = cur_gpu_img;
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

// #ifdef UNIT_SPHERE_ERROR
//         Eigen::Vector3d un_pt(x, y, z);
//         un_pt.normalize();
//         x = un_pt.x();
//         y = un_pt.y();
//         z = un_pt.z();
// #endif

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

// #ifdef UNIT_SPHERE_ERROR
//         Eigen::Vector3d un_pt(x, y, z);
//         un_pt.normalize();
//         x = un_pt.x();
//         y = un_pt.y();
//         z = un_pt.z();
// #endif
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
    cv::imshow(name, undistortedImg);
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
    cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        if(ENABLE_DOWNSAMPLE) {
            cv::circle(imTrack, curLeftPts[j]*2, 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
        } else {
            cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
        }
    }
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
    
    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if(mapIt != prevLeftPtsMap.end())
        {
            if(ENABLE_DOWNSAMPLE) {
                cv::arrowedLine(imTrack, curLeftPts[i]*2, mapIt->second*2, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
            } else {
                cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
            }
        }
    }

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