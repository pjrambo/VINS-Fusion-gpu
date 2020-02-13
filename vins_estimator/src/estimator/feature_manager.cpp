/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "feature_manager.h"

// #define DEBUG_DISABLE_RETRIANGULATE
int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &_it : feature)
    {
        auto & it = _it.second;
        it.used_num = it.feature_per_frame.size();
        if (it.used_num >= 4 && it.good_for_solving)
        {
            cnt++;
        }
    }
    return cnt;
}


bool FeatureManager::addFeatureCheckParallax(int frame_count, const FeatureFrame &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    last_average_parallax = 0;
    new_feature_num = 0;
    long_track_num = 0;
    for (auto &id_pts : image)
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);
        //In common stereo; the pts in left must in right
        //But for stereo fisheye; this is not true due to the downward top view
        //We need to modified this to enable downward top view
        // assert(id_pts.second[0].first == 0);
        if (id_pts.second[0].first != 0) {
            //This point is right/down observation only
            f_per_fra.camera = 1;
        }
        
        if(id_pts.second.size() == 2)
        {
            // ROS_INFO("Stereo feature %d", id_pts.first);
            f_per_fra.rightObservation(id_pts.second[1].second);
            // assert(id_pts.second[1].first == 1);
            if (id_pts.second[1].first != 1) {
                ROS_WARN("Bug occurs on pt, skip");
                continue;
            }
        }

        int feature_id = id_pts.first;

        if (feature.find(feature_id) == feature.end()) {
            //Insert
            FeaturePerId fre(feature_id, frame_count);
            fre.main_cam = f_per_fra.camera;
            feature.emplace(feature_id, fre);
            feature[feature_id].feature_per_frame.push_back(f_per_fra);
            new_feature_num++;
        } else {
            feature[feature_id].feature_per_frame.push_back(f_per_fra);
            last_track_num++;
            if( feature[feature_id].feature_per_frame.size() >= 4)
                long_track_num++;
        }  
    }

    if (frame_count < 2 || last_track_num < 20 || long_track_num < KEYFRAME_LONGTRACK_THRES || new_feature_num > 0.5 * last_track_num) {
        ROS_INFO("Add KF LAST %d LONG %d new %d", last_track_num, long_track_num, new_feature_num);
        return true;
    }

    for (auto &_it : feature)
    {
        auto & it_per_id = _it.second;
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0)
    {
        ROS_INFO("Add KF: Parallax num ==0");
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        last_average_parallax = parallax_sum / parallax_num * FOCAL_LENGTH;
        bool is_kf = parallax_sum / parallax_num >= MIN_PARALLAX;
        if (is_kf) {
            ROS_INFO("Add KF: Parallax is bigger than required == 0");
        } 
        return is_kf;
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &_it : feature)
    {
        auto & it = _it.second;
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(std::map<int, double> deps)
{
    for (auto &it : deps)
    {
        int _id = it.first;
        double depth = it.second;
        auto & it_per_id = feature[_id];

        it_per_id.used_num = it_per_id.feature_per_frame.size();

        it_per_id.estimated_depth = 1.0 / depth;
        it_per_id.need_triangulation = false;
        it_per_id.depth_inited = true;
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
            it_per_id.good_for_solving = false;
            it_per_id.depth_inited = false;
        }
        else {
            it_per_id.solve_flag = 1;
        }
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        auto & _it = it->second;
        it_next++;
        if (_it.solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth()
{
    for (auto &_it : feature) {
        auto & it_per_id = _it.second;
        it_per_id.estimated_depth = -1;
        it_per_id.depth_inited = false;
        it_per_id.good_for_solving = false;
    }
}

std::map<int, double> FeatureManager::getDepthVector()
{
    //This function gives actually points for solving; We only use oldest max_solve_cnt point, oldest pts has good track
    //As for some feature point not solve all the time; we do re triangulate on it
    std::map<int, double> dep_vec;
    for (auto &_it : feature) {
        auto & it_per_id = _it.second;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        bool id_in_outouliers = outlier_features.find(it_per_id.feature_id) != outlier_features.end();
        if(dep_vec.size() < MAX_SOLVE_CNT && it_per_id.used_num >= 4 
            && it_per_id.good_for_solving && !id_in_outouliers) {
            dep_vec[it_per_id.feature_id] = 1. / it_per_id.estimated_depth;
            ft->setFeatureStatus(it_per_id.feature_id, 3);
        } else {
            //Clear depth; wait for re triangulate
#ifndef DEBUG_DISABLE_RETRIANGULATE
            it_per_id.need_triangulation = true;
#endif
        }
    }
    return dep_vec;
}


void FeatureManager::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point =
              design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


void FeatureManager::triangulatePoint3DPts(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector3d &point0, Eigen::Vector3d &point1, Eigen::Vector3d &point_3d)
{
    //TODO:Rewrite this for 3d point
    
    double p0x = point0[0];
    double p0y = point0[1];
    double p0z = point0[2];

    double p1x = point1[0];
    double p1y = point1[1];
    double p1z = point1[2];

    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = p0x * Pose0.row(2) - p0z*Pose0.row(0);
    design_matrix.row(1) = p0y * Pose0.row(2) - p0z*Pose0.row(1);
    design_matrix.row(2) = p1x * Pose1.row(2) - p1z*Pose1.row(0);
    design_matrix.row(3) = p1y * Pose1.row(2) - p1z*Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point =
              design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

double FeatureManager::triangulatePoint3DPts(vector<Eigen::Matrix<double, 3, 4>> &poses, vector<Eigen::Vector3d> &points, Eigen::Vector3d &point_3d)
{
    //TODO:Rewrite this for 3d point
    Eigen::MatrixXd design_matrix(poses.size()*2, 4);
    assert(poses.size() > 0 && poses.size() == points.size() && "We at least have 2 poses and number of pts and poses must equal");
    for (unsigned int i = 0; i < poses.size(); i ++) {
        double p0x = points[i][0];
        double p0y = points[i][1];
        double p0z = points[i][2];
        design_matrix.row(i*2) = p0x * poses[i].row(2) - p0z*poses[i].row(0);
        design_matrix.row(i*2+1) = p0y * poses[i].row(2) - p0z*poses[i].row(1);

    }
    Eigen::Vector4d triangulated_point;
    triangulated_point =
              design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);

    Eigen::MatrixXd pts(4, 1);
    pts << point_3d.x(), point_3d.y(), point_3d.z(), 1;
    Eigen::MatrixXd errs = design_matrix*pts;
    // std::cout << "ERR" << errs.sum() << std::endl;
    return errs.norm()/ errs.rows(); 
}



bool FeatureManager::solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P, 
                                      vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D)
{
    Eigen::Matrix3d R_initial;
    Eigen::Vector3d P_initial;

    // w_T_cam ---> cam_T_w 
    R_initial = R.inverse();
    P_initial = -(R_initial * P);

    //printf("pnp size %d \n",(int)pts2D.size() );
    if (int(pts2D.size()) < 4)
    {
        printf("feature tracking not enough, please slowly move you device! \n");
        return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);  
    bool pnp_succ;
    pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, 1);
    //pnp_succ = solvePnPRansac(pts3D, pts2D, K, D, rvec, t, true, 100, 8.0 / focalLength, 0.99, inliers);

    if(!pnp_succ)
    {
        printf("pnp failed ! \n");
        return false;
    }
    cv::Rodrigues(rvec, r);
    //cout << "r " << endl << r << endl;
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);

    // cam_T_w ---> w_T_cam
    R = R_pnp.transpose();
    P = R * (-T_pnp);

    return true;
}

void FeatureManager::initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{

    if(frameCnt > 0)
    {
        vector<cv::Point2f> pts2D;
        vector<cv::Point3f> pts3D;
        for (auto &_it : feature) {
            auto & it_per_id = _it.second;
            if (it_per_id.depth_inited && it_per_id.good_for_solving && it_per_id.main_cam == 0 && it_per_id.used_num >= 4)
            {
                int index = frameCnt - it_per_id.start_frame;
                if((int)it_per_id.feature_per_frame.size() >= index + 1)
                {
                    Vector3d ptsInCam = ric[0] * (it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth) + tic[0];
                    Vector3d ptsInWorld = Rs[it_per_id.start_frame] * ptsInCam + Ps[it_per_id.start_frame];

                    cv::Point3f point3d(ptsInWorld.x(), ptsInWorld.y(), ptsInWorld.z());
                    //Because PnP require 2d points; We hack here to use only z > 1 unit sphere point to init pnp
                    Eigen::Vector3d pt = it_per_id.feature_per_frame[index].point;
                    if (pt.z() > 0.1) {
                        cv::Point2f point2d(pt.x()/pt.z(), pt.y()/pt.z());
                        pts3D.push_back(point3d);
                        pts2D.push_back(point2d);                         
                    }
                }
            }
        }
        Eigen::Matrix3d RCam;
        Eigen::Vector3d PCam;
        // trans to w_T_cam
        RCam = Rs[frameCnt - 1] * ric[0];
        PCam = Rs[frameCnt - 1] * tic[0] + Ps[frameCnt - 1];

        if(solvePoseByPnP(RCam, PCam, pts2D, pts3D))
        {
            // trans to w_T_imu
            Rs[frameCnt] = RCam * ric[0].transpose(); 
            Ps[frameCnt] = -RCam * ric[0].transpose() * tic[0] + PCam;

            Eigen::Quaterniond Q(Rs[frameCnt]);
            //cout << "frameCnt: " << frameCnt <<  " pnp Q " << Q.w() << " " << Q.vec().transpose() << endl;
            // cout << "frameCnt: " << frameCnt << " pnp P " << Ps[frameCnt].transpose() << endl;
        }
    }
}

void FeatureManager::triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &_it : feature) {
        auto & it_per_id = _it.second;
        //Only solving point dnot re-triangulate
        if (!it_per_id.need_triangulation) {
            continue;
        }
        if (outlier_features.find(it_per_id.feature_id) != outlier_features.end()) {
            //Is in outliers
            ft->setFeatureStatus(it_per_id.feature_id, -1);
            continue;
        }

        std::vector<Eigen::Matrix<double, 3, 4>> poses;
        std::vector<Eigen::Vector3d> ptss;
        Eigen::Matrix<double, 3, 4> origin_pose;
        auto t0 = Ps[it_per_id.start_frame] + Rs[it_per_id.start_frame] * tic[0];
        auto R0 = Rs[it_per_id.start_frame] * ric[0];
        origin_pose.leftCols<3>() = R0.transpose();
        origin_pose.rightCols<1>() = -R0.transpose() * t0;
        bool has_stereo = false;

        Eigen::Vector3d _min = t0;
        Eigen::Vector3d _max = t0;

        for (unsigned int frame = 0; frame < it_per_id.feature_per_frame.size(); frame ++) {
            int imu_i = it_per_id.start_frame + frame;
            Eigen::Matrix<double, 3, 4> leftPose;
            Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
            _max.x() = max(t0.x(), _max.x());
            _max.y() = max(t0.y(), _max.y());
            _max.z() = max(t0.z(), _max.z());
            
            _min.x() = min(t0.x(), _min.x());
            _min.y() = min(t0.y(), _min.y());
            _min.z() = min(t0.z(), _min.z());

            Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
            leftPose.leftCols<3>() = R0.transpose();
            leftPose.rightCols<1>() = -R0.transpose() * t0;
            Eigen::Vector3d point0 = it_per_id.feature_per_frame[frame].point;

            poses.push_back(leftPose);
            ptss.push_back(point0);

            if(STEREO && it_per_id.feature_per_frame[frame].is_stereo) {
                has_stereo = true;
                Eigen::Matrix<double, 3, 4> rightPose;
                Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[1];
                Eigen::Matrix3d R1 = Rs[imu_i] * ric[1];
                rightPose.leftCols<3>() = R1.transpose();
                rightPose.rightCols<1>() = -R1.transpose() * t1;
                auto point1 = it_per_id.feature_per_frame[frame].pointRight;
                poses.push_back(rightPose);
                ptss.push_back(point1);

                _max.x() = max(t1.x(), _max.x());
                _max.y() = max(t1.y(), _max.y());
                _max.z() = max(t1.z(), _max.z());
                
                _min.x() = min(t1.x(), _min.x());
                _min.y() = min(t1.y(), _min.y());
                _min.z() = min(t1.z(), _min.z());
            }
        }

        if (poses.size() < 2) {
            //No enough information
            continue;
        }

        if (!has_stereo) {
            //We need calculate baseline
            if ((_max - _min).norm() < depth_estimate_baseline) {
                //Baseline too small; skipping
                continue;
            }
        }

        Eigen::Vector3d point3d;
        double err = triangulatePoint3DPts(poses, ptss, point3d)*FOCAL_LENGTH;
        if (err > triangulate_max_err) {
            // ROS_WARN("Feature ID %d CAM %d IS stereo %d poses %ld dep %d %f AVG ERR: %f", 
            //     it_per_id.feature_id, 
            //     it_per_id.main_cam,
            //     it_per_id.feature_per_frame[0].is_stereo,
            //     poses.size(),
            //     it_per_id.depth_inited, it_per_id.estimated_depth, err);
            ft->setFeatureStatus(it_per_id.feature_id, 2);
            it_per_id.good_for_solving = false;
            it_per_id.depth_inited = false;
            it_per_id.need_triangulation = true;
        } else {
            Eigen::Vector3d localPoint;
            localPoint = origin_pose.leftCols<3>() * point3d + origin_pose.rightCols<1>();
            if (it_per_id.feature_per_frame.size() >= 4) {
                ft->setFeatureStatus(it_per_id.feature_id, 1);            
            }
            it_per_id.depth_inited = true;
            it_per_id.good_for_solving = true;
            it_per_id.estimated_depth = localPoint.norm();
        }
        // ROS_INFO("Pt3d %f %f %f LocalPt %f %f %f", point3d.x(), point3d.y(), point3d.z(), localPoint.x(), localPoint.y(), localPoint.z());
    }
}

void FeatureManager::removeOutlier(set<int> &outlierIndex)
{
    std::set<int>::iterator itSet;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        int index = it->first;
        itSet = outlierIndex.find(index);
        if(itSet != outlierIndex.end())
        {
            ft->setFeatureStatus(it->second.feature_id, -1);
            feature.erase(it);
            outlier_features.insert(it->second.feature_id);
            //printf("remove outlier %d \n", index);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto _it = feature.begin(), it_next = feature.begin();
         _it != feature.end(); _it = it_next)
    {
        auto & it = _it->second; 
        it_next++;

        if (it.start_frame != 0)
            it.start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it.feature_per_frame[0].point;  
            it.feature_per_frame.erase(it.feature_per_frame.begin());
            if (it.feature_per_frame.size() < 2)
            {
                feature.erase(_it);
                ft->setFeatureStatus(_it->second.feature_id, -1);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it.estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j.norm();

                it.depth_inited = true;
                //We retriangulate this point
#ifndef DEBUG_DISABLE_RETRIANGULATE
                it.need_triangulation = true;
#endif
                if (FISHEYE) {
                    it.estimated_depth = pts_j.norm();
                } else {
                    if (dep_j > 0)
                        it.estimated_depth = dep_j;
                    else
                        it.estimated_depth = INIT_DEPTH;
                }
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->second.start_frame != 0)
            it->second.start_frame--;
        else
        {
            it->second.feature_per_frame.erase(it->second.feature_per_frame.begin());
            if (it->second.feature_per_frame.size() == 0) {
                ft->setFeatureStatus(it->second.feature_id, -1);
                feature.erase(it);
            }
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->second.start_frame == frame_count)
        {
            it->second.start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->second.start_frame;
            if (it->second.endFrame() < frame_count - 1)
                continue;
            it->second.feature_per_frame.erase(it->second.feature_per_frame.begin() + j);
            if (it->second.feature_per_frame.size() == 0) {
                ft->setFeatureStatus(it->second.feature_id, -1);
                feature.erase(it);
            }
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    if (FISHEYE) {
        return (frame_i.point - frame_j.point).norm();
    } else {
        double ans = 0;
        Vector3d p_j = frame_j.point;

        double u_j = p_j(0);
        double v_j = p_j(1);

        Vector3d p_i = frame_i.point;

        //int r_i = frame_count - 2;
        //int r_j = frame_count - 1;
        //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
        double dep_i = p_i(2);
        double u_i = p_i(0) / dep_i;
        double v_i = p_i(1) / dep_i;
        double du = u_i - u_j, dv = v_i - v_j;

        ans = sqrt(du * du + dv * dv);

        return ans;
    }
    return -1;
}