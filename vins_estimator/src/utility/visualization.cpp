/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "visualization.h"
#include <vins/VIOKeyframe.h>
#include <sensor_msgs/PointCloud.h>
#include <vins/FlattenImages.h>
#include "cv_bridge/cv_bridge.h"

using namespace ros;
using namespace Eigen;
ros::Publisher pub_odometry, pub_latest_odometry;
ros::Publisher pub_path;
ros::Publisher pub_point_cloud, pub_margin_cloud;
ros::Publisher pub_key_poses;
ros::Publisher pub_camera_pose;
ros::Publisher pub_camera_pose_right;
ros::Publisher pub_rectify_pose_left;
ros::Publisher pub_rectify_pose_right;
ros::Publisher pub_camera_pose_visual;
nav_msgs::Path path;
ros::Publisher pub_flatten_images;
ros::Publisher pub_keyframe_pose;
ros::Publisher pub_keyframe_point;
ros::Publisher pub_extrinsic;
ros::Publisher pub_viokeyframe;
ros::Publisher pub_viononkeyframe;

CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
static double sum_of_path = 0;
static Vector3d last_path(0.0, 0.0, 0.0);

size_t pub_counter = 0;

void registerPub(ros::NodeHandle &n)
{
    pub_latest_odometry = n.advertise<nav_msgs::Odometry>("imu_propagate", 1000);
    pub_path = n.advertise<nav_msgs::Path>("path", 1000);
    pub_odometry = n.advertise<nav_msgs::Odometry>("odometry", 1000);
    pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("point_cloud", 1000);
    pub_margin_cloud = n.advertise<sensor_msgs::PointCloud>("margin_cloud", 1000);
    pub_key_poses = n.advertise<visualization_msgs::Marker>("key_poses", 1000);
    pub_camera_pose = n.advertise<nav_msgs::Odometry>("camera_pose", 1000);
    pub_camera_pose_right = n.advertise<nav_msgs::Odometry>("camera_pose_right", 1000);
    pub_rectify_pose_left = n.advertise<geometry_msgs::PoseStamped>("rectify_pose_left", 1000);
    pub_rectify_pose_right = n.advertise<geometry_msgs::PoseStamped>("rectify_pose_right", 1000);
    pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
    pub_keyframe_pose = n.advertise<nav_msgs::Odometry>("keyframe_pose", 1000);
    pub_keyframe_point = n.advertise<sensor_msgs::PointCloud>("keyframe_point", 1000);
    pub_extrinsic = n.advertise<nav_msgs::Odometry>("extrinsic", 1000);
    pub_viokeyframe = n.advertise<vins::VIOKeyframe>("viokeyframe", 1000);
    pub_viononkeyframe = n.advertise<vins::VIOKeyframe>("viononkeyframe", 1000);
    pub_flatten_images = n.advertise<vins::FlattenImages>("flatten_images", 1000);

    cameraposevisual.setScale(0.1);
    cameraposevisual.setLineWidth(0.01);
}


geometry_msgs::Pose pose_from_PQ(Eigen::Vector3d P, 
    const Eigen::Quaterniond & Q) {
    geometry_msgs::Pose pose;
    pose.position.x = P.x();
    pose.position.y = P.y();
    pose.position.z = P.z();
    pose.orientation.x = Q.x();
    pose.orientation.y = Q.y();
    pose.orientation.z = Q.z();
    pose.orientation.w = Q.w();
    return pose;
}


void pubFlattenImages(const Estimator &estimator, const std_msgs::Header &header, 
    const Eigen::Vector3d & P, const Eigen::Quaterniond & Q, 
    std::vector<cv::cuda::GpuMat> & up_images, std::vector<cv::cuda::GpuMat> & down_images) {
    vins::FlattenImages images;
    images.header = header;
    images.pose_drone.position.x = P.x();
    images.pose_drone.position.y = P.y();
    images.pose_drone.position.z = P.z();
    images.pose_drone.orientation.x = Q.x();
    images.pose_drone.orientation.y = Q.y();
    images.pose_drone.orientation.z = Q.z();
    images.pose_drone.orientation.w = Q.w();
    static Eigen::Quaterniond t_left = Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(1, 0, 0)));
    static Eigen::Quaterniond t_front = t_left * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
    static Eigen::Quaterniond t_down = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1, 0, 0)));

    images.extrinsic_up_cams.push_back(
        pose_from_PQ(estimator.tic[0], Eigen::Quaterniond(estimator.ric[0])*t_front)
    );

    images.extrinsic_down_cams.push_back(
        pose_from_PQ(estimator.tic[1], Eigen::Quaterniond(estimator.ric[1])*t_down*t_front)
    );

    cv::Mat up, down;
    up_images[2].download(up);
    down_images[2].download(down);
    cv_bridge::CvImage outImg;
    outImg.header = header;
    outImg.encoding = "bgr8";
    outImg.image = up;
    images.up_cams.push_back(*outImg.toImageMsg());

    outImg.image = down;
    images.down_cams.push_back(*outImg.toImageMsg());

    pub_flatten_images.publish(images);
}

void pubFlattenImages(const Estimator &estimator, const std_msgs::Header &header, 
    const Eigen::Vector3d & P, const Eigen::Quaterniond & Q, 
    std::vector<cv::Mat> & up_images, std::vector<cv::Mat> & down_images) {
    vins::FlattenImages images;
    images.header = header;
    images.pose_drone.position.x = P.x();
    images.pose_drone.position.y = P.y();
    images.pose_drone.position.z = P.z();
    images.pose_drone.orientation.x = Q.x();
    images.pose_drone.orientation.y = Q.y();
    images.pose_drone.orientation.z = Q.z();
    images.pose_drone.orientation.w = Q.w();
    static Eigen::Quaterniond t_left = Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(1, 0, 0)));
    static Eigen::Quaterniond t_front = t_left * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
    static Eigen::Quaterniond t_right = t_front * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
    static Eigen::Quaterniond t_down = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1, 0, 0)));

    Eigen::Quaterniond t_arra[3] = {t_left, t_front, t_right};
    static int count = 0;

    int pub_index = count++ % 3 + 1;
    images.extrinsic_up_cams.push_back(
        pose_from_PQ(estimator.tic[0], Eigen::Quaterniond(estimator.ric[0])*t_arra[pub_index-1])
    );

    images.extrinsic_down_cams.push_back(
        pose_from_PQ(estimator.tic[1], Eigen::Quaterniond(estimator.ric[1])*t_down*t_arra[pub_index-1])
    );

    cv::Mat &up = up_images[pub_index];
    cv::Mat &down = down_images[pub_index];
    cv_bridge::CvImage outImg;
    outImg.header = header;
    outImg.encoding = "bgr8";
    outImg.image = up;
    images.up_cams.push_back(*outImg.toImageMsg());

    outImg.image = down;
    images.down_cams.push_back(*outImg.toImageMsg());

    pub_flatten_images.publish(images);
}

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t)
{
    nav_msgs::Odometry odometry;
    odometry.header.stamp = ros::Time(t);
    odometry.header.frame_id = "world";
    odometry.child_frame_id = "odometry";
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = Q.x();
    odometry.pose.pose.orientation.y = Q.y();
    odometry.pose.pose.orientation.z = Q.z();
    odometry.pose.pose.orientation.w = Q.w();
    odometry.twist.twist.linear.x = V.x();
    odometry.twist.twist.linear.y = V.y();
    odometry.twist.twist.linear.z = V.z();
    pub_latest_odometry.publish(odometry);

}

void printStatistics(const Estimator &estimator, double t)
{
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    //printf("position: %f, %f, %f\r", estimator.Ps[WINDOW_SIZE].x(), estimator.Ps[WINDOW_SIZE].y(), estimator.Ps[WINDOW_SIZE].z());
    ROS_DEBUG_STREAM("position: " << estimator.Ps[WINDOW_SIZE].transpose());
    ROS_DEBUG_STREAM("orientation: " << estimator.Vs[WINDOW_SIZE].transpose());
    if (ESTIMATE_EXTRINSIC)
    {
        cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            //ROS_DEBUG("calibration result for camera %d", i);
            ROS_DEBUG_STREAM("extirnsic tic: " << estimator.tic[i].transpose());
            ROS_DEBUG_STREAM("extrinsic ric: " << Utility::R2ypr(estimator.ric[i]).transpose());

            Eigen::Matrix4d eigen_T = Eigen::Matrix4d::Identity();
            eigen_T.block<3, 3>(0, 0) = estimator.ric[i];
            eigen_T.block<3, 1>(0, 3) = estimator.tic[i];
            cv::Mat cv_T;
            cv::eigen2cv(eigen_T, cv_T);
            if(i == 0)
                fs << "body_T_cam0" << cv_T ;
            else
                fs << "body_T_cam1" << cv_T ;
        }
        fs.release();
    }

    static double sum_of_time = 0;
    static int sum_of_calculation = 0;
    sum_of_time += t;
    sum_of_calculation++;
    ROS_DEBUG("vo solver costs: %f ms", t);
    ROS_DEBUG("average of time %f ms", sum_of_time / sum_of_calculation);

    sum_of_path += (estimator.Ps[WINDOW_SIZE] - last_path).norm();
    last_path = estimator.Ps[WINDOW_SIZE];
    ROS_DEBUG("sum of path %f", sum_of_path);
    if (ESTIMATE_TD)
        ROS_INFO("td %f", estimator.td);
}

void pubOdometry(const Estimator &estimator, const std_msgs::Header &header)
{

    
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.child_frame_id = "odometry";
        Quaterniond tmp_Q;
        tmp_Q = Quaterniond(estimator.Rs[WINDOW_SIZE]);
        odometry.pose.pose.position.x = estimator.Ps[WINDOW_SIZE].x();
        odometry.pose.pose.position.y = estimator.Ps[WINDOW_SIZE].y();
        odometry.pose.pose.position.z = estimator.Ps[WINDOW_SIZE].z();
        odometry.pose.pose.orientation.x = tmp_Q.x();
        odometry.pose.pose.orientation.y = tmp_Q.y();
        odometry.pose.pose.orientation.z = tmp_Q.z();
        odometry.pose.pose.orientation.w = tmp_Q.w();
        odometry.twist.twist.linear.x = estimator.Vs[WINDOW_SIZE].x();
        odometry.twist.twist.linear.y = estimator.Vs[WINDOW_SIZE].y();
        odometry.twist.twist.linear.z = estimator.Vs[WINDOW_SIZE].z();
        pub_odometry.publish(odometry);

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose = odometry.pose.pose;
        path.header = header;
        path.header.frame_id = "world";
        path.poses.push_back(pose_stamped);
        pub_path.publish(path);

        // write result to file
        ofstream foutC(VINS_RESULT_PATH, ios::app);
        foutC.setf(ios::fixed, ios::floatfield);
        foutC.precision(0);
        foutC << header.stamp.toSec() * 1e9 << ",";
        foutC.precision(5);
        foutC << estimator.Ps[WINDOW_SIZE].x() << ","
              << estimator.Ps[WINDOW_SIZE].y() << ","
              << estimator.Ps[WINDOW_SIZE].z() << ","
              << tmp_Q.w() << ","
              << tmp_Q.x() << ","
              << tmp_Q.y() << ","
              << tmp_Q.z() << ","
              << estimator.Vs[WINDOW_SIZE].x() << ","
              << estimator.Vs[WINDOW_SIZE].y() << ","
              << estimator.Vs[WINDOW_SIZE].z() << "," << endl;
        foutC.close();
        Eigen::Vector3d tmp_T = estimator.Ps[WINDOW_SIZE];
        printf("time: %f, t: %f %f %f q: %f %f %f %f \n", header.stamp.toSec(), tmp_T.x(), tmp_T.y(), tmp_T.z(),
                                                          tmp_Q.w(), tmp_Q.x(), tmp_Q.y(), tmp_Q.z());

        vins::VIOKeyframe vkf;
        vkf.header = header;
        int i = WINDOW_SIZE;
        Vector3d P = estimator.Ps[i];
        Quaterniond R = Quaterniond(estimator.Rs[i]);
        Vector3d P_r = P + R * estimator.tic[0];
        Quaterniond R_r = Quaterniond(R * estimator.ric[0]);
        vkf.pose_cam.position.x = P_r.x();
        vkf.pose_cam.position.y = P_r.y();
        vkf.pose_cam.position.z = P_r.z();
        vkf.pose_cam.orientation.x = R_r.x();
        vkf.pose_cam.orientation.y = R_r.y();
        vkf.pose_cam.orientation.z = R_r.z();
        vkf.pose_cam.orientation.w = R_r.w();

        vkf.camera_extrisinc.position.x = estimator.tic[0].x();
        vkf.camera_extrisinc.position.y = estimator.tic[0].y();
        vkf.camera_extrisinc.position.z = estimator.tic[0].z();

        Quaterniond ric = Quaterniond(estimator.ric[0]);
        ric.normalize();

        vkf.camera_extrisinc.orientation.x = ric.x();
        vkf.camera_extrisinc.orientation.y = ric.y();
        vkf.camera_extrisinc.orientation.z = ric.z();
        vkf.camera_extrisinc.orientation.w = ric.w();

        vkf.pose_drone = odometry.pose.pose;
        
        vkf.header.stamp = odometry.header.stamp;

        for (auto &_it : estimator.f_manager.feature)
        {
            auto & it_per_id = _it.second;
            int frame_size = it_per_id.feature_per_frame.size();
            // ROS_INFO("START FRAME %d FRAME_SIZE %d WIN SIZE %d solve flag %d", it_per_id.start_frame, frame_size, WINDOW_SIZE, it_per_id.solve_flag);
            if(it_per_id.start_frame < WINDOW_SIZE && it_per_id.start_frame + frame_size >= WINDOW_SIZE&& it_per_id.solve_flag < 2)
            {
                geometry_msgs::Point32 fp2d_uv;
                geometry_msgs::Point32 fp2d_norm;
                int imu_j = frame_size - 1;

                fp2d_uv.x = it_per_id.feature_per_frame[imu_j].uv.x();
                fp2d_uv.y = it_per_id.feature_per_frame[imu_j].uv.y();
                fp2d_uv.z = 0;

                fp2d_norm.x = it_per_id.feature_per_frame[imu_j].point.x();
                fp2d_norm.y = it_per_id.feature_per_frame[imu_j].point.y();
                fp2d_norm.z = 0;

                vkf.feature_points_id.push_back(it_per_id.feature_id);
                vkf.feature_points_2d_uv.push_back(fp2d_uv);
                vkf.feature_points_2d_norm.push_back(fp2d_norm);

                Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                Vector3d w_pts_i = estimator.Rs[imu_j] * (estimator.ric[it_per_id.main_cam] * pts_i + estimator.tic[it_per_id.main_cam])
                                    + estimator.Ps[imu_j];

                geometry_msgs::Point32 p;
                p.x = w_pts_i(0);
                p.y = w_pts_i(1);
                p.z = w_pts_i(2);

                vkf.feature_points_3d.push_back(p);
                vkf.feature_points_flag.push_back(it_per_id.solve_flag);
            }

        }
        pub_viononkeyframe.publish(vkf);
    }
    
   
}

void pubKeyPoses(const Estimator &estimator, const std_msgs::Header &header)
{
    if (estimator.key_poses.size() == 0)
        return;
    visualization_msgs::Marker key_poses;
    key_poses.header = header;
    key_poses.header.frame_id = "world";
    key_poses.ns = "key_poses";
    key_poses.type = visualization_msgs::Marker::SPHERE_LIST;
    key_poses.action = visualization_msgs::Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    key_poses.lifetime = ros::Duration();

    //static int key_poses_id = 0;
    key_poses.id = 0; //key_poses_id++;
    key_poses.scale.x = 0.05;
    key_poses.scale.y = 0.05;
    key_poses.scale.z = 0.05;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        geometry_msgs::Point pose_marker;
        Vector3d correct_pose;
        correct_pose = estimator.key_poses[i];
        pose_marker.x = correct_pose.x();
        pose_marker.y = correct_pose.y();
        pose_marker.z = correct_pose.z();
        key_poses.points.push_back(pose_marker);
    }
    pub_key_poses.publish(key_poses);
}

void pubCameraPose(const Estimator &estimator, const std_msgs::Header &header)
{
    int idx2 = WINDOW_SIZE - 1;

    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        int i = idx2;
        Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[0]);

        nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();

        if(STEREO)
        {
            Vector3d P_r = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[1];
            Quaterniond R_r = Quaterniond(estimator.Rs[i] * estimator.ric[1]);

            nav_msgs::Odometry odometry_r;
            odometry_r.header = header;
            odometry_r.header.frame_id = "world";
            odometry_r.pose.pose.position.x = P_r.x();
            odometry_r.pose.pose.position.y = P_r.y();
            odometry_r.pose.pose.position.z = P_r.z();
            odometry_r.pose.pose.orientation.x = R_r.x();
            odometry_r.pose.pose.orientation.y = R_r.y();
            odometry_r.pose.pose.orientation.z = R_r.z();
            odometry_r.pose.pose.orientation.w = R_r.w();
            pub_camera_pose_right.publish(odometry_r);
            if(PUB_RECTIFY)
            {
                Vector3d R_P_l = P;
                Vector3d R_P_r = P_r;
                Quaterniond R_R_l = Quaterniond(estimator.Rs[i] * estimator.ric[0] * rectify_R_left.inverse());
                Quaterniond R_R_r = Quaterniond(estimator.Rs[i] * estimator.ric[1] * rectify_R_right.inverse());
                geometry_msgs::PoseStamped R_pose_l, R_pose_r;
                R_pose_l.header = header;
                R_pose_r.header = header;
                R_pose_l.header.frame_id = "world";
                R_pose_r.header.frame_id = "world";
                R_pose_l.pose.position.x = R_P_l.x();
                R_pose_l.pose.position.y = R_P_l.y();
                R_pose_l.pose.position.z = R_P_l.z();
                R_pose_l.pose.orientation.x = R_R_l.x();
                R_pose_l.pose.orientation.y = R_R_l.y();
                R_pose_l.pose.orientation.z = R_R_l.z();
                R_pose_l.pose.orientation.w = R_R_l.w();

                R_pose_r.pose.position.x = R_P_r.x();
                R_pose_r.pose.position.y = R_P_r.y();
                R_pose_r.pose.position.z = R_P_r.z();
                R_pose_r.pose.orientation.x = R_R_r.x();
                R_pose_r.pose.orientation.y = R_R_r.y();
                R_pose_r.pose.orientation.z = R_R_r.z();
                R_pose_r.pose.orientation.w = R_R_r.w();

                pub_rectify_pose_left.publish(R_pose_l);
                pub_rectify_pose_right.publish(R_pose_r);

            }
        }

        pub_camera_pose.publish(odometry);

        cameraposevisual.reset();
        cameraposevisual.add_pose(P, R);
        if(STEREO)
        {
            Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[1];
            Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[1]);
            cameraposevisual.add_pose(P, R);
        }
        cameraposevisual.publish_by(pub_camera_pose_visual, odometry.header);
    }
}


void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header)
{
    sensor_msgs::PointCloud point_cloud, loop_point_cloud;
    point_cloud.header = header;
    loop_point_cloud.header = header;


    for (auto _it : estimator.f_manager.feature)
    {
        auto it_per_id = _it.second;
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1)
            continue;
        int imu_i = it_per_id.start_frame;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[it_per_id.main_cam] * pts_i + estimator.tic[it_per_id.main_cam]) + estimator.Ps[imu_i];

        geometry_msgs::Point32 p;
        p.x = w_pts_i(0);
        p.y = w_pts_i(1);
        p.z = w_pts_i(2);
        point_cloud.points.push_back(p);
    }
    pub_point_cloud.publish(point_cloud);


    // pub margined potin
    sensor_msgs::PointCloud margin_cloud;
    margin_cloud.header = header;

    for (auto &_it : estimator.f_manager.feature)
    {
        auto & it_per_id = _it.second;
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        //if (it_per_id->start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id->solve_flag != 1)
        //        continue;

        if (it_per_id.start_frame == 0 && it_per_id.feature_per_frame.size() <= 2 
            && it_per_id.solve_flag == 1 )
        {
            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
            Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[it_per_id.main_cam] * pts_i + estimator.tic[it_per_id.main_cam]) + estimator.Ps[imu_i];

            geometry_msgs::Point32 p;
            p.x = w_pts_i(0);
            p.y = w_pts_i(1);
            p.z = w_pts_i(2);
            margin_cloud.points.push_back(p);
        }
    }
    pub_margin_cloud.publish(margin_cloud);
}


void pubTF(const Estimator &estimator, const std_msgs::Header &header)
{
    if( estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    // body frame
    Vector3d correct_t;
    Quaterniond correct_q;
    correct_t = estimator.Ps[WINDOW_SIZE];
    correct_q = estimator.Rs[WINDOW_SIZE];

    transform.setOrigin(tf::Vector3(correct_t(0),
                                    correct_t(1),
                                    correct_t(2)));
    q.setW(correct_q.w());
    q.setX(correct_q.x());
    q.setY(correct_q.y());
    q.setZ(correct_q.z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "world", "body"));

    // camera frame
    transform.setOrigin(tf::Vector3(estimator.tic[0].x(),
                                    estimator.tic[0].y(),
                                    estimator.tic[0].z()));
    q.setW(Quaterniond(estimator.ric[0]).w());
    q.setX(Quaterniond(estimator.ric[0]).x());
    q.setY(Quaterniond(estimator.ric[0]).y());
    q.setZ(Quaterniond(estimator.ric[0]).z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "body", "camera"));

    
    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = estimator.tic[0].x();
    odometry.pose.pose.position.y = estimator.tic[0].y();
    odometry.pose.pose.position.z = estimator.tic[0].z();
    Quaterniond tmp_q{estimator.ric[0]};
    odometry.pose.pose.orientation.x = tmp_q.x();
    odometry.pose.pose.orientation.y = tmp_q.y();
    odometry.pose.pose.orientation.z = tmp_q.z();
    odometry.pose.pose.orientation.w = tmp_q.w();
    pub_extrinsic.publish(odometry);

}

void pubKeyframe(const Estimator &estimator)
{
    // pub camera pose, 2D-3D points of keyframe
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR && estimator.marginalization_flag == 0)
    {
        vins::VIOKeyframe vkf;
        int i = WINDOW_SIZE - 2;
        //Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Vector3d P = estimator.Ps[i];
        Quaterniond R = Quaterniond(estimator.Rs[i]);

        nav_msgs::Odometry odometry;
        odometry.header.stamp = ros::Time(estimator.Headers[WINDOW_SIZE - 2]);
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();


        //This is pose of left camera!!!!
        Vector3d P_r = P + R * estimator.tic[0];
        Quaterniond R_r = Quaterniond(R * estimator.ric[0]);
        R_r.normalize();
        //printf("time: %f t: %f %f %f r: %f %f %f %f\n", odometry.header.stamp.toSec(), P.x(), P.y(), P.z(), R.w(), R.x(), R.y(), R.z());
        vkf.pose_cam.position.x = P_r.x();
        vkf.pose_cam.position.y = P_r.y();
        vkf.pose_cam.position.z = P_r.z();
        vkf.pose_cam.orientation.x = R_r.x();
        vkf.pose_cam.orientation.y = R_r.y();
        vkf.pose_cam.orientation.z = R_r.z();
        vkf.pose_cam.orientation.w = R_r.w();

        vkf.camera_extrisinc.position.x = estimator.tic[0].x();
        vkf.camera_extrisinc.position.y = estimator.tic[0].y();
        vkf.camera_extrisinc.position.z = estimator.tic[0].z();

        Quaterniond ric = Quaterniond(estimator.ric[0]);
        ric.normalize();

        vkf.camera_extrisinc.orientation.x = ric.x();
        vkf.camera_extrisinc.orientation.y = ric.y();
        vkf.camera_extrisinc.orientation.z = ric.z();
        vkf.camera_extrisinc.orientation.w = ric.w();

        vkf.pose_drone = odometry.pose.pose;
        
        vkf.header.stamp = odometry.header.stamp;


        pub_keyframe_pose.publish(odometry);


        sensor_msgs::PointCloud point_cloud;
        point_cloud.header.stamp = ros::Time(estimator.Headers[WINDOW_SIZE - 2]);
        point_cloud.header.frame_id = "world";
        for (auto &_it : estimator.f_manager.feature)
        {
            auto & it_per_id = _it.second;
            int frame_size = it_per_id.feature_per_frame.size();
            if(it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 && it_per_id.solve_flag < 2)
            {

                int imu_i = it_per_id.start_frame;
                Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[it_per_id.main_cam] * pts_i + estimator.tic[it_per_id.main_cam])
                                      + estimator.Ps[imu_i];
                geometry_msgs::Point32 p;
                p.x = w_pts_i(0);
                p.y = w_pts_i(1);
                p.z = w_pts_i(2);
                point_cloud.points.push_back(p);

                vkf.feature_points_3d.push_back(p);

                // int imu_j = frame_size - 2;
                int imu_j =  WINDOW_SIZE - 2 - it_per_id.start_frame;
                sensor_msgs::ChannelFloat32 p_2d;
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.y());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.y());
                p_2d.values.push_back(it_per_id.feature_id);
                point_cloud.channels.push_back(p_2d);

                geometry_msgs::Point32 fp2d_uv;
                geometry_msgs::Point32 fp2d_norm;
                fp2d_uv.x = it_per_id.feature_per_frame[imu_j].uv.x();
                fp2d_uv.y = it_per_id.feature_per_frame[imu_j].uv.y();
                fp2d_uv.z = 0;

                fp2d_norm.x = it_per_id.feature_per_frame[imu_j].point.x();
                fp2d_norm.y = it_per_id.feature_per_frame[imu_j].point.y();
                fp2d_norm.z = 0;

                vkf.feature_points_id.push_back(it_per_id.feature_id);
                vkf.feature_points_2d_uv.push_back(fp2d_uv);
                vkf.feature_points_2d_norm.push_back(fp2d_norm);
                vkf.feature_points_flag.push_back(it_per_id.solve_flag);
            }

        }
        pub_keyframe_point.publish(point_cloud);
        pub_viokeyframe.publish(vkf);
    }
}