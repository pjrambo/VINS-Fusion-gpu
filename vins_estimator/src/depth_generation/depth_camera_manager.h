#pragma once

#include <sensor_msgs/PointCloud.h>
#include <ros/ros.h>
#include <eigen3/Eigen/Eigen>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <tf/transform_broadcaster.h>
#include "depth_estimator.h"
#include "color_disparity_graph.hpp"
#include <camodocal/camera_models/CameraFactory.h>
#include <camodocal/camera_models/PinholeCamera.h>

class FisheyeUndist;

class DepthCamManager {
    std::vector<ros::Publisher> pub_depth_clouds;
    std::vector<ros::Publisher> pub_depth_maps;
    std::vector<ros::Publisher> pub_depthcam_poses;

    ros::Publisher pub_depth_cloud;
    ros::Publisher up_cam_info_pub, down_cam_info_pub;
    ros::Publisher pub_camera_up, pub_camera_down;
    ros::NodeHandle nh;
    tf::TransformBroadcaster br;
    bool publish_raw_image = false;

    bool estimate_front_depth = true;
    bool estimate_left_depth = false;
    bool estimate_right_depth = false;
    bool estimate_rear_depth = false;
    bool pub_cloud_all = false;
    bool pub_cloud_per_direction = false;
    bool pub_depth_map = false;
    
    SGMParams sgm_params;
    
    double downsample_ratio = 1.0;
    Eigen::Matrix3d cam_side;
    Eigen::Matrix3d cam_side_transpose;
    cv::Mat cam_side_cv, cam_side_cv_transpose;

    int pub_cloud_step = 1;

    std::vector<DepthEstimator *> deps;
    std::vector<cv::Mat> depth_maps;
    std::vector<cv::Mat> pts_3ds;
    std::vector<cv::Mat> texture_imgs;

    int show_disparity = 0;
    int enable_extrinsic_calib_for_depth = 0;
    double depth_cloud_radius = 5;
    camodocal::CameraPtr depth_cam;

    std::vector<std::string> dep_RT_config;

public:

    void init_with_extrinsic(Eigen::Matrix3d ric1, Eigen::Vector3d tic1, 
        Eigen::Matrix3d ric2, Eigen::Vector3d tic2);
    FisheyeUndist * fisheye = nullptr;

    Eigen::Quaterniond t_left, t_front, t_right, t_rear, t_down, t_rotate;
    double f_side, cx_side, cy_side;

    DepthCamManager(ros::NodeHandle & _nh, FisheyeUndist * _fisheye);

    void update_depth_image(int direction, cv::Mat _up_front, cv::Mat _down_front, 
        Eigen::Matrix3d ric1, Eigen::Matrix3d ric_dept);

    void update_depth_image(int direction, cv::cuda::GpuMat _up_front, cv::cuda::GpuMat _down_front, 
        Eigen::Matrix3d ric1, Eigen::Matrix3d ric_depth);

    void update_pcl_depth_from_image(ros::Time stamp, int direction, Eigen::Matrix3d ric1, Eigen::Vector3d tic1, 
        Eigen::Matrix3d R, Eigen::Vector3d P, Eigen::Matrix3d ric_depth, sensor_msgs::PointCloud & pcl);

    void pub_depths_from_buf(ros::Time stamp, Eigen::Matrix3d ric1, Eigen::Vector3d tic1, 
        Eigen::Matrix3d R, Eigen::Vector3d P);

    DepthEstimator * create_depth_estimator(int direction, Eigen::Matrix3d r01, Eigen::Vector3d t01);
    
    void publish_world_point_cloud(cv::Mat pts3d, Eigen::Matrix3d R, Eigen::Vector3d P, ros::Time stamp,
        int dir, int step = 3, cv::Mat color = cv::Mat());
    
    void add_pts_point_cloud(cv::Mat pts3d, Eigen::Matrix3d R, Eigen::Vector3d P, ros::Time stamp,
        sensor_msgs::PointCloud & pcl, int step = 3, cv::Mat color = cv::Mat());

    cv::Mat generate_depthmap(cv::Mat pts3d, Eigen::Matrix3d rel_ric_depth) const;

    template<typename cvMat>
    void update_depth_image(ros::Time stamp, cvMat _up_front, cvMat _down_front, 
        Eigen::Matrix3d ric1, Eigen::Vector3d tic1,
        Eigen::Matrix3d R, Eigen::Vector3d P, int direction, sensor_msgs::PointCloud & pcl, Eigen::Matrix3d ric_depth) {

        this->update_depth_image(direction, _up_front, _down_front, ric1, ric_depth);
        update_pcl_depth_from_image(stamp, direction, ric1, tic1, R, P, ric_depth, pcl);
        
    }

    template<typename cvMat>
    void update_images(ros::Time stamp, std::vector<cvMat> & up_cams, std::vector<cvMat> & down_cams,
            Eigen::Matrix3d ric1, Eigen::Vector3d tic1,
            Eigen::Matrix3d ric2, Eigen::Vector3d tic2, 
            Eigen::Matrix3d R, Eigen::Vector3d P
        ) {
        
        sensor_msgs::PointCloud point_cloud;
        point_cloud.header.stamp = stamp;
        point_cloud.header.frame_id = "world";
        point_cloud.channels.resize(3);
        point_cloud.channels[0].name = "rgb";
        point_cloud.channels[0].values.resize(0);
        point_cloud.channels[1].name = "u";
        point_cloud.channels[1].values.resize(0);
        point_cloud.channels[2].name = "v";
        point_cloud.channels[2].values.resize(0);

        if (estimate_front_depth) {
            update_depth_image(stamp, up_cams[2], down_cams[2], ric1*t_front*t_rotate, 
                tic1, R, P, 1, point_cloud, ric1*t_front);
        }

        if (estimate_left_depth) {
            update_depth_image(stamp, up_cams[1], down_cams[1], ric1*t_left*t_rotate, 
                tic1, R, P, 0, point_cloud, ric1*t_left);
        }

        if (estimate_right_depth) {
            update_depth_image(stamp, up_cams[3], down_cams[3], ric1*t_right*t_rotate, 
                tic1, R, P, 2, point_cloud, ric1*t_right);
        }

        if (estimate_rear_depth) {
            update_depth_image(stamp, up_cams[4], down_cams[4], ric1*t_rear*t_rotate, 
                tic1, R, P, 3, point_cloud, ric1*t_rear);
        }

        if (pub_cloud_all) {
            pub_depth_cloud.publish(point_cloud);
        }

    }



    template<typename cvMat>
    void update_images_to_buf(std::vector<cvMat> & up_cams, std::vector<cvMat> & down_cams) {
        
        if (estimate_front_depth) {
            update_depth_image(1, up_cams[2], down_cams[2], 
                (t_front*t_rotate).toRotationMatrix(), t_front.toRotationMatrix());
        }

        if (estimate_left_depth) {
            update_depth_image(0, up_cams[1], down_cams[1], 
                (t_left*t_rotate).toRotationMatrix(), t_left.toRotationMatrix());
        }

        if (estimate_right_depth) {
            update_depth_image(2, up_cams[3], down_cams[3], 
                (t_right*t_rotate).toRotationMatrix(), t_right.toRotationMatrix());
        }

        if (estimate_rear_depth) {
            update_depth_image(3, up_cams[4], down_cams[4], 
                (t_rear*t_rotate).toRotationMatrix(), t_rear.toRotationMatrix());
        }
    }
};