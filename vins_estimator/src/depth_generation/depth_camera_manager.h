#pragma once

#include <sensor_msgs/PointCloud.h>
#include <ros/ros.h>
#include <eigen3/Eigen/Eigen>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include "../featureTracker/fisheye_undist.hpp"
#include <tf/transform_broadcaster.h>
#include "depth_estimator.h"
#include "color_disparity_graph.hpp"

class DepthCamManager {

    ros::Publisher pub_depth_cloud_front;
    ros::Publisher pub_depth_cloud_left;
    ros::Publisher pub_depth_cloud_right;
    ros::Publisher pub_depth_cloud_rear;

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
    
    SGMParams sgm_params;
    
    double downsample_ratio = 1.0;
    Eigen::Matrix3d cam_side;
    Eigen::Matrix3d cam_side_transpose;
    cv::Mat cam_side_cv, cam_side_cv_transpose;

    int pub_cloud_step = 1;

    std::vector<DepthEstimator *> deps;

    int show_disparity = 0;
    double depth_cloud_radius = 5;

public:
    FisheyeUndist * fisheye = nullptr;

    Eigen::Quaterniond t1, t2, t3, t4, t_down, t_transpose;
    double f_side, cx_side, cy_side;

    DepthCamManager(ros::NodeHandle & _nh, FisheyeUndist * _fisheye);

    void update_depth_image(ros::Time stamp, cv::cuda::GpuMat _up_front, cv::cuda::GpuMat _down_front, 
        Eigen::Matrix3d ric1, Eigen::Vector3d tic1, 
        Eigen::Matrix3d ric2, Eigen::Vector3d tic2,
        Eigen::Matrix3d R, Eigen::Vector3d P, int direction);

    void update_images(ros::Time stamp, std::vector<cv::cuda::GpuMat> & up_cams, std::vector<cv::cuda::GpuMat> & down_cams,
        Eigen::Matrix3d ric1, Eigen::Vector3d tic1,
        Eigen::Matrix3d ric2, Eigen::Vector3d tic2, 
        Eigen::Matrix3d R, Eigen::Vector3d P
    );
    
    void publish_world_point_cloud(cv::Mat pts3d, Eigen::Matrix3d R, Eigen::Vector3d P, ros::Time stamp,
        int dir, int step = 3, cv::Mat color = cv::Mat());

    void publish_front_images_for_external_sbgm(ros::Time stamp, const cv::cuda::GpuMat front_up, const cv::cuda::GpuMat front_down,
            Eigen::Matrix3d ric1, Eigen::Vector3d tic1,
            Eigen::Matrix3d ric2, Eigen::Vector3d tic2, 
            Eigen::Matrix3d R, Eigen::Vector3d P);
};