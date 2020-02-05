#pragma once

#include <sensor_msgs/PointCloud.h>
#include <ros/ros.h>
#include <eigen3/Eigen/Eigen>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include "fisheye_undist.hpp"

class DepthCamManager {

    ros::Publisher pub_depth_map;
    ros::Publisher pub_depth_cloud;
    ros::Publisher up_cam_info_pub, down_cam_info_pub;
    ros::Publisher pub_camera_up, pub_camera_down;
    ros::NodeHandle nh;

public:
    FisheyeUndist * fisheye = nullptr;

    Eigen::Quaterniond t1, t2, t3, t4, t_down;


    DepthCamManager(ros::NodeHandle & _nh): nh(_nh) {
        pub_depth_map = nh.advertise<sensor_msgs::Image>("front_depthmap", 1);


        pub_camera_up = nh.advertise<sensor_msgs::Image>("/front_stereo/left/image_raw", 1);
        pub_camera_down = nh.advertise<sensor_msgs::Image>("/front_stereo/right/image_raw", 1);

        up_cam_info_pub = nh.advertise<sensor_msgs::CameraInfo>("/front_stereo/left/camera_info", 1);
        down_cam_info_pub = nh.advertise<sensor_msgs::CameraInfo>("/front_stereo/right/camera_info", 1);

        pub_depth_cloud = nh.advertise<sensor_msgs::PointCloud>("depth_cloud", 1000);
        t1 = Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(1, 0, 0)));
        t2 = t1 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
        t3 = t2 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
        t4 = t3 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
        t_down = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1, 0, 0)));
    }

    void update_images(ros::Time stamp, std::vector<cv::cuda::GpuMat> & up_cams, std::vector<cv::cuda::GpuMat> & down_cams,
        Eigen::Matrix3d ric1, Eigen::Vector3d tic1,
        Eigen::Matrix3d ric2, Eigen::Vector3d tic2, 
        Eigen::Matrix3d R, Eigen::Vector3d P
    ) {
        publish_front_images(stamp, up_cams[2], down_cams[2], ric1, tic1, ric2, tic2, R, P);
    }

    void publish_front_images(ros::Time stamp, const cv::cuda::GpuMat front_up, const cv::cuda::GpuMat front_down,
            Eigen::Matrix3d ric1, Eigen::Vector3d tic1,
            Eigen::Matrix3d ric2, Eigen::Vector3d tic2, 
            Eigen::Matrix3d R, Eigen::Vector3d P) {
        sensor_msgs::CameraInfo cam_info_left, cam_info_right;
        cam_info_left.K[0] = fisheye->f_side;
        cam_info_left.K[1] = 0;
        cam_info_left.K[2] = fisheye->cx_side;
        cam_info_left.K[3] = 0;
        cam_info_left.K[4] = fisheye->f_side;
        cam_info_left.K[5] = fisheye->cy_side;
        cam_info_left.K[6] = 0;
        cam_info_left.K[7] = 0;
        cam_info_left.K[8] = 1;

        cam_info_left.header.stamp = stamp;
        cam_info_left.width = fisheye->imgWidth;
        cam_info_left.height = fisheye->sideImgHeight;
        // cam_info_left.
        // cam_info_left.
        cam_info_left.P[0] = fisheye->f_side;
        cam_info_left.P[1] = 0;
        cam_info_left.P[2] = fisheye->cx_side;
        cam_info_left.P[3] = 0;

        cam_info_left.P[4] = 0;
        cam_info_left.P[5] = fisheye->f_side;
        cam_info_left.P[6] = fisheye->cy_side;
        cam_info_left.P[7] = 0;
        
        cam_info_left.P[8] = 0;
        cam_info_left.P[9] = 0;
        cam_info_left.P[10] = 1;
        cam_info_left.P[11] = 0;

        Eigen::Vector3d t01 = tic2 - tic1;
        // t01 = R0.transpose() * t01;
        t01 = ric1.transpose() * t01;
        
        Eigen::Matrix3d R01 = (ric1.transpose() * ric2);

        cam_info_right = cam_info_left;
        cam_info_right.P[3] = t01.x();
        cam_info_right.P[7] = t01.y();
        cam_info_right.P[11] = 0;

        memcpy(cam_info_right.R.data(), R01.data(), 9*sizeof(double));


        up_cam_info_pub.publish(cam_info_left);
        down_cam_info_pub.publish(cam_info_right);

        cv::Mat up_cam, down_cam;
        front_up.download(up_cam);
        front_down.download(down_cam);

        sensor_msgs::ImagePtr up_img_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", up_cam).toImageMsg();
        sensor_msgs::ImagePtr down_img_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", down_cam).toImageMsg();

        pub_camera_up.publish(up_img_msg);
        pub_camera_down.publish(down_img_msg);
    }

    void publish_depthmap(cv::Mat & depthmap) {
        sensor_msgs::ImagePtr depth_img_msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", depthmap).toImageMsg();
        pub_depth_map.publish(depth_img_msg);
    }

    void publish_point_cloud(cv::Mat pts3d, Eigen::Matrix3d ric, Eigen::Vector3d tic, Eigen::Matrix3d R, Eigen::Vector3d P) {
        sensor_msgs::PointCloud point_cloud;
        // point_cloud.header = header;
        /*
        for(int v = 0; v < estimator.depthmap_front.rows; v += 3){
            for(int u = 0; u < estimator.depthmap_front.cols; u += 3)  
            {
                // double z = estimator.depthmap_front.at<float>(v, u);
                cv::Vec3f vec = pts3d.at<cv::Vec3f>(v, u);
                double x = vec[0];
                double y = vec[1];
                double z = vec[2];
                if (z > 0.2) {
                    Vector3d pts_i(x, y, z);
                    Eigen::Quaterniond t1(Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(1, 0, 0)));
                    Eigen::Quaterniond t2 =  t1 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
                    Vector3d w_pts_i = R * ric * t2 * pts_i + P;

                    geometry_msgs::Point32 p;
                    p.x = w_pts_i(0);
                    p.y = w_pts_i(1);
                    p.z = w_pts_i(2);
                    
                    point_cloud.points.push_back(p);
                }
            }
        }*/
        pub_depth_cloud.publish(point_cloud);
    }
};