#include <opencv/cv.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>

class DepthEstimator {
    Eigen::Vector3d t01;
    Eigen::Matrix3d R01;
    cv::Mat cameraMatrix;
    bool show = false;
    int num_disp = 32;
    bool use_sgbm = true;
    cv::Mat _map11, _map12, _map21, _map22;
    cv::cuda::GpuMat map11, map12, map21, map22;
    bool first_init = true;
    cv::Mat R,T, R1, R2, P1, P2, Q;
    double baseline = 0;

public:
    DepthEstimator(Eigen::Vector3d t0, Eigen::Matrix3d R0, Eigen::Vector3d t1, Eigen::Matrix3d R1, cv::Mat camera_mat,
    bool _show):
        cameraMatrix(camera_mat),show(_show)
    {
        t01 = t1 - t0;
        // t01 = R0.transpose() * t01;
        t01 = R0.transpose() * t01;
        
        R01 = (R0.transpose() * R1);
        cv::eigen2cv(R01, R);
        cv::eigen2cv(t01, T);
        // std::cout << "R0" << R0 << std::endl;
        // std::cout << "R1" << R1 << std::endl;
        // std::cout << "R01" << R01 << std::endl;
    }

    cv::Mat ComputeDispartiyMap(cv::cuda::GpuMat & left, cv::cuda::GpuMat & right);
    cv::Mat ComputeDepthImage(cv::cuda::GpuMat & left, cv::cuda::GpuMat & right);
};