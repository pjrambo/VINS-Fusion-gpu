#pragma once
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include "../utility/tic_toc.h"
#include "../utility/utility.h"

#ifdef USE_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#endif

#define ORB_HAMMING_DISTANCE 40 //Max hamming
#define ORB_UV_DISTANCE 1.5 //UV distance bigger than mid*this will be removed
#define MINIUM_ESSENTIALMAT_SIZE 10
#define GOOD_R_THRES 0.1
#define GOOD_T_THRES 0.1
#define MAX_FIND_ESSENTIALMAT_PTS 10000
#define MAX_ESSENTIAL_OUTLIER_COST 0.02
#define PTS_NUM_REG 200
#define MAX_ACCEPT_COV 0.1
using namespace std;

#define CALIBCOLS 3
#define CALIBROWS 6

class StereoOnlineCalib {
    cv::Mat cameraMatrix;
    cv::Mat R, T;
    cv::Mat R0, T0;
    cv::Mat E;
    Eigen::Matrix3d E_eig, E0_eig;
    Eigen::Vector3d T_eig, T0_eig;
    Eigen::Vector3d T_eig_norm;
    Eigen::Matrix3d R_eig, R0_eig;

    int width;
    int height;
    bool show;
    std::vector<cv::Point2f> left_pts, right_pts;
    double baseline = 0;
public:
    StereoOnlineCalib(cv::Mat _R, cv::Mat _T, cv::Mat _cameraMatrix, int _width, int _height, bool _show):
        cameraMatrix(_cameraMatrix), R0(_R), T0(_T), width(_width), height(_height), show(_show)
    {
        cv::cv2eigen(_R, R0_eig);
        cv::cv2eigen(_T, T0_eig);
        baseline = -T0_eig.x();
        update(_R, _T);
        E0_eig = E_eig;
    }

    cv::Mat get_rotation() const {
        return R;
    }

    cv::Mat get_translation() const {
        return T;
    }
    
    void update(cv::Mat R, cv::Mat T) {
        this->R = R;
        this->T = T;
        cv::cv2eigen(T, T_eig);
        cv::cv2eigen(R, R_eig);

        T_eig_norm = T_eig.normalized();

        std::cerr << "Teig_norm" << T_eig_norm << std::endl;

        auto rpy = Utility::R2ypr(R_eig);

        ROS_WARN("New Relative pose Y %f P %f R %f", rpy.x(), rpy.y(), rpy.z());

        Eigen::Matrix3d Tcross;
        Tcross << 0, -T_eig_norm.z(), T_eig_norm.y(),
                T_eig_norm.z(), 0, -T_eig_norm.x(),
                -T_eig_norm.y(), T_eig_norm.x(), 0;
        E_eig = Tcross*R_eig;
        cv::eigen2cv(E_eig, E);
    }


    void filter_points_by_region(std::vector<cv::Point2f> & good_left, std::vector<cv::Point2f> & good_right);

#ifdef USE_CUDA
    void find_corresponding_pts(cv::cuda::GpuMat & img1, cv::cuda::GpuMat & img2, std::vector<cv::Point2f> & Pts1, std::vector<cv::Point2f> & Pts2);
    bool calibrate_extrincic(cv::cuda::GpuMat & left, cv::cuda::GpuMat & right);
#endif

    void find_corresponding_pts(cv::Mat & img1, cv::Mat & img2, std::vector<cv::Point2f> & Pts1, std::vector<cv::Point2f> & Pts2) {}
    bool calibrate_extrincic(cv::Mat & left, cv::Mat & right) {}

    static std::vector<cv::KeyPoint> detect_orb_by_region(cv::Mat & _img, int features, int cols = 2, int rows = 4);
    bool calibrate_extrinsic_opencv(const std::vector<cv::Point2f> & left_pts, const std::vector<cv::Point2f> & right_pts);
    bool calibrate_extrinsic_optimize(const std::vector<cv::Point2f> & left_pts, const std::vector<cv::Point2f> & right_pts);
};
