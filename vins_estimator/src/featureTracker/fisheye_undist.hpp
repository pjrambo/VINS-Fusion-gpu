#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/cudaimgproc.hpp>
#include <camodocal/camera_models/CameraFactory.h>
#include "cv_bridge/cv_bridge.h"
#include <experimental/filesystem>
#include <opencv2/cudawarping.hpp>

#define DEG_TO_RAD (M_PI / 180.0)



class FisheyeUndist {

    camodocal::CameraPtr cam;
    int imgWidth = 0;
    double fov = 0; //in degree
    std::vector<cv::Mat> undistMaps;
    std::vector<cv::cuda::GpuMat> undistMapsGPUX;
    std::vector<cv::cuda::GpuMat> undistMapsGPUY;
    bool enable_cuda = false;
    Eigen::Vector3d cameraRotation;
public:
    camodocal::CameraPtr cam_top;
    camodocal::CameraPtr cam_side;

    FisheyeUndist(const std::string & camera_config_file, bool _enable_cuda = true, int imgWidth = 500):
    imgWidth(imgWidth), fov(90), cameraRotation(0, 0, 0), enable_cuda(_enable_cuda) {
        cam = camodocal::CameraFactory::instance()
            ->generateCameraFromYamlFile(camera_config_file);

        undistMaps = generateAllUndistMap(cam, cameraRotation, imgWidth, fov);
        if (enable_cuda) {
            for (auto mat : undistMaps) {
                cv::Mat xy[2];
                cv::split(mat, xy);
                undistMapsGPUX.push_back(cv::cuda::GpuMat(xy[0]));
                undistMapsGPUY.push_back(cv::cuda::GpuMat(xy[1]));
            }
        }
    }

    cv::cuda::GpuMat undist_id_cuda(cv::Mat image, int _id) {
        cv::cuda::GpuMat img_cuda(image);
        cv::cuda::GpuMat output;
        cv::cuda::remap(img_cuda, output, undistMapsGPUX[_id], undistMapsGPUY[_id], cv::INTER_LINEAR);
        return output;
    }

    std::vector<cv::cuda::GpuMat> undist_all_cuda(cv::Mat image) {
        cv::cuda::GpuMat img_cuda(image);
        std::vector<cv::cuda::GpuMat> ret;
        for (int i = 0; i < undistMaps.size(); i++) {
            cv::cuda::GpuMat output;
            cv::cuda::remap(img_cuda, output, undistMapsGPUX[i], undistMapsGPUY[i], cv::INTER_LINEAR);
            ret.push_back(output);
        }
        return ret;
    }


    std::vector<cv::Mat> generateAllUndistMap(camodocal::CameraPtr p_cam,
                                          Eigen::Vector3d rotation,
                                          const unsigned &imgWidth,
                                          const double &fov //degree
    ) {
        ROS_INFO("Generating undistortion maps:");
        double sideVerticalFOV = (fov - 180) * DEG_TO_RAD;
        if (sideVerticalFOV < 0)
            sideVerticalFOV = 0;
        double centerFOV = fov * DEG_TO_RAD - sideVerticalFOV * 2;
        ROS_INFO("Center FOV: %f_center", centerFOV);
        int sideImgHeight = sideVerticalFOV / centerFOV * imgWidth;
        ROS_INFO("Side image height: %d", sideImgHeight);
        std::vector<cv::Mat> maps;
        maps.reserve(5);

        // test points
        Eigen::Vector3d testPoints[] = {
            Eigen::Vector3d(0, 0, 1),
            Eigen::Vector3d(1, 0, 1),
            Eigen::Vector3d(0, 1, 1),
            Eigen::Vector3d(1, 1, 1),
        };
        for (int i = 0; i < sizeof(testPoints) / sizeof(Eigen::Vector3d); i++)
        {
            Eigen::Vector2d temp;
            p_cam->spaceToPlane(testPoints[i], temp);
            ROS_INFO("Test point %d : (%.2f,%.2f,%.2f) projected to (%.2f,%.2f)", i,
                    testPoints[i][0], testPoints[i][1], testPoints[i][2],
                    temp[0], temp[1]);
        }

        // center pinhole camera orientation
        // auto t = Eigen::AngleAxis<double>(rotation.norm(), rotation.normalized()).inverse();
        auto t = Eigen::AngleAxisd(rotation.z() / 180 * M_PI, Eigen::Vector3d::UnitZ()) * 
            Eigen::AngleAxisd(rotation.y() / 180 * M_PI, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(rotation.x() / 180 * M_PI, Eigen::Vector3d::UnitX());
        // .inverse();

        // calculate focal length of fake pinhole cameras (pixel size = 1 unit)
        double f_center = (double)imgWidth / 2 / tan(centerFOV / 2);
        double f_side = (double)imgWidth / 2;
        ROS_INFO("Pinhole cameras focal length: center %f side %f", f_center, f_side);

        cam_top = camodocal::PinholeCameraPtr( new camodocal::PinholeCamera("top",
                  imgWidth, imgWidth,0, 0, 0, 0,
                  f_center, f_center, imgWidth/2, imgWidth/2));
         

        cam_side = camodocal::PinholeCameraPtr(new camodocal::PinholeCamera("side",
                  imgWidth, sideImgHeight,0, 0, 0, 0,
                  f_side, f_side, imgWidth/2, sideImgHeight/2));

        maps.push_back(genOneUndistMap(p_cam, t, imgWidth, imgWidth, f_center));

        if (sideImgHeight > 0)
        {
            //facing y
            t = t * Eigen::AngleAxis<double>(-M_PI / 2, Eigen::Vector3d(1, 0, 0));
            maps.push_back(genOneUndistMap(p_cam, t, imgWidth, sideImgHeight, f_side));

            //turn right/left?
            t = t * Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0));
            maps.push_back(genOneUndistMap(p_cam, t, imgWidth, sideImgHeight, f_side));
            t = t * Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0));
            maps.push_back(genOneUndistMap(p_cam, t, imgWidth, sideImgHeight, f_side));
            t = t * Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0));
            maps.push_back(genOneUndistMap(p_cam, t, imgWidth, sideImgHeight, f_side));
        }
        return maps;
    }

    cv::Mat genOneUndistMap(
        camodocal::CameraPtr p_cam,
        Eigen::Quaterniond rotation,
        const unsigned &imgWidth,
        const unsigned &imgHeight,
        const double &f_center) {
                cv::Mat map = cv::Mat(imgHeight, imgWidth, CV_32FC2);
        ROS_INFO("Generating map of size (%d,%d)", map.size[0], map.size[1]);
        ROS_INFO("Perspective facing (%.2f,%.2f,%.2f)",
                (rotation * Eigen::Vector3d(0, 0, 1))[0],
                (rotation * Eigen::Vector3d(0, 0, 1))[1],
                (rotation * Eigen::Vector3d(0, 0, 1))[2]);
        for (int x = 0; x < imgWidth; x++)
            for (int y = 0; y < imgHeight; y++)
            {
                Eigen::Vector3d objPoint =
                    rotation *
                    Eigen::Vector3d(
                        ((double)x - (double)imgWidth / 2),
                        ((double)y - (double)imgHeight / 2),
                        f_center);
                Eigen::Vector2d imgPoint;
                p_cam->spaceToPlane(objPoint, imgPoint);
                map.at<cv::Vec2f>(cv::Point(x, y)) = cv::Vec2f(imgPoint.x(), imgPoint.y());
            }

        ROS_INFO("Upper corners: (%.2f, %.2f), (%.2f, %.2f)",
                map.at<cv::Vec2f>(cv::Point(0, 0))[0],
                map.at<cv::Vec2f>(cv::Point(0, 0))[1],
                map.at<cv::Vec2f>(cv::Point(imgWidth - 1, 0))[0],
                map.at<cv::Vec2f>(cv::Point(imgWidth - 1, 0))[1]);

        Eigen::Vector3d objPoint =
            rotation *
            Eigen::Vector3d(
                ((double)0 - (double)imgWidth / 2),
                ((double)0 - (double)imgHeight / 2),
                f_center);
        std::cout << objPoint << std::endl;

        objPoint =
            rotation *
            Eigen::Vector3d(
                ((double)imgWidth / 2),
                ((double)0 - (double)imgHeight / 2),
                f_center);
        std::cout << objPoint << std::endl;

        return map;
    }

};

