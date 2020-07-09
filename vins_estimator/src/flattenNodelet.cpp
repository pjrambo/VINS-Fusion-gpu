#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include "estimator/parameters.h"
#include <boost/thread.hpp>
#include "depth_generation/depth_camera_manager.h"
#include "featureTracker/fisheye_undist.hpp"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "utility/tic_toc.h"
#include "vins/FlattenImages.h"

namespace vins_nodelet_pkg
{
    class FlattenNodeletClass : public nodelet::Nodelet
    {
        public:
            FlattenNodeletClass() {}
        private:
            message_filters::Subscriber<sensor_msgs::Image> * image_sub_l;
            message_filters::Subscriber<sensor_msgs::Image> * image_sub_r;
            DepthCamManager * cam_manager = nullptr;
            vector<FisheyeUndist> fisheys_undists;

            ros::Publisher flatten_pub;

            virtual void onInit()
            {
                auto n = getNodeHandle();
                auto private_n = getPrivateNodeHandle();
                std::string config_file;
                private_n.getParam("config_file", config_file);
                
                std::cout << "config file is " << config_file << '\n';
                readParameters(config_file);

                readIntrinsicParameter(CAM_NAMES);
                
                if (ENABLE_DEPTH) {
                    // cam_manager = new DepthCamManager(n, &(estimator.featureTracker.fisheys_undists[0]));
                    // cam_manager -> init_with_extrinsic(estimator.ric[0], estimator.tic[0], estimator.ric[1], estimator.tic[1]);
                    // estimator.depth_cam_manager = cam_manager;
                }
                ROS_WARN("Flatten nodelet start waiting for image and imu...");

                image_sub_l = new message_filters::Subscriber<sensor_msgs::Image> (n, IMAGE0_TOPIC, 100);
                image_sub_r = new message_filters::Subscriber<sensor_msgs::Image> (n, IMAGE1_TOPIC, 100);
                sync = new message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> (*image_sub_l, *image_sub_r, 10);
                sync->registerCallback(boost::bind(&FlattenNodeletClass::img_callback, this, _1, _2));
                
                flatten_pub = n.advertise<vins::FlattenImages>("/vins_estimator/flattened_raw", 1);
            }

            void img_callback(const sensor_msgs::ImageConstPtr &img1_msg, const sensor_msgs::ImageConstPtr &img2_msg)
            {
                auto img1 = getImageFromMsg(img1_msg);
                auto img2 = getImageFromMsg(img2_msg);

                // ROS_INFO("Imgs recived, start flatten");
                vector<cv::Mat> fisheye_imgs_up;
                vector<cv::Mat> fisheye_imgs_down;

                static double flatten_time_sum = 0, pack_send_time = 0;
                static double count = 0;

                count += 1;

                TicToc t_f;

                bool is_color = false;

                if (USE_GPU) {
                    is_color = true;
                    fisheye_imgs_up = fisheys_undists[0].undist_all_cuda_cpu(img1->image, is_color); 
                    fisheye_imgs_down = fisheys_undists[1].undist_all_cuda_cpu(img2->image, is_color);
                } else {
                    fisheys_undists[0].stereo_flatten(img1->image, img2->image, &fisheys_undists[1], 
                        fisheye_imgs_up, fisheye_imgs_down, false, 
                        enable_up_top, enable_rear_side, enable_down_top, enable_rear_side);
                }

                flatten_time_sum += t_f.toc();

                TicToc t_p;
                vins::FlattenImages images;
                images.header = img1_msg->header;

                for (unsigned int i = 0; i < fisheye_imgs_up.size(); i++) {
                    cv_bridge::CvImage outImg;
                    outImg.header = img1_msg->header;
                    if (is_color) {
                        outImg.encoding = "bgr8";
                    } else {
                        outImg.encoding = "mono8";
                    }

                    outImg.image = fisheye_imgs_up[i];
                    images.up_cams.push_back(*outImg.toImageMsg());
                }

                for (unsigned int i = 0; i < fisheye_imgs_down.size(); i++) {
                    cv_bridge::CvImage outImg;
                    outImg.header = img1_msg->header;
                    if (is_color) {
                        outImg.encoding = "bgr8";
                    } else {
                        outImg.encoding = "mono8";
                    }

                    outImg.image = fisheye_imgs_down[i];
                    images.down_cams.push_back(*outImg.toImageMsg());
                }

                flatten_pub.publish(images);
                pack_send_time += t_p.toc();

                ROS_INFO("Flatten AVG %fms; Pack and send AVG %fms", flatten_time_sum/count, pack_send_time/count);
            }


            void readIntrinsicParameter(const vector<string> &calib_file)
            {
                for (size_t i = 0; i < calib_file.size(); i++)
                {
                    if (FISHEYE) {
                        ROS_INFO("Flatten read fisheye %s, id %ld", calib_file[i].c_str(), i);
                        FisheyeUndist un(calib_file[i].c_str(), i, FISHEYE_FOV, true, COL);
                        fisheys_undists.push_back(un);
                    }
                }
            }


            cv_bridge::CvImageConstPtr getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
            {
                cv_bridge::CvImageConstPtr ptr;
                if (img_msg->encoding == "8UC1")
                {
                    ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::MONO8);
                }
                else
                {
                    if (FISHEYE) {
                        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
                    } else {
                        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);        
                    }
                }
                return ptr;
            }

            message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> * sync;
            double last_time;
            bool last_time_initialized;
    };
    PLUGINLIB_EXPORT_CLASS(vins_nodelet_pkg::FlattenNodeletClass, nodelet::Nodelet);

}
