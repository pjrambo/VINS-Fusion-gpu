#include "depth_camera_manager.h" 
#include "../estimator/parameters.h"
#include <geometry_msgs/PoseStamped.h>
#include "../utility/tic_toc.h"

using namespace Eigen;

DepthCamManager::DepthCamManager(ros::NodeHandle & _nh, FisheyeUndist * _fisheye): nh(_nh), fisheye(_fisheye) {
    pub_depth_clouds.push_back(nh.advertise<sensor_msgs::PointCloud>("depth_cloud_left", 1));
    pub_depth_clouds.push_back(nh.advertise<sensor_msgs::PointCloud>("depth_cloud_front", 1));
    pub_depth_clouds.push_back(nh.advertise<sensor_msgs::PointCloud>("depth_cloud_right", 1));
    pub_depth_clouds.push_back(nh.advertise<sensor_msgs::PointCloud>("depth_cloud_rear", 1));
    pub_depth_cloud = nh.advertise<sensor_msgs::PointCloud>("depth_cloud_rear", 1);

    pub_depth_maps.push_back(nh.advertise<sensor_msgs::Image>("depth_left", 1));
    pub_depth_maps.push_back(nh.advertise<sensor_msgs::Image>("depth_front", 1));
    pub_depth_maps.push_back(nh.advertise<sensor_msgs::Image>("depth_right", 1));
    pub_depth_maps.push_back(nh.advertise<sensor_msgs::Image>("depth_rear", 1));

    pub_depthcam_poses.push_back(nh.advertise<geometry_msgs::PoseStamped>("pose_left", 1));
    pub_depthcam_poses.push_back(nh.advertise<geometry_msgs::PoseStamped>("pose_front", 1));
    pub_depthcam_poses.push_back(nh.advertise<geometry_msgs::PoseStamped>("pose_right", 1));
    pub_depthcam_poses.push_back(nh.advertise<geometry_msgs::PoseStamped>("pose_rear", 1));



    pub_camera_up = nh.advertise<sensor_msgs::Image>("/front_stereo/left/image_raw", 1);
    pub_camera_down = nh.advertise<sensor_msgs::Image>("/front_stereo/right/image_raw", 1);

    up_cam_info_pub = nh.advertise<sensor_msgs::CameraInfo>("/front_stereo/left/camera_info", 1);
    down_cam_info_pub = nh.advertise<sensor_msgs::CameraInfo>("/front_stereo/right/camera_info", 1);

    pub_depth_cloud = nh.advertise<sensor_msgs::PointCloud>("depth_cloud", 1000);
    t_left = Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(1, 0, 0)));
    t_front = t_left * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
    t_right = t_front * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
    t_rear = t_rear * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
    t_down = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1, 0, 0)));
    t_rotate = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 0, 1));
    // t_transpose = Eigen::AngleAxisd(0, Eigen::Vector3d(0, 0, 1));

    ROS_INFO("Reading depth config from %s", depth_config.c_str());
    FILE *fh = fopen(depth_config.c_str(),"r");
    if(fh == NULL){
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        return;          
    } else {
        cv::FileStorage fsSettings(depth_config, cv::FileStorage::READ);
        estimate_front_depth = (int) fsSettings["enable_front"];
        estimate_left_depth = (int) fsSettings["enable_left"];
        estimate_right_depth = (int) fsSettings["enable_right"];
        estimate_rear_depth = (int) fsSettings["enable_rear"];
        downsample_ratio = fsSettings["downsample_ratio"];


        sgm_params.use_vworks = (int) fsSettings["use_vworks"];
        sgm_params.num_disp = fsSettings["num_disp"];
        sgm_params.block_size = fsSettings["block_size"];
        sgm_params.min_disparity = fsSettings["min_disparity"];
        sgm_params.disp12Maxdiff = fsSettings["disp12Maxdiff"];
        sgm_params.prefilterCap = fsSettings["prefilterCap"];
        sgm_params.prefilterSize = fsSettings["prefilterSize"];
        sgm_params.uniquenessRatio = fsSettings["uniquenessRatio"];
        sgm_params.speckleWindowSize = fsSettings["speckleWindowSize"];
        sgm_params.speckleRange = fsSettings["speckleRange"];
        sgm_params.mode = fsSettings["mode"];
        sgm_params.p1 = fsSettings["p1"];
        sgm_params.p2 = fsSettings["p2"];

        sgm_params.bt_clip_value = fsSettings["bt_clip_value"];
        sgm_params.ct_win_size = fsSettings["ct_win_size"];
        sgm_params.hc_win_size = fsSettings["hc_win_size"];
        sgm_params.flags = fsSettings["flags"];
        sgm_params.scanlines_mask = fsSettings["scanlines_mask"];

        pub_cloud_step = fsSettings["pub_cloud_step"];
        if (pub_cloud_step <= 0) {
            pub_cloud_step = 1;
        }
        show_disparity = fsSettings["show_disparity"];
        depth_cloud_radius = fsSettings["depth_cloud_radius"];

        pub_depth_map = (int)fsSettings["pub_depth_map"];
        pub_cloud_all =  (int)fsSettings["pub_cloud_all"];
        enable_extrinsic_calib_for_depth = (int)fsSettings["enable_extrinsic_calib"];
        std::string cfg;
        fsSettings["left_depth_RT"] >> cfg;
        dep_RT_config.push_back(cfg);        

        fsSettings["front_depth_RT"] >> cfg;
        dep_RT_config.push_back(cfg);        
        
        fsSettings["right_depth_RT"] >> cfg;
        dep_RT_config.push_back(cfg);        
        
        fsSettings["rear_depth_RT"] >> cfg;
        dep_RT_config.push_back(cfg);
    }
    fclose(fh);


    f_side = fisheye->f_side;
    cx_side = fisheye->cx_side;
    cy_side = fisheye->cy_side;
    cam_side << f_side*downsample_ratio, 0, cx_side*downsample_ratio,
                    0, f_side*downsample_ratio, cy_side*downsample_ratio, 0, 0, 1;
    cam_side_transpose << f_side*downsample_ratio, 0, cy_side*downsample_ratio,
            0, f_side*downsample_ratio, cx_side*downsample_ratio, 0, 0, 1;
    cv::eigen2cv(cam_side, cam_side_cv);
    cv::eigen2cv(cam_side_transpose, cam_side_cv_transpose);

    depth_cam = camodocal::PinholeCameraPtr(new camodocal::PinholeCamera("depth",
                  fisheye->imgWidth*downsample_ratio, fisheye->sideImgHeight*downsample_ratio,0, 0, 0, 0,
                  f_side*downsample_ratio, f_side*downsample_ratio, cx_side*downsample_ratio, cy_side*downsample_ratio));

    deps.push_back(nullptr);
    deps.push_back(nullptr);
    deps.push_back(nullptr);
    deps.push_back(nullptr);

    depth_maps.resize(4);
    pts_3ds.resize(4);
    texture_imgs.resize(4);
}


void DepthCamManager::init_with_extrinsic(Eigen::Matrix3d _ric1, Eigen::Vector3d tic1, 
        Eigen::Matrix3d _ric2, Eigen::Vector3d tic2) {

    Eigen::Vector3d _t01 = tic1 - tic2;
    auto ric1 = _ric1*t_left*t_rotate; 
    auto ric2 = _ric2*t_down*t_left*t_rotate;
    Eigen::Vector3d t01 = ric2.transpose()*_t01;
    create_depth_estimator(0, ric2.transpose() * ric1, t01);
    
    ric1 = _ric1*t_front*t_rotate; 
    ric2 = _ric2*t_down*t_front*t_rotate;
    t01 = ric2.transpose()*_t01;
    create_depth_estimator(1, ric2.transpose() * ric1, t01);

    ric1 = _ric1*t_right*t_rotate; 
    ric2 = _ric2*t_down*t_right*t_rotate;
    t01 = ric2.transpose()*_t01;
    create_depth_estimator(2, ric2.transpose() * ric1, t01);

    ric1 = _ric1*t_rear*t_rotate; 
    ric2 = _ric2*t_down*t_rear*t_rotate;
    t01 = ric2.transpose()*_t01;
    create_depth_estimator(3, ric2.transpose() * ric1, t01);
}

DepthEstimator * DepthCamManager::create_depth_estimator(int direction, Eigen::Matrix3d r01, Eigen::Vector3d t01) {
    // ROS_INFO("Creating %d depth generator...", direction);
    DepthEstimator * dep_est;
    
    std::string _output_path;
    if (direction == 0) {
        _output_path = OUTPUT_FOLDER + "/left_dep.yaml";
    }
    if (direction == 1) {
        _output_path = OUTPUT_FOLDER + "/front_dep.yaml";
    }
    if (direction == 2) {
        _output_path = OUTPUT_FOLDER + "/right_dep.yaml";
    }
    if (direction == 3) {

        _output_path = OUTPUT_FOLDER + "/rear_dep.yaml";
    }
    if (dep_RT_config[direction] != "") {
        //Build stereo with estimate extrinsic
        deps[direction] = new DepthEstimator(sgm_params, configPath + "/" + dep_RT_config[direction], cam_side_cv_transpose, show_disparity,
            enable_extrinsic_calib_for_depth, _output_path);
    } else {
        deps[direction] = new DepthEstimator(sgm_params, t01, r01, cam_side_cv_transpose, show_disparity,
            enable_extrinsic_calib_for_depth, _output_path);
    }
    return deps[direction];
}

void DepthCamManager::update_pcl_depth_from_image(ros::Time stamp, int direction, Eigen::Matrix3d ric1, Eigen::Vector3d tic1,
        Eigen::Matrix3d R, Eigen::Vector3d P, Eigen::Matrix3d ric_depth, sensor_msgs::PointCloud & pcl) {
    auto & texture_img = texture_imgs[direction];

    if (pub_cloud_step > 0 && pub_cloud_all) { 
        add_pts_point_cloud(pts_3ds[direction], R*ric1, P+R*tic1, stamp, pcl, pub_cloud_step, texture_img);
    }

    if(pub_depth_map) {
        cv::Mat depthmap = depth_maps[direction];
        sensor_msgs::ImagePtr depth_img_msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", depthmap).toImageMsg();
        depth_img_msg->header.stamp = stamp;
        pub_depth_maps[direction].publish(depth_img_msg);

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = stamp;
        pose_stamped.header.frame_id = "world";
        Eigen::Vector3d t_dep = P + R*tic1;
        Eigen::Quaterniond q_dep = Eigen::Quaterniond(R*ric_depth).normalized();
        pose_stamped.pose.position.x = t_dep.x();
        pose_stamped.pose.position.y = t_dep.y();
        pose_stamped.pose.position.z = t_dep.z();
        pose_stamped.pose.orientation.w = q_dep.w();
        pose_stamped.pose.orientation.x = q_dep.x();
        pose_stamped.pose.orientation.y = q_dep.y();
        pose_stamped.pose.orientation.z = q_dep.z();
        pub_depthcam_poses[direction].publish(pose_stamped);
    }
}

void DepthCamManager::update_depth_image(ros::Time stamp, cv::cuda::GpuMat _up_front, cv::cuda::GpuMat _down_front, 
    Eigen::Matrix3d ric1, Eigen::Vector3d tic1,
    Eigen::Matrix3d R, Eigen::Vector3d P, int direction, sensor_msgs::PointCloud & pcl, Eigen::Matrix3d ric_depth) {
    cv::cuda::GpuMat up_front, down_front;

    this->update_depth_image(direction, _up_front, _down_front, ric1, ric_depth);
    update_pcl_depth_from_image(stamp, direction, ric1, tic1, R, P, ric_depth, pcl);
    
}


void DepthCamManager::update_depth_image(int direction, cv::cuda::GpuMat _up_front, cv::cuda::GpuMat _down_front, 
    Eigen::Matrix3d ric1, Eigen::Matrix3d ric_depth) {
    cv::cuda::GpuMat up_front, down_front;

    TicToc tic_resize;
    cv::cuda::resize(_up_front, up_front, cv::Size(), downsample_ratio, downsample_ratio);
    cv::cuda::resize(_down_front, down_front, cv::Size(), downsample_ratio, downsample_ratio);

    if(ENABLE_PERF_OUTPUT) {
        ROS_INFO("Up to Resize cost %f", tic_resize.toc());
    }

    //After transpose, we need flip for rotation

    cv::cuda::GpuMat tmp;
    cv::Size size = up_front.size();
    if (_up_front.channels() == 3) {
        cv::cuda::cvtColor(up_front, up_front, cv::COLOR_BGR2GRAY);
        cv::cuda::cvtColor(down_front, down_front, cv::COLOR_BGR2GRAY);
    }


    if(ENABLE_PERF_OUTPUT) {
        ROS_INFO("Up to cvtcolor cost %f", tic_resize.toc());
    }

    cv::cuda::transpose(up_front, tmp);
    cv::cuda::flip(tmp, up_front, 0);

    cv::cuda::transpose(down_front, tmp);
    cv::cuda::flip(tmp, down_front, 0);

    if(ENABLE_PERF_OUTPUT) {
        ROS_INFO("Up to rotate cost %f", tic_resize.toc());
    }

    auto dep_est = deps[direction];
    // ROS_WARN("Dep est %d from %d", dep_est, direction);
    
    cv::Mat pointcloud_up = dep_est->ComputeDepthCloud(up_front, down_front);

    if(ENABLE_PERF_OUTPUT) {
        ROS_INFO("Up to ComputeDepthCloud cost %f", tic_resize.toc());
    }

    cv::Mat depthmap;
    if(pub_depth_map) {
        depthmap = generate_depthmap(pointcloud_up, ric1.transpose()*ric_depth);
    }

    if(ENABLE_PERF_OUTPUT) {
        ROS_INFO("Up to generate_depthmap cost %f", tic_resize.toc());
    }

     if (pub_cloud_all && RGB_DEPTH_CLOUD == 1) {
        cv::Mat up_front_cpu, tmp2;
        _up_front.download(up_front_cpu);
        cv::transpose(up_front_cpu, tmp2);
        cv::flip(tmp2, up_front_cpu, 0);
        cv::resize(up_front_cpu, up_front_cpu, cv::Size(), downsample_ratio, downsample_ratio);
        texture_imgs[direction] = up_front_cpu;
    } 



    if (pub_cloud_all && RGB_DEPTH_CLOUD == 0) {
        cv::Mat up_front_cpu;
        up_front.download(up_front_cpu);
        texture_imgs[direction] = up_front_cpu;
    }


    if(ENABLE_PERF_OUTPUT) {
        ROS_INFO("Up to save_texture cost %f", tic_resize.toc());
    }
   
    depth_maps[direction] = depthmap;
    pts_3ds[direction] = pointcloud_up;

}


void DepthCamManager::update_images(ros::Time stamp, std::vector<cv::cuda::GpuMat> & up_cams, std::vector<cv::cuda::GpuMat> & down_cams,
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

    if (publish_raw_image) {
        publish_front_images_for_external_sbgm(stamp, up_cams[2], down_cams[2], ric1, tic1, ric2, tic2, R, P);
    }

    if (pub_cloud_all) {
        pub_depth_cloud.publish(point_cloud);
    }

}

void DepthCamManager::update_images_to_buf(std::vector<cv::cuda::GpuMat> & up_cams, std::vector<cv::cuda::GpuMat> & down_cams) {
    
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


void DepthCamManager::pub_depths_from_buf(ros::Time stamp, Eigen::Matrix3d ric1, Eigen::Vector3d tic1, 
        Eigen::Matrix3d R, Eigen::Vector3d P) {
    sensor_msgs::PointCloud point_cloud;
    point_cloud.header.stamp = stamp;
    point_cloud.header.frame_id = "world";
    point_cloud.channels.resize(3);

    if (RGB_DEPTH_CLOUD >= 0) {
        point_cloud.channels[0].name = "rgb";
        point_cloud.channels[0].values.resize(0);
        point_cloud.channels[1].name = "u";
        point_cloud.channels[1].values.resize(0);
        point_cloud.channels[2].name = "v";
        point_cloud.channels[2].values.resize(0);
    }


    if (estimate_front_depth) {
        update_pcl_depth_from_image(stamp, 1, ric1*t_front*t_rotate, tic1, R, P, ric1*t_front, point_cloud);
    }

    if (estimate_left_depth) {
        update_pcl_depth_from_image(stamp, 0, ric1*t_left*t_rotate, tic1, R, P, ric1*t_left, point_cloud);
    }

    if (estimate_right_depth) {
        update_pcl_depth_from_image(stamp, 2, ric1*t_right*t_rotate, tic1, R, P, ric1*t_right, point_cloud);
    }
    
    if (estimate_rear_depth) {
        update_pcl_depth_from_image(stamp, 3, ric1*t_rear*t_rotate, tic1, R, P, ric1*t_rear, point_cloud);
    }


    if (pub_cloud_all) {
        pub_depth_cloud.publish(point_cloud);
    }
}

void DepthCamManager::add_pts_point_cloud(cv::Mat pts3d, Eigen::Matrix3d R, Eigen::Vector3d P, ros::Time stamp,
    sensor_msgs::PointCloud & pcl, int step, cv::Mat color) {
    bool rgb_color = color.channels() == 3;
    for(int v = 0; v < pts3d.rows; v += step){
        for(int u = 0; u < pts3d.cols; u += step)  
        {
            cv::Vec3f vec = pts3d.at<cv::Vec3f>(v, u);
            double x = vec[0];
            double y = vec[1];
            double z = vec[2];
            Vector3d pts_i(x, y, z);

            if (pts_i.norm() < depth_cloud_radius) {
                Vector3d w_pts_i = R * pts_i + P;
                // Vector3d w_pts_i = pts_i;

                geometry_msgs::Point32 p;
                p.x = w_pts_i(0);
                p.y = w_pts_i(1);
                p.z = w_pts_i(2);
                
                pcl.points.push_back(p);

                if (!color.empty()) {
                    int32_t rgb_packed;
                    if(rgb_color) {
                        const cv::Vec3b& bgr = color.at<cv::Vec3b>(v, u);
                        rgb_packed = (bgr[2] << 16) | (bgr[1] << 8) | bgr[0];
                    } else {
                        const uchar& bgr = color.at<uchar>(v, u);
                        rgb_packed = (bgr << 16) | (bgr << 8) | bgr;
                    }

                    pcl.channels[0].values.push_back(*(float*)(&rgb_packed));
                    pcl.channels[1].values.push_back(u);
                    pcl.channels[2].values.push_back(v);
                }
            }
        }
    }
}

void DepthCamManager::publish_world_point_cloud(cv::Mat pts3d, Eigen::Matrix3d R, Eigen::Vector3d P, ros::Time stamp,
    int dir, int step, cv::Mat color) {
    // std::cout<< "Pts3d Size " << pts3d.size() << std::endl;
    // std::cout<< "Color Size " << color.size() << std::endl;
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

    for(int v = 0; v < pts3d.rows; v += step){
        for(int u = 0; u < pts3d.cols; u += step)  
        {
            cv::Vec3f vec = pts3d.at<cv::Vec3f>(v, u);
            double x = vec[0];
            double y = vec[1];
            double z = vec[2];
            Vector3d pts_i(x, y, z);

            if (z > 0.2 && pts_i.norm() < depth_cloud_radius) {
                Vector3d w_pts_i = R * pts_i + P;
                // Vector3d w_pts_i = pts_i;

                geometry_msgs::Point32 p;
                p.x = w_pts_i(0);
                p.y = w_pts_i(1);
                p.z = w_pts_i(2);
                
                point_cloud.points.push_back(p);

                if (!color.empty()) {
                    const cv::Vec3b& bgr = color.at<cv::Vec3b>(v, u);
                    int32_t rgb_packed = (bgr[2] << 16) | (bgr[1] << 8) | bgr[0];
                    point_cloud.channels[0].values.push_back(*(float*)(&rgb_packed));

                    point_cloud.channels[1].values.push_back(u);
                    point_cloud.channels[2].values.push_back(v);
                }
            }
        }
    }

    pub_depth_clouds[dir].publish(point_cloud);
}


cv::Mat DepthCamManager::generate_depthmap(cv::Mat pts3d, Eigen::Matrix3d rel_ric_depth) const {
    cv::Mat depthmap(depth_cam->imageHeight(), depth_cam->imageWidth(), CV_32F);
    depthmap.setTo(0);
    Eigen::Matrix3d cam_mat;
    for(int v = 0; v < pts3d.rows; v += 1) {
        for(int u = 0; u < pts3d.cols; u += 1)  
        {
            cv::Vec3f vec = pts3d.at<cv::Vec3f>(v, u);
            double z = vec[2];

            Eigen::Vector3d vec3d(vec[0], vec[1], vec[2]);
            vec3d = rel_ric_depth.transpose()*vec3d;
            Eigen::Vector2d p_sphere;
            depth_cam->spaceToPlane(vec3d, p_sphere);
            //Than here is the undist points
            int px = p_sphere.x();
            int py = p_sphere.y();

            // printf("Py %d Px %d\n", py, px);
            if (py < depth_cam->imageHeight() && px < depth_cam->imageWidth() && px > 0 && py > 0) {
                depthmap.at<float>(py, px) = z;
            }
        }
    }

    return depthmap;
}



void DepthCamManager::publish_front_images_for_external_sbgm(ros::Time stamp, const cv::cuda::GpuMat front_up, const cv::cuda::GpuMat front_down,
        Eigen::Matrix3d ric1, Eigen::Vector3d tic1,
        Eigen::Matrix3d ric2, Eigen::Vector3d tic2, 
        Eigen::Matrix3d R, Eigen::Vector3d P) {

    sensor_msgs::CameraInfo cam_info_left, cam_info_right;
    cam_info_left.K[0] = fisheye->f_side*downsample_ratio;
    cam_info_left.K[1] = 0;
    // cam_info_left.K[2] = fisheye->cx_side;
    cam_info_left.K[2] = fisheye->cy_side*downsample_ratio;
    cam_info_left.K[3] = 0;
    cam_info_left.K[4] = fisheye->f_side;
    // cam_info_left.K[5] = fisheye->cy_side;
    cam_info_left.K[5] = fisheye->cx_side*downsample_ratio;
    cam_info_left.K[6] = 0;
    cam_info_left.K[7] = 0;
    cam_info_left.K[8] = 1;

    cam_info_left.header.stamp = stamp;
    cam_info_left.header.frame_id = "camera_up_front";
    // cam_info_left.width = fisheye->imgWidth;
    // cam_info_left.height = fisheye->sideImgHeight;
    cam_info_left.width = fisheye->sideImgHeight*downsample_ratio;
    cam_info_left.height = fisheye->imgWidth*downsample_ratio;
    // cam_info_left.
    // cam_info_left.
    cam_info_left.P[0] = fisheye->f_side*downsample_ratio;
    cam_info_left.P[1] = 0;
    cam_info_left.P[2] = cam_info_left.K[2];
    cam_info_left.P[3] = 0;

    cam_info_left.P[4] = 0;
    cam_info_left.P[5] = fisheye->f_side*downsample_ratio;
    cam_info_left.P[6] = cam_info_left.K[5];
    cam_info_left.P[7] = 0;
    
    cam_info_left.P[8] = 0;
    cam_info_left.P[9] = 0;
    cam_info_left.P[10] = 1;
    cam_info_left.P[11] = 0;

    // ric1 = Eigen::Matrix3d::Identity();
    // ric2 = Eigen::Matrix3d::Identity();
    ric1 = ric1*t_front*t_rotate;
    ric2 = ric2*t_down*t_front*t_rotate;
    Eigen::Vector3d t01 = tic2 - tic1;
    // t01 = R0.transpose() * t01;
    t01 = ric1.transpose() * t01;
    
    Eigen::Matrix3d R01 = (ric1.transpose() * ric2);

    cam_info_right = cam_info_left;
    cam_info_right.P[3] = -t01.x()*fisheye->f_side*downsample_ratio;
    cam_info_right.P[7] = 0;
    cam_info_right.P[11] = 0;

    Eigen::Matrix3d R_iden = Eigen::Matrix3d::Identity();
    memcpy(cam_info_left.R.data(), R_iden.data(), 9*sizeof(double));
    memcpy(cam_info_right.R.data(), R01.data(), 9*sizeof(double));


    up_cam_info_pub.publish(cam_info_left);
    down_cam_info_pub.publish(cam_info_right);

    cv::Mat up_cam, down_cam;
    front_up.download(up_cam);
    front_down.download(down_cam);

    cv::transpose(up_cam, up_cam);
    cv::flip(up_cam, up_cam, 0);
    cv::transpose(down_cam, down_cam);
    cv::flip(down_cam, down_cam, 0);
    cv::resize(up_cam, up_cam, cv::Size(), 0.5, 0.5);
    cv::resize(down_cam, down_cam, cv::Size(), 0.5, 0.5);
    sensor_msgs::ImagePtr up_img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", up_cam).toImageMsg();
    up_img_msg->header.stamp = stamp;
    up_img_msg->header.frame_id = "camera_up_front";
    sensor_msgs::ImagePtr down_img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", down_cam).toImageMsg();
    down_img_msg->header.stamp = stamp;
    down_img_msg->header.frame_id = "camera_up_front";
    

    pub_camera_up.publish(up_img_msg);
    pub_camera_down.publish(down_img_msg);

    Eigen::Vector3d cam_pos = tic1 + P;
    Eigen::Quaterniond cam_quat(ric1);
    tf::Transform transform;
    transform.setOrigin( tf::Vector3(cam_pos.x(), cam_pos.y(), cam_pos.z()) );
    tf::Quaternion q(cam_quat.x(), cam_quat.y(), cam_quat.z(), cam_quat.w());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "camera_up_front"));
}
