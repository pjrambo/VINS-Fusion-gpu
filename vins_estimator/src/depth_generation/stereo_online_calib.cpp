#include "stereo_online_calib.hpp"

bool StereoOnlineCalib::calibrate_extrinsic_opencv(const std::vector<cv::Point2f> & left_pts, 
    const std::vector<cv::Point2f> & right_pts) {
    if (left_pts.size() < 50) {
        return false;
    }
    TicToc tic2;
    vector<uchar> status;
    std::vector<cv::Point2f> left_good, right_good;
    cv::Mat essentialMat = cv::findEssentialMat(left_pts, right_pts, cameraMatrix, cv::RANSAC, 0.99, 1000, status);
    int status_count = 0;
    for (int i = 0; i < status.size(); i++) {
        auto u = status[i];
        if (u > 0) {
            status_count += u;
            left_good.push_back(left_pts[i]);
            right_good.push_back(right_pts[i]);
        }
    }

    if (status_count < MINIUM_ESSENTIALMAT_SIZE) {
        return false;
    }

    ROS_INFO("Find EssentialMat with %ld/%d pts use %fms", left_pts.size(), status_count, tic2.toc());
    cv::Mat _R, t;
    int num = cv::recoverPose(essentialMat, left_pts, right_pts, cameraMatrix, _R, t);
    ROS_INFO("Recorver pose take with %ld/%d pts use %fms", left_good.size(), num, tic2.toc());

    double disR = norm(R0 - _R);
    double disT = norm(t - T0/scale);

    // std::cerr << "R" << _R << "DIS R" << disR << std::endl;
    // std::cerr << "T" << t*scale << "DIS T" << disT << std::endl;

    if (disR < GOOD_R_THRES && disT < GOOD_T_THRES) {
        update(_R, t*scale);
        ROS_INFO("Update R T");
        std::cout << "R" << R << std::endl;
        std::cout << "T" << T << std::endl;
        return true;
    }
}

bool StereoOnlineCalib::calibrate_extrincic(cv::cuda::GpuMat & left, cv::cuda::GpuMat & right) {

    std::vector<cv::Point2f> Pts1;
    std::vector<cv::Point2f> Pts2;
    find_corresponding_pts(left, right, Pts1, Pts2);

    if (Pts1.size() < MINIUM_ESSENTIALMAT_SIZE) {
        return false;
    }

    left_pts.insert( left_pts.end(), Pts1.begin(), Pts1.end() );
    right_pts.insert( right_pts.end(), Pts2.begin(), Pts2.end() );

    while (left_pts.size() > MAX_FIND_ESSENTIALMAT_PTS) {
        left_pts.erase(left_pts.begin());
        right_pts.erase(right_pts.begin());
    }

    return calibrate_extrinsic_opencv(left_pts, right_pts);
}

std::vector<cv::KeyPoint> StereoOnlineCalib::detect_orb_by_region(cv::Mat & _img, int features, int cols, int rows) {
    int small_width = _img.cols / cols;
    int small_height = _img.rows / rows;
    // printf("Cut to W %d H %d for FAST\n", small_width, small_height);
    
    auto _orb = cv::ORB::create(100, 1.2f, 8, 31, 0, 4, cv::ORB::HARRIS_SCORE, 31, 20);
    std::vector<cv::KeyPoint> ret;
    for (int i = 0; i < cols; i ++) {
        for (int j = 0; j < rows; j ++) {
            std::vector<cv::KeyPoint> kpts;
            cv::imshow("Rect", _img(cv::Rect(small_width*i, small_width*j, small_width, small_height)));
            _orb->detect(_img(cv::Rect(small_width*i, small_width*j, small_width, small_height)), kpts);
            // printf("Detect %ld feature in reigion %d %d\n", kpts.size(), i, j);

            for (auto kp : kpts) {
                kp.pt.x = kp.pt.x + small_width*i;
                kp.pt.y = kp.pt.y + small_width*j;
                ret.push_back(kp);
            }
        }
    }

    return ret;
}

std::vector<cv::DMatch> filter_by_x(const std::vector<cv::DMatch> & matches, 
    std::vector<cv::KeyPoint> query_pts, 
    std::vector<cv::KeyPoint> train_pts, double OUTLIER_XY_PRECENT) {
    std::vector<cv::DMatch> good_matches;
    std::vector<float> dxs;
    for (auto gm : matches) {
        dxs.push_back(query_pts[gm.queryIdx].pt.x - train_pts[gm.trainIdx].pt.x);
    }

    std::sort(dxs.begin(), dxs.end());

    int num = dxs.size();
    int l = num*OUTLIER_XY_PRECENT;
    if (l == 0) {
        l = 1;
    }
    int r = num*(1-OUTLIER_XY_PRECENT);
    if (r >= num - 1) {
        r = num - 2;
    }

    if (r <= l ) {
        return good_matches;
    }

    // printf("MIN DX DIS:%f, l:%f m:%f r:%f END:%f\n", dxs[0], dxs[l], dxs[num/2], dxs[r], dxs[dxs.size() - 1]);

    double lv = dxs[l];
    double rv = dxs[r];

    for (auto gm: matches) {
        if (query_pts[gm.queryIdx].pt.x - train_pts[gm.trainIdx].pt.x > lv && query_pts[gm.queryIdx].pt.x - train_pts[gm.trainIdx].pt.x < rv) {
            good_matches.push_back(gm);
        }
    }

    return good_matches;
}

std::vector<cv::DMatch> filter_by_y(const std::vector<cv::DMatch> & matches, 
    std::vector<cv::KeyPoint> query_pts, 
    std::vector<cv::KeyPoint> train_pts, double OUTLIER_XY_PRECENT) {
    std::vector<cv::DMatch> good_matches;
    std::vector<float> dys;
    for (auto gm : matches) {
        dys.push_back(query_pts[gm.queryIdx].pt.y - train_pts[gm.trainIdx].pt.y);
    }

    std::sort(dys.begin(), dys.end());

    int num = dys.size();
    int l = num*OUTLIER_XY_PRECENT;
    if (l == 0) {
        l = 1;
    }
    int r = num*(1-OUTLIER_XY_PRECENT);
    if (r >= num - 1) {
        r = num - 2;
    }

    if (r <= l ) {
        return good_matches;
    }

    // printf("MIN DX DIS:%f, l:%f m:%f r:%f END:%f\n", dys[0], dys[l], dys[num/2], dys[r], dys[dys.size() - 1]);

    double lv = dys[l];
    double rv = dys[r];

    for (auto gm: matches) {
        if (query_pts[gm.queryIdx].pt.y - train_pts[gm.trainIdx].pt.y > lv && query_pts[gm.queryIdx].pt.y - train_pts[gm.trainIdx].pt.y < rv) {
            good_matches.push_back(gm);
        }
    }

    return good_matches;
}

std::vector<cv::DMatch> filter_by_hamming(const std::vector<cv::DMatch> & matches) {
    std::vector<cv::DMatch> good_matches;
    std::vector<float> dys;
    for (auto gm : matches) {
        dys.push_back(gm.distance);
    }

    if (dys.size() == 0) {
        return good_matches;
    }

    std::sort(dys.begin(), dys.end());

    // printf("MIN DX DIS:%f, 2min %fm ax %f\n", dys[0], 2*dys[0], dys[dys.size() - 1]);

    double max_hamming = 2*dys[0];
    if (max_hamming < ORB_HAMMING_DISTANCE) {
        max_hamming = ORB_HAMMING_DISTANCE;
    }
    for (auto gm: matches) {
        if (gm.distance < max_hamming) {
            good_matches.push_back(gm);
        }
    }

    return good_matches;
}


//Assue undist image
Eigen::Vector3d undist(const cv::Point2f & pt, const cv::Mat & cameraMatrix) {
    double x = (pt.x - cameraMatrix.at<double>(0, 2))/ cameraMatrix.at<double>(0, 0);
    double y = (pt.y - cameraMatrix.at<double>(1, 2))/ cameraMatrix.at<double>(1, 1);
    return Eigen::Vector3d(x, y, 1);
}

std::vector<cv::DMatch> filter_by_E(const std::vector<cv::DMatch> & matches,     
    std::vector<cv::KeyPoint> query_pts, 
    std::vector<cv::KeyPoint> train_pts, 
    cv::Mat cameraMatrix, Eigen::Matrix3d E) {
    std::vector<cv::DMatch> good_matches;
    std::vector<float> dys;

    for (auto gm: matches) {
        auto pt1 = train_pts[gm.trainIdx].pt;
        auto pt2 = query_pts[gm.queryIdx].pt;

        auto f1 = undist(pt1, cameraMatrix);
        auto f2 = undist(pt2, cameraMatrix);

        auto cost = f1.transpose()*E*f2;
        if (cost.norm() < MAX_ESSENTIAL_OUTLIER_COST) {
            good_matches.push_back(gm);
        }
    }
    return good_matches;
}

void StereoOnlineCalib::find_corresponding_pts(cv::cuda::GpuMat & img1, cv::cuda::GpuMat & img2, std::vector<cv::Point2f> & Pts1, std::vector<cv::Point2f> & Pts2) {
    TicToc tic;
    std::vector<cv::KeyPoint> kps1, kps2;
    std::vector<cv::DMatch> good_matches;
    // bool use_surf = false;

    // auto _orb = cv::ORB::create(1000, 1.2f, 8, 31, 0, 4, cv::ORB::HARRIS_SCORE, 31, 20);
    std::cout << img1.size() << std::endl;
    // auto _orb = cv::cuda::ORB::create(1000, 1.2f, 8, 31, 0, 4, cv::ORB::HARRIS_SCORE, 31, 20);
    // cv::Mat _mask(img1.size(), CV_8UC1, cv::Scalar(255));
    // cv::cuda::GpuMat mask(_mask);
    
    cv::Mat desc1, desc2;
    cv::Mat _img1, _img2, mask;
    
    img1.download(_img1);
    img2.download(_img2);
    // detect_orb_by_region

    auto _orb = cv::ORB::create(1000, 1.2f, 8, 31, 0, 4, cv::ORB::HARRIS_SCORE, 31, 20);
    _orb->detectAndCompute(_img1, mask, kps1, desc1);
    _orb->detectAndCompute(_img2, mask, kps2, desc2);

    // auto _orb = cv::ORB::create(1000, 1.2f, 8, 31, 0, 4, cv::ORB::HARRIS_SCORE, 31, 20);
    // kps1 = detect_orb_by_region(_img1, 1000);
    // kps2 = detect_orb_by_region(_img2, 1000);
    // _orb->compute(_img1, kps1, desc1);
    // _orb->compute(_img2, kps2, desc2);

    size_t j = 0;

    cv::BFMatcher bfmatcher(cv::NORM_HAMMING2, true);
    std::vector<cv::DMatch> matches;
    bfmatcher.match(desc2, desc1, matches);
    matches = filter_by_hamming(matches);

    double thres = 0.05;
    
    // matches = filter_by_x(matches, kps2, kps1, thres);
    // matches = filter_by_y(matches, kps2, kps1, thres);

    // matches = filter_by_E(matches, kps2, kps1, cameraMatrix, E_eig);

    vector<cv::Point2f> _pts1, _pts2;
    vector<uchar> status;
    for (auto gm : matches) {
        auto _id1 = gm.trainIdx;
        auto _id2 = gm.queryIdx;
        _pts1.push_back(kps1[_id1].pt);
        _pts2.push_back(kps2[_id2].pt);
    }

    ROS_INFO("BRIEF MATCH cost %fms", tic.toc());

    TicToc tic0;
    if (_pts1.size() > MINIUM_ESSENTIALMAT_SIZE) {
        cv::findEssentialMat(_pts1, _pts2, cameraMatrix, cv::RANSAC, 0.99, 1.0, status);
    }

    for(int i = 0; i < _pts1.size(); i ++) {
        if (i < status.size() && status[i]) {
            Pts1.push_back(_pts1[i]);
            Pts2.push_back(_pts2[i]);
            good_matches.push_back(matches[i]);
        }
    }
    // good_matches = matches;

    ROS_INFO("Total %ld cost %fms Find Essential cost %fms", Pts1.size(), tic.toc(), tic0.toc());
    
    if (show) {
        cv::Mat img1_cpu, img2_cpu, _show;
        // img1.download(_img1);
        // img2.download(_img2);
        cv::drawMatches(_img2, kps2, _img1, kps1, good_matches, _show);
        // cv::resize(_show, _show, cv::Size(), VISUALIZE_SCALE, VISUALIZE_SCALE);
        cv::imshow("KNNMatch", _show);
        cv::waitKey(2);
    }
}
