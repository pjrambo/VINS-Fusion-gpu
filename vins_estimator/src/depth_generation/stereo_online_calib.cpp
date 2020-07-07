#include "stereo_online_calib.hpp"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

Eigen::Vector3d undist(const cv::Point2f & pt, const cv::Mat & cameraMatrix) {
    double x = (pt.x - cameraMatrix.at<double>(0, 2))/ cameraMatrix.at<double>(0, 0);
    double y = (pt.y - cameraMatrix.at<double>(1, 2))/ cameraMatrix.at<double>(1, 1);
    return Eigen::Vector3d(x, y, 1);
}

std::vector<cv::DMatch> filter_by_hamming(const std::vector<cv::DMatch> & matches);
std::vector<cv::DMatch> filter_by_E(const std::vector<cv::DMatch> & matches,     
    std::vector<cv::KeyPoint> query_pts, 
    std::vector<cv::KeyPoint> train_pts, 
    cv::Mat cameraMatrix, Eigen::Matrix3d E);

// using namespace ceres;
struct StereoCostFunctor {

    template <typename T>
    static void EssentialMatfromRT(const T * R, T x, T y, T z, T * E) {
        E[0] = R[6]*y-R[3]*z;
        E[1] = R[7]*y-R[4]*z;
        E[2] = R[8]*y-R[5]*z;

        E[3] = -R[6]*x+R[0]*z;
        E[4] = -R[7]*x+R[1]*z;
        E[5] = -R[8]*x+R[2]*z;

        E[6] = R[3]*x - R[0]*y;
        E[7] = R[4]*x - R[1]*y;
        E[8] = R[5]*x - R[2]*y;
    }

    //We assue the R is near identity and T is near -1, 0, 0
    //We use roll pitch yaw to represent rotation;
    //-1 * [cos(theta)cos(phi), cos(theta)* sin(phi), sin(theta)]
    //X is [roll, pitch, yaw, theta, phi]
    template <typename T>
    bool operator()(T const *const * _x, T* residual) const {
        // residual[0] = T(10.0) - x[0];
        T R[9];
        ceres::EulerAnglesToRotationMatrix(_x[0], 3, R);
        // std::cerr << "R" << R << std::endl;
        
        T tx = T(-1);
        T ty = _x[0][3];
        T tz = _x[0][4];

        T E[9];
        EssentialMatfromRT(R, tx, ty, tz, E);

        Eigen::Matrix<T, 3, 3, Eigen::RowMajor> E_eig = Eigen::Map <Eigen::Matrix <T, 3, 3, Eigen::RowMajor>> (E);
        // std::cerr << E_eig << std::endl;

        for (int i = 0; i < left_pts.size(); i++) {
            Eigen::Matrix<T, 1, 1, Eigen::RowMajor> ret = right_pts[i].transpose()*E_eig*left_pts[i]*100.0;
            residual[i] = ret(0, 0);
        }

        return true;
    }


    template <typename T>
    bool Evalute(T const * _x) const {
        // residual[0] = T(10.0) - x[0];
        T R[9];
        ceres::EulerAnglesToRotationMatrix(_x, 3, R);
        T tx = -1;
        T ty = _x[3];
        T tz = _x[4];
        
        Eigen::Matrix<T, 3, 3, Eigen::RowMajor> R_eig = Eigen::Map <Eigen::Matrix <T, 3, 3, Eigen::RowMajor>> (R);

        Eigen::Matrix<T, 3, 3, Eigen::RowMajor> E_eig;

        Eigen::Matrix<T, 3, 3, Eigen::RowMajor> Tcross;
        Tcross <<   0, -tz, ty, 
                    tz, 0, -tx,
                    -ty, tx, 0;

        E_eig = Tcross*R_eig;
        // std::cerr << "t" << tx  << ":" << ty <<":" << tz << std::endl;
        // std::cerr << "Reig" << R_eig << std::endl;
        // std::cerr << "Eeig" << E_eig << std::endl;

        T E[9];
        EssentialMatfromRT(R, tx, ty, tz, E);
        
        E_eig = Eigen::Map <Eigen::Matrix <T, 3, 3, Eigen::RowMajor>> (E);
        // std::cerr <<  "E ceres" << E_eig << std::endl;

        for (int i = 0; i < left_pts.size(); i++) {
            Eigen::Matrix<T, 1, 1, Eigen::RowMajor> ret = right_pts[i].transpose()*E_eig*left_pts[i];
        }

        return true;
    }

    std::vector<Eigen::Vector3d> left_pts, right_pts;
    StereoCostFunctor(const std::vector<cv::Point2f> & _left_pts, 
        const std::vector<cv::Point2f> & _right_pts, cv::Mat cameraMatrix)
    {

        assert(_left_pts.size() == _right_pts.size() && _left_pts.size() > 0 && "Solver need equal LR pts more than zero");
        for (size_t i = 0; i < _left_pts.size(); i++) {
            
            Eigen::Vector3d pt_l(undist(_left_pts[i], cameraMatrix));
            Eigen::Vector3d pt_r(undist(_right_pts[i], cameraMatrix));

            left_pts.push_back(pt_l);
            right_pts.push_back(pt_r);
        }
    }
};

Eigen::Vector3d thetaphi2xyz(double theta, double phi) {
    double tx = - cos(theta)*cos(phi);
    double ty = - cos(theta)*sin(phi);
    double tz = - sin(theta);
    return Eigen::Vector3d(tx, ty, tz);
}

std::pair<double, double> xyz2thetaphi(Eigen::Vector3d xyz) {
    xyz = - xyz.normalized();
    // double tx = cos(theta)*cos(phi);
    // double ty = cos(theta)*sin(phi);
    // double tz = sin(theta);
    double theta = asin(xyz.z());
    double phi = atan2(xyz.y(), xyz.x());

    return make_pair(theta, phi);
}


bool StereoOnlineCalib::calibrate_extrinsic_optimize(const std::vector<cv::Point2f> & left_pts, 
    const std::vector<cv::Point2f> & right_pts) {
    
    auto stereo_func = new StereoCostFunctor(left_pts, right_pts, cameraMatrix);
    auto cost_function = new ceres::DynamicAutoDiffCostFunction<StereoCostFunctor, 7>(stereo_func);
    cost_function->AddParameterBlock(5);
    cost_function->SetNumResiduals(left_pts.size());

    std::cerr << "solve extrinsic with " << left_pts.size() << "Pts" << std::endl;
    std::cerr << "Initial R" << R_eig << std::endl;
    std::cerr << "Initial T" << T_eig << std::endl;

    auto t_init = T_eig / -T_eig.x();


    Eigen::Vector3d ypr = Utility::R2ypr(R_eig, true);
    std::cerr << "T init non scale" << t_init << std::endl;
    ROS_WARN("\nInital RPY %f %f %f deg", ypr.x(), ypr.y(), ypr.z());

    std::vector<double> x;
    //x is roll pitch yaw
    x.push_back(ypr.z());
    x.push_back(ypr.y());
    x.push_back(ypr.x());

    x.push_back(t_init.y());
    x.push_back(t_init.z());
    
    stereo_func->Evalute<double>(x.data());
    ceres::Problem problem;
    problem.AddResidualBlock(cost_function, NULL, x.data());
    ceres::Solver::Options options;

    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cerr << summary.BriefReport() << " time " << summary.minimizer_time_in_seconds*1000 << "ms\n";

    Eigen::Vector3d _t_eig(-1, x[3], x[4]);
    _t_eig = _t_eig * baseline;
    auto _R_eig = Utility::ypr2R( Eigen::Vector3d(x[2], x[1], x[0]));
    cv::Mat _R;
    cv::Mat _T;

    cv::eigen2cv(_R_eig, _R);
    cv::eigen2cv(_t_eig, _T);

    ceres::Covariance::Options covoptions;
    ceres::Covariance covariance(covoptions);

    vector<pair<const double*, const double*> > covariance_blocks;
    covariance_blocks.push_back(make_pair(x.data(), x.data()));
    CHECK(covariance.Compute(covariance_blocks, &problem));
    Eigen::Matrix<double, 5, 5> cov;
    covariance.GetCovarianceBlock(x.data(), x.data(), cov.data());


    std::cerr << "cov\n" << cov << "Max" << cov.maxCoeff() << std::endl;

    ROS_WARN("Solved Y %f P %f R %f", x[2], x[1], x[0]);
    std::cerr << _R << std::endl;
    ROS_WARN("Solved T %f %f %f", _t_eig.x(), _t_eig.y(), _t_eig.z());

    if (cov.maxCoeff() < MAX_ACCEPT_COV) {
        ROS_WARN("Update R T");
        std::cerr << _R << std::endl;
        std::cerr << _T << std::endl;
        update(_R, _T);
    }

    return true;
}

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

    ROS_WARN("Find EssentialMat with %ld/%d pts use %fms", left_pts.size(), status_count, tic2.toc());
    cv::Mat _R, t;
    int num = cv::recoverPose(essentialMat, left_pts, right_pts, cameraMatrix, _R, t);
    ROS_WARN("Recorver pose take with %ld/%d pts use %fms", left_good.size(), num, tic2.toc());

    double disR = norm(R0 - _R);
    double disT = norm(t - T0/baseline);

    // std::cerr << "R" << _R << "DIS R" << disR << std::endl;
    // std::cerr << "T" << t*scale << "DIS T" << disT << std::endl;

    if (disR < GOOD_R_THRES && disT < GOOD_T_THRES) {
        update(_R, t*baseline);
        ROS_WARN("Update R T");
        std::cerr << "R" << R << std::endl;
        std::cerr << "T" << T << std::endl;
        std::cerr << "t" << t << std::endl;
        std::cerr << "E" << essentialMat << std::endl;
        
        // return calibrate_extrinsic_optimize(left_pts, right_pts);
        return true;
    }

    return false;
}

typedef std::pair<cv::Point2f, cv::Point2f> point_pair;
typedef std::vector<point_pair> matched_points;

bool compareDisparity(point_pair p1, point_pair p2) 
{ 
    return cv::norm(p1.first - p1.second) > cv::norm(p2.first - p2.second);
}

void StereoOnlineCalib::filter_points_by_region(std::vector<cv::Point2f> & good_left, std::vector<cv::Point2f> & good_right) {
    std::map<int, matched_points> regions;
    for (int i = 0; i < left_pts.size(); i ++) {
        float x =  left_pts[i].x;
        float y =  left_pts[i].y;
        // printf("%f %f\n", width, height);
        int _reg_id = (int)(x*CALIBCOLS/width)*1000 + (int)(y*CALIBROWS/height);
        if (regions.find(_reg_id) == regions.end()) {
            regions[_reg_id] = matched_points();
        }
        // printf("Reg_id %d\n", _reg_id);
        regions[_reg_id].push_back(make_pair(left_pts[i], right_pts[i]));
    }

    
    for (auto reg : regions) {
        int count = 0;
        auto & pts_stack = reg.second;
        std::random_shuffle ( pts_stack.begin(), pts_stack.end() );

        // std::sort(pts_stack.begin(), pts_stack.end(), compareDisparity);
        for (auto it: pts_stack) {
            good_left.push_back(it.first);
            good_right.push_back(it.second);
            count ++;
            if (count > PTS_NUM_REG) {
                break;
            }
        }
    }

}

#ifdef USE_CUDA
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

    std::vector<cv::Point2f> good_left;
    std::vector<cv::Point2f> good_right;
    
    filter_points_by_region(good_left, good_right);
    cv::Mat _show;
    left.download(_show);
    cv::cvtColor(_show, _show, cv::COLOR_GRAY2BGR);

    std::map<int, cv::Scalar> region_colors;
    

    /*    
    for (int i = 0; i < good_left.size(); i ++) {
        float x = good_left[i].x;
        float y = good_right[i].y;
        int _reg_id = (int)(x*CALIBCOLS/width)*1000 + (int)(y*CALIBROWS/height);
        if (region_colors.find(_reg_id) == region_colors.end()) {
            region_colors[_reg_id] = cv::Scalar(rand()%255, rand()%255, rand()%255);
        }
        cv::circle(_show, good_left[i], 1, region_colors[_reg_id], 2);
        cv::arrowedLine(_show, good_left[i], good_right[i], cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    }
    cv::imshow("Regions", _show);*/

    return calibrate_extrinsic_optimize(good_left, good_right);
    // return calibrate_extrinsic_opencv(left_pts, right_pts);
}

void StereoOnlineCalib::find_corresponding_pts(cv::cuda::GpuMat & img1, cv::cuda::GpuMat & img2, 
    std::vector<cv::Point2f> & Pts1, std::vector<cv::Point2f> & Pts2) {
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

    if (desc1.empty() || desc2.empty()) {
        return;
    }
    size_t j = 0;

    cv::BFMatcher bfmatcher(cv::NORM_HAMMING2, true);
    std::vector<cv::DMatch> matches;
    bfmatcher.match(desc2, desc1, matches);
    matches = filter_by_hamming(matches);

    double thres = 0.05;
    
    // matches = filter_by_x(matches, kps2, kps1, thres);
    // matches = filter_by_y(matches, kps2, kps1, thres);

    matches = filter_by_E(matches, kps2, kps1, cameraMatrix, E_eig);

    vector<cv::Point2f> _pts1, _pts2;
    vector<uchar> status;
    for (auto gm : matches) {
        auto _id1 = gm.trainIdx;
        auto _id2 = gm.queryIdx;
        _pts1.push_back(kps1[_id1].pt);
        _pts2.push_back(kps2[_id2].pt);
    }

    ROS_INFO("BRIEF MATCH cost %fms", tic.toc());


    for(int i = 0; i < _pts1.size(); i ++) {
        Pts1.push_back(_pts1[i]);
        Pts2.push_back(_pts2[i]);
        good_matches.push_back(matches[i]);
    }

    // if (_pts1.size() > MINIUM_ESSENTIALMAT_SIZE) {
    //     cv::findEssentialMat(_pts1, _pts2, cameraMatrix, cv::RANSAC, 0.999, 1.0, status);
    // }

    // for(int i = 0; i < _pts1.size(); i ++) {
    //     if (i < status.size() && status[i]) {
    //         Pts1.push_back(_pts1[i]);
    //         Pts2.push_back(_pts2[i]);
    //         good_matches.push_back(matches[i]);
    //     }
    // }
    // good_matches = matches;

    // ROS_INFO("Total %ld cost %fms Find Essential cost %fms", Pts1.size(), tic.toc(), tic0.toc());
    
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
#endif

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

        auto cost = f2.transpose()*E*f1;
        if (cost.norm() < MAX_ESSENTIAL_OUTLIER_COST) {
            good_matches.push_back(gm);
        }
    }
    return good_matches;
}

