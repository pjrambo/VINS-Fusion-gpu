#ifndef CV_UTLIS_H
#define CV_UTLIS_H

#include <opencv2/opencv.hpp>

namespace cv_utils
{
namespace fisheye
{
class PreProcess
{
    public:
    PreProcess( );
    PreProcess( const cv::Size _raw_image_size,
                const cv::Size _roi_size,
                const cv::Point _center,
                const float _resize_scale );
    void resetPreProcess( cv::Size _roi_size, cv::Point _center, float _resize_scale );

    cv::Mat do_preprocess( cv::Mat image_input );

    float resize_scale;
    int roi_row_start;
    int roi_col_start;
    int roi_row_end;
    int roi_col_end;

    bool is_preprocess;
    bool is_resize_only;
};
}
}
#endif // CV_UTLIS_H
