#include <camera_model/code_utils/cv_utils.h>

cv_utils::fisheye::PreProcess::PreProcess( const cv::Size _raw_image_size,
                                           const cv::Size _roi_size,
                                           const cv::Point _center,
                                           const float _resize_scale )
: is_preprocess( false )
, is_resize_only( false )
{
    /* clang-format off */
    if (    _raw_image_size.width >= _roi_size.width
            && _raw_image_size.height >= _roi_size.height
            && _center.x <= _raw_image_size.width
            && _center.y <= _raw_image_size.height
            && _roi_size.width != 0
            && _roi_size.height != 0 )
      is_preprocess = true;
    else
      is_preprocess = false;
    /* clang-format on */

    if ( is_preprocess                               //
         && _raw_image_size.width == _roi_size.width //
         && _raw_image_size.height == _roi_size.height )
    {
        is_resize_only = true;
        std::cout << "[#INFO] resize_only." << std::endl;
    }

    if ( is_preprocess )
        resetPreProcess( _roi_size, _center, _resize_scale );
    else
        std::cout << "[#ERROR] Parameters error." << std::endl;
}

void
cv_utils::fisheye::PreProcess::resetPreProcess( cv::Size _roi_size, cv::Point _center, float _resize_scale )
{
    if ( _resize_scale < 0 )
        resize_scale = 1.0;
    else
        resize_scale = _resize_scale;

    roi_row_start = _center.y - _roi_size.height / 2;
    roi_row_end   = roi_row_start + _roi_size.height;
    roi_col_start = _center.x - _roi_size.width / 2;
    roi_col_end   = roi_col_start + _roi_size.width;

    std::cout << "[#INFO] ROI row: start " << roi_row_start << " ,end " << roi_row_end << std::endl;
    std::cout << "[#INFO] ROI col: start " << roi_col_start << " ,end " << roi_col_end << std::endl;
}

cv::Mat
cv_utils::fisheye::PreProcess::do_preprocess( cv::Mat image_input )
{
    if ( is_resize_only )
    {
        cv::Mat image_input_resized;
        cv::resize( image_input,
                    image_input_resized,
                    cv::Size( image_input.cols * resize_scale, //
                              image_input.rows * resize_scale ) );
        return image_input_resized;
    }
    else if ( is_preprocess )
    {
        //        std::cout << " is_preprocess true " << std::endl;
        cv::Mat image_input_roi = image_input( cv::Range( roi_row_start, roi_row_end ),
                                               cv::Range( roi_col_start, roi_col_end ) );

        cv::Mat image_input_resized;
        cv::resize( image_input_roi,
                    image_input_resized,
                    cv::Size( image_input_roi.cols * resize_scale, //
                              image_input_roi.rows * resize_scale ) );
        return image_input_resized;
    }
    else
    {
        //        std::cout << " is_preprocess false " << std::endl;
        return image_input;
    }
}
