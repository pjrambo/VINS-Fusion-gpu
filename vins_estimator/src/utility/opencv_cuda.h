#pragma once

#include <opencv2/opencv.hpp>

#ifndef USE_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <libsgm.h>
#include <opencv2/cudafeatures2d.hpp>
#else
namespace cv {
namespace cuda {
    typedef cv::Mat GpuMat;
};
};
#endif
