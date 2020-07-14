/*
# Copyright (c) 2014-2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifdef WITH_VWORKS

#include "vworks_feature_tracker.hpp"

#include <climits>
#include <cfloat>
#include <iostream>
#include <iomanip>

#include <VX/vxu.h>
#include <NVX/nvx.h>

#include "ovx_replaceheader.hpp"

//
// The feature_tracker.cpp contains the implementation of the  virtual void
// functions: track() and init()
//

namespace
{
    //
    // FeatureTracker based on Harris/Fast Track + Optical Flow PyrLK
    //

    class FeatureTrackerImpl : public nvx::FeatureTracker
    {
    public:
        FeatureTrackerImpl(vx_context context, const Params& params);
        ~FeatureTrackerImpl();

        void init(vx_image firstFrame, vx_image mask);
        void track(vx_image newFrame, vx_image mask, bool lr_mode = false);

        vx_array getPrevFeatures() const;
        vx_array getCurrFeatures() const;

        void printPerfs() const;

    private:
        void createDataObjects();

        void processFirstFrame(vx_image frame, vx_image mask);
        void createMainGraph(vx_image frame, vx_image mask);

        void release();

        Params params_;

        vx_context context_;

        // Format for current frames
        vx_df_image format_;
        vx_uint32 width_;
        vx_uint32 height_;

        // Pyramids for two successive frames
        vx_delay pyr_delay_;

        // Points to track for two successive frames
        vx_delay pts_delay_;

        // Tracked points
        vx_array kp_curr_list_;

        // Main graph
        vx_graph main_graph_;

        // Node from main graph (used to print performance results)
        vx_node cvt_color_node_;
        vx_node pyr_node_;
        vx_node opt_flow_node_;
        vx_node feature_track_node_;

        bool use_rgb;
    };

    FeatureTrackerImpl::FeatureTrackerImpl(vx_context context, const Params& params) :
        params_(params)
    {
        context_ = context;

        format_ = VX_DF_IMAGE_VIRT;
        width_ = 0;
        height_ = 0;

        pyr_delay_ = nullptr;
        pts_delay_ = nullptr;
        kp_curr_list_ = nullptr;

        main_graph_ = nullptr;
        cvt_color_node_ = nullptr;
        pyr_node_ = nullptr;
        opt_flow_node_ = nullptr;
        feature_track_node_ = nullptr;
    }

    FeatureTrackerImpl::~FeatureTrackerImpl()
    {
        release();
    }

    void FeatureTrackerImpl::init(vx_image firstFrame, vx_image mask)
    {
        // Check input format

        vx_df_image format = VX_DF_IMAGE_VIRT;
        vx_uint32 width = 0;
        vx_uint32 height = 0;

        NVXIO_SAFE_CALL( vxQueryImage(firstFrame, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)) );
        NVXIO_SAFE_CALL( vxQueryImage(firstFrame, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)) );
        NVXIO_SAFE_CALL( vxQueryImage(firstFrame, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)) );

        if(params_.use_rgb) {
            NVXIO_ASSERT(format == VX_DF_IMAGE_RGB);
        } else {
            NVXIO_ASSERT(format == VX_DF_IMAGE_U8);
        }

        if (mask)
        {
            vx_df_image mask_format = VX_DF_IMAGE_VIRT;
            vx_uint32 mask_width = 0;
            vx_uint32 mask_height = 0;

            NVXIO_SAFE_CALL( vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_FORMAT, &mask_format, sizeof(mask_format)) );
            NVXIO_SAFE_CALL( vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_WIDTH, &mask_width, sizeof(mask_width)) );
            NVXIO_SAFE_CALL( vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_HEIGHT, &mask_height, sizeof(mask_height)) );

            NVXIO_ASSERT(mask_format == VX_DF_IMAGE_U8);
            NVXIO_ASSERT(mask_width == width);
            NVXIO_ASSERT(mask_height == height);
        }

        // Re-create graph if the input size was changed

        if (width != width_ || height != height_)
        {
            release();

            format_ = format;
            width_ = width;
            height_ = height;

            createDataObjects();

            createMainGraph(firstFrame, mask);
        }

        // Process first frame

        processFirstFrame(firstFrame, mask);
    }

    //
    // For the subsequent frames, we call FeatureTracker::track() which
    // essentially updates the input parameters passed to the graph. The previous
    // pyramid and the tracked points in the previous frame are set by the
    // vxAgeDelay(). The current Frame and the current mask are set by
    // vxSetParameterByIndex. Finally vxProcessGraph() is called to execute the graph
    //

    void FeatureTrackerImpl::track(vx_image newFrame, vx_image mask, bool lr_mode)
    {
        // Check input format

        vx_df_image format = VX_DF_IMAGE_VIRT;
        vx_uint32 width = 0;
        vx_uint32 height = 0;

        NVXIO_SAFE_CALL( vxQueryImage(newFrame, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)) );
        NVXIO_SAFE_CALL( vxQueryImage(newFrame, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)) );
        NVXIO_SAFE_CALL( vxQueryImage(newFrame, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)) );

        NVXIO_ASSERT(format == format_);
        NVXIO_ASSERT(width == width_);
        NVXIO_ASSERT(height == height_);

        if (mask)
        {
            vx_df_image mask_format = VX_DF_IMAGE_VIRT;
            vx_uint32 mask_width = 0;
            vx_uint32 mask_height = 0;

            NVXIO_SAFE_CALL( vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_FORMAT, &mask_format, sizeof(mask_format)) );
            NVXIO_SAFE_CALL( vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_WIDTH, &mask_width, sizeof(mask_width)) );
            NVXIO_SAFE_CALL( vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_HEIGHT, &mask_height, sizeof(mask_height)) );

            NVXIO_ASSERT(mask_format == VX_DF_IMAGE_U8);
            NVXIO_ASSERT(mask_width == width_);
            NVXIO_ASSERT(mask_height == height_);
        }

        // Update input parameters for next graph execution
        if (params_.use_rgb) {
            NVXIO_SAFE_CALL( vxSetParameterByIndex(cvt_color_node_, 0, (vx_reference)newFrame) );
        } else {
            NVXIO_SAFE_CALL( vxSetParameterByIndex(pyr_node_, 0, (vx_reference)newFrame) );
        }
        NVXIO_SAFE_CALL( vxSetParameterByIndex(feature_track_node_, 2, (vx_reference)mask) );

        // Age the delay objects (pyramid, points to track) before graph execution
        NVXIO_SAFE_CALL( vxAgeDelay(pyr_delay_) );
        NVXIO_SAFE_CALL( vxAgeDelay(pts_delay_) );

        // Process graph
        NVXIO_SAFE_CALL( vxProcessGraph(main_graph_) );
    }

    vx_array FeatureTrackerImpl::getPrevFeatures() const
    {
        return (vx_array)vxGetReferenceFromDelay(pts_delay_, -1);
    }

    vx_array FeatureTrackerImpl::getCurrFeatures() const
    {
        return kp_curr_list_;
    }

    void FeatureTrackerImpl::printPerfs() const
    {
        vx_size num_items, num_items_new = 0;
        NVXIO_SAFE_CALL( vxQueryArray((vx_array)vxGetReferenceFromDelay(pts_delay_, -1), VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_items, sizeof(num_items)) );
        NVXIO_SAFE_CALL( vxQueryArray(kp_curr_list_, VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_items_new, sizeof(num_items_new)) );
#ifdef __ANDROID__
        NVXIO_LOGI("FeatureTracker", "Found " VX_FMT_SIZE " Features", num_items);
#else
        std::cout << "Found old" << num_items << " Features New"<< num_items_new << std::endl;
#endif

#ifdef OVX
        ovxio::printPerf(main_graph_, "Feature Tracker");
        if (params_.use_rgb) {
            ovxio::printPerf(cvt_color_node_, "Color Convert");
        }

        ovxio::printPerf(pyr_node_, "Pyramid");
        ovxio::printPerf(feature_track_node_, "Feature Track");
        ovxio::printPerf(opt_flow_node_, "Optical Flow");
#endif
    }

    void FeatureTrackerImpl::release()
    {
        format_ = VX_DF_IMAGE_VIRT;
        width_ = 0;
        height_ = 0;

        vxReleaseDelay(&pyr_delay_);
        vxReleaseDelay(&pts_delay_);
        vxReleaseArray(&kp_curr_list_);

        vxReleaseNode(&cvt_color_node_);
        vxReleaseNode(&pyr_node_);
        vxReleaseNode(&opt_flow_node_);
        vxReleaseNode(&feature_track_node_);

        vxReleaseGraph(&main_graph_);
    }

    //
    // CreateDataObjects creates data objects that are not entirely linked to
    // graphs. It creates two vx_delay references: pyr_delay_ and pts_delay_.
    // pyr_delay_ holds image pyramids of two successive frames and pts_delay_
    // holds the tracked points from the previous frame that will be used as an
    // input to the pipeline, which is constructed by the createMainGraph()
    //

    void FeatureTrackerImpl::createDataObjects()
    {
        //
        // Image pyramids for two successive frames are necessary for the computation.
        // A delay object with 2 slots is created for this purpose
        //

        vx_pyramid pyr_exemplar = vxCreatePyramid(context_, params_.pyr_levels, VX_SCALE_PYRAMID_HALF, width_, height_, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(pyr_exemplar);
        pyr_delay_ = vxCreateDelay(context_, (vx_reference)pyr_exemplar, 2);
        NVXIO_CHECK_REFERENCE(pyr_delay_);
        vxReleasePyramid(&pyr_exemplar);

        //
        // Input points to track need to kept for two successive frames.
        // A delay object with two slots is created for this purpose
        //

        vx_array pts_exemplar = vxCreateArray(context_, NVX_TYPE_KEYPOINTF, params_.array_capacity);
        NVXIO_CHECK_REFERENCE(pts_exemplar);
        pts_delay_ = vxCreateDelay(context_, (vx_reference)pts_exemplar, 2);
        NVXIO_CHECK_REFERENCE(pts_delay_);
        vxReleaseArray(&pts_exemplar);

        //
        // Create the list of tracked points. This is the output of the frame processing
        //

        kp_curr_list_ = vxCreateArray(context_, NVX_TYPE_KEYPOINTF, params_.array_capacity);
        NVXIO_CHECK_REFERENCE(kp_curr_list_);
    }

    //
    // The processFirstFrame() converts the first frame into grayscale,
    // builds initial Gaussian pyramid, and detects initial keypoints.
    //

    void FeatureTrackerImpl::processFirstFrame(vx_image frame, vx_image mask)
    {
        vx_image frameGray = vxCreateImage(context_, width_, height_, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(frameGray);
        
        if(params_.use_rgb) {
            NVXIO_SAFE_CALL( vxuColorConvert(context_, frame, frameGray) );
            NVXIO_SAFE_CALL( vxuGaussianPyramid(context_, frameGray, (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0)) );

            if (params_.use_harris_detector)
            {
                NVXIO_SAFE_CALL( nvxuHarrisTrack(context_, frameGray,
                                                (vx_array)vxGetReferenceFromDelay(pts_delay_, 0),
                                                mask, nullptr,
                                                params_.harris_k, params_.harris_thresh,
                                                params_.detector_cell_size, nullptr) );
            }
            else
            {
                NVXIO_SAFE_CALL( nvxuFastTrack(context_, frameGray,
                                            (vx_array)vxGetReferenceFromDelay(pts_delay_, 0),
                                            mask, nullptr,
                                            params_.fast_type, params_.fast_thresh,
                                            params_.detector_cell_size, nullptr) );
            }

            vxReleaseImage(&frameGray);
        } else {
            NVXIO_SAFE_CALL( vxuGaussianPyramid(context_, frame, (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0)) );

            if (params_.use_harris_detector)
            {
                NVXIO_SAFE_CALL( nvxuHarrisTrack(context_, frame,
                                                (vx_array)vxGetReferenceFromDelay(pts_delay_, 0),
                                                mask, nullptr,
                                                params_.harris_k, params_.harris_thresh,
                                                params_.detector_cell_size, nullptr) );
            }
            else
            {
                NVXIO_SAFE_CALL( nvxuFastTrack(context_, frame,
                                            (vx_array)vxGetReferenceFromDelay(pts_delay_, 0),
                                            mask, nullptr,
                                            params_.fast_type, params_.fast_thresh,
                                            params_.detector_cell_size, nullptr) );
            }
        }

        
    }

    //
    // The createMainGraph() creates the pipeline. frame is passed as an input
    // argument to createMainGraph(). It is subsequently overwritten in the function
    // track() via the vxSetParameterByIndex()
    //

    void FeatureTrackerImpl::createMainGraph(vx_image frame, vx_image mask)
    {
        main_graph_ = vxCreateGraph(context_);
        NVXIO_CHECK_REFERENCE(main_graph_);

        //
        // Intermediate images. Both images are created as virtual in order to
        // inform the OpenVX framework that the application will never access their
        // content
        //
        vx_pyramid pyrGray = (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0);
        NVXIO_CHECK_REFERENCE(pyrGray);
        vx_image frameGray = vxGetPyramidLevel(pyrGray, 0);
        NVXIO_CHECK_REFERENCE(frameGray);

        //
        // Lucas-Kanade optical flow node
        // Note: keypoints of the previous frame are also given as 'new points
        // estimates'
        //

        vx_float32 lk_epsilon = 0.01f;
        vx_scalar s_lk_epsilon = vxCreateScalar(context_, VX_TYPE_FLOAT32, &lk_epsilon);
        NVXIO_CHECK_REFERENCE(s_lk_epsilon);

        vx_scalar s_lk_num_iters = vxCreateScalar(context_, VX_TYPE_UINT32, &params_.lk_num_iters);
        NVXIO_CHECK_REFERENCE(s_lk_num_iters);

        vx_bool lk_use_init_est = vx_false_e;
        vx_scalar s_lk_use_init_est = vxCreateScalar(context_, VX_TYPE_BOOL, &lk_use_init_est);
        NVXIO_CHECK_REFERENCE(s_lk_use_init_est);

        //
        // RGB to Y conversion nodes
        //

        if (params_.use_rgb) {
            cvt_color_node_ = vxColorConvertNode(main_graph_, frame, frameGray);
            NVXIO_CHECK_REFERENCE(cvt_color_node_);

            //
            // Pyramid image node
            //

            pyr_node_ = vxGaussianPyramidNode(main_graph_, frameGray, pyrGray);
            NVXIO_CHECK_REFERENCE(pyr_node_);
        } else {
            pyr_node_ = vxGaussianPyramidNode(main_graph_, frame, pyrGray);
            NVXIO_CHECK_REFERENCE(pyr_node_);
        }
        //
        // vxOpticalFlowPyrLKNode accepts input arguements as current pyramid,
        // previous pyramid and points tracked in the previous frame. The output
        // is the set of points tracked in the current frame
        //
        //pts_delay(-1) and kp_curr_list is correspondin
        opt_flow_node_ = vxOpticalFlowPyrLKNode(main_graph_,
                    (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, -1), // previous pyramid
                    (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0),  // current pyramid
                    (vx_array)vxGetReferenceFromDelay(pts_delay_, -1),   // points to track from previous frame
                    (vx_array)vxGetReferenceFromDelay(pts_delay_, -1),
                    kp_curr_list_,                                       // points tracked in current frame
                    VX_TERM_CRITERIA_BOTH,
                    s_lk_epsilon,
                    s_lk_num_iters,
                    s_lk_use_init_est,
                    params_.lk_win_size);
        NVXIO_CHECK_REFERENCE(opt_flow_node_);

        // Corner track node
        if (params_.use_harris_detector)
        {
            feature_track_node_ = nvxHarrisTrackNode(main_graph_, frameGray,
                                                     (vx_array)vxGetReferenceFromDelay(pts_delay_, 0),
                                                     mask, kp_curr_list_,
                                                     params_.harris_k, params_.harris_thresh,
                                                     params_.detector_cell_size, nullptr);
        }
        else
        {
            feature_track_node_ = nvxFastTrackNode(main_graph_, frameGray,
                                                   (vx_array)vxGetReferenceFromDelay(pts_delay_, 0),
                                                   mask, kp_curr_list_,
                                                   params_.fast_type, params_.fast_thresh,
                                                   params_.detector_cell_size, nullptr);
        }
        NVXIO_CHECK_REFERENCE(feature_track_node_);

        // Ensure highest graph optimization level
        const char* option = "-O3";
        NVXIO_SAFE_CALL( vxSetGraphAttribute(main_graph_, NVX_GRAPH_VERIFY_OPTIONS, option, strlen(option)) );

        //
        // Graph verification
        // Note: This verification is mandatory prior to graph execution
        //

        NVXIO_SAFE_CALL( vxVerifyGraph(main_graph_) );

        vxReleaseScalar(&s_lk_epsilon);
        vxReleaseScalar(&s_lk_num_iters);
        vxReleaseScalar(&s_lk_use_init_est);
        vxReleaseImage(&frameGray);
    }
}

nvx::FeatureTracker::Params::Params()
{
    // Parameters for optical flow node
    pyr_levels = 6;
    lk_num_iters = 5;
    lk_win_size = 10;

    // Common parameters for corner detector node
    array_capacity = 2000;
    detector_cell_size = 18;
    use_harris_detector = true;

    // Parameters for harris_track node
    harris_k = 0.04f;
    harris_thresh = 100.0f;

    // Parameters for fast_track node
    fast_type = 9;
    fast_thresh = 25;

    use_rgb = false;
}

nvx::FeatureTracker* nvx::FeatureTracker::create(vx_context context, const Params& params)
{
    return new FeatureTrackerImpl(context, params);
}
#endif