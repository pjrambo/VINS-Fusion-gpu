/*
# Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
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

#include "stereo_matching.hpp"

#include <climits>
#include <cfloat>
#include <iostream>
#include <iomanip>

#include <VX/vxu.h>
#include <NVX/nvx.h>

#include <OVX/UtilityOVX.hpp>

#ifdef __ANDROID__
#define LOG_TAG "SGBM"
#endif

//
// SGM-based stereo matching
//
// This file contains 3 implementations of the StereoMatching interface
// (declared in the file stereo_matching.hpp). They can be created by calling
// the static function StereoMatching::createStereoMatching() and providing it
// the corresponding value of StereoMatching::ImplementationType enum. The
// available implementations are:
//
// - HIGH_LEVEL_API: stereo is evaluated by a single nvxSemiGlobalMatchingNode
// - LOW_LEVEL_API: the stereo pipeline consists of 4 nodes:
//   - nvxComputeModifiedCostBTNode
//   - nvxConvolveCostNode
//   - nvxAggregateCostScanlinesNode
//   - nvxComputeDisparityNode
// - LOW_LEVEL_API_PYRAMIDAL: the same low-level nodes are used, but the
//   evaluation is organized in a "pyramidal" scheme to improve performance and
//   reduce memory footprint
//

namespace hlsgm
{
    //
    // This implementation uses an nvxSemiGlobalMatchingNode to evaluate
    // stereo. This is the most simple and straight-forward way to do this. For
    // some advanced processing, you can check out other implementations.
    //

    class SGBM : public StereoMatching
    {
    public:
        SGBM(vx_context context, const StereoMatchingParams& params,
             vx_image left, vx_image right, vx_image disparity);
        ~SGBM();

        virtual void run();

        void printPerfs() const;

    private:
        vx_graph main_graph_;
        vx_node left_cvt_color_node_;
        vx_node right_cvt_color_node_;
        vx_node semi_global_matching_node_;
        vx_node convert_depth_node_;
    };

    void SGBM::run()
    {
        NVXIO_SAFE_CALL( vxProcessGraph(main_graph_) );
    }

    SGBM::~SGBM()
    {
        vxReleaseGraph(&main_graph_);
    }

    SGBM::SGBM(vx_context context, const StereoMatchingParams& params,
               vx_image left, vx_image right, vx_image disparity)
        : main_graph_(nullptr)
    {
        vx_df_image format = VX_DF_IMAGE_VIRT;
        vx_uint32 width = 0;
        vx_uint32 height = 0;

        NVXIO_SAFE_CALL( vxQueryImage(left, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)) );
        NVXIO_SAFE_CALL( vxQueryImage(left, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)) );
        NVXIO_SAFE_CALL( vxQueryImage(left, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)) );

        main_graph_ = vxCreateGraph(context);
        NVXIO_CHECK_REFERENCE(main_graph_);

        // convert images to grayscale
        vx_image left_gray = vxCreateVirtualImage(main_graph_, width, height, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(left_gray);

        vx_image right_gray = vxCreateVirtualImage(main_graph_, width, height, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(right_gray);

        left_cvt_color_node_ = vxColorConvertNode(main_graph_, left, left_gray);
        NVXIO_CHECK_REFERENCE(left_cvt_color_node_);

        right_cvt_color_node_ = vxColorConvertNode(main_graph_, right, right_gray);
        NVXIO_CHECK_REFERENCE(right_cvt_color_node_);

        // evaluate stereo
        vx_image disparity_short = vxCreateVirtualImage(main_graph_, width, height, VX_DF_IMAGE_S16);
        NVXIO_CHECK_REFERENCE(disparity_short);

        //
        // The SGM algorithm is now added as a node to the graph via the
        // nvxSemiGlobalMatchingNode().The input to the SGM node is previosuly
        // constructed left_gray and right_gray vx_images and the configuration
        // parameters. The output of the SGM node is the disparity_short image
        // that holds S16 fixed-point disparity values. The fixed-point values
        // have Q11.4 format (one sign bit, eleven integer bits and four
        // fractional bits). For the ease of further processing, we convert the
        // disparity map from fixed-point representation to U8 disparity
        // image. To do this, we drop the 4 fractional bits by right-shifting
        // the S16 values and then simply scale down the bit-width precision via
        // the vxConvertDepthNode().
        //

        semi_global_matching_node_ = nvxSemiGlobalMatchingNode(
            main_graph_,
            left_gray,
            right_gray,
            disparity_short,
            params.min_disparity,
            params.max_disparity,
            params.P1,
            params.P2,
            params.sad,
            params.ct_win_size,
            params.hc_win_size,
            params.bt_clip_value,
            params.max_diff,
            params.uniqueness_ratio,
            params.scanlines_mask,
            params.flags);
        NVXIO_CHECK_REFERENCE(semi_global_matching_node_);

        // convert disparity from fixed point to grayscale
        vx_int32 shift = 4;
        vx_scalar s_shift = vxCreateScalar(context, VX_TYPE_INT32, &shift);
        NVXIO_CHECK_REFERENCE(s_shift);
        convert_depth_node_ = vxConvertDepthNode(main_graph_, disparity_short, disparity, VX_CONVERT_POLICY_SATURATE, s_shift);
        vxReleaseScalar(&s_shift);
        NVXIO_CHECK_REFERENCE(convert_depth_node_);

        // verify the graph
        NVXIO_SAFE_CALL( vxVerifyGraph(main_graph_) );

        // clean up
        vxReleaseImage(&left_gray);
        vxReleaseImage(&right_gray);

        vxReleaseImage(&disparity_short);
    }

    void SGBM::printPerfs() const
    {
        ovxio::printPerf(main_graph_, "Stereo");
        ovxio::printPerf(left_cvt_color_node_, "Left Color Convert");
        ovxio::printPerf(right_cvt_color_node_, "Right Color Convert");
        ovxio::printPerf(semi_global_matching_node_, "SGBM");
        ovxio::printPerf(convert_depth_node_, "Convert Depth");
    }
}

namespace llsgm
{
    //
    // This implementation uses 4 nodes:
    //   - nvxComputeModifiedCostBTNode
    //   - nvxConvolveCostNode
    //   - nvxAggregateCostScanlinesNode
    //   - nvxComputeDisparityNode
    // to build a very basic stereo pipeline.
    //
    // You can modify it, for example, to use different types of cost functions
    // or apply additional filters.

    class SGBM : public StereoMatching
    {
    public:
        SGBM(vx_context context, const StereoMatchingParams& params,
             vx_image left, vx_image right, vx_image disparity);
        ~SGBM();

        virtual void run();

        void printPerfs() const;

    private:
        vx_graph main_graph_;
        vx_node compute_cost_node_;
        vx_node convolve_cost_node_;
        vx_node aggregate_cost_scanlines_node_;
        vx_node compute_disparity_node_;
        vx_node left_cvt_color_node_;
        vx_node right_cvt_color_node_;
        vx_node convert_depth_node_;
        vx_node left_census_node_;
        vx_node right_census_node_;
    };

    void SGBM::run()
    {
        NVXIO_SAFE_CALL( vxProcessGraph(main_graph_) );
    }

    SGBM::~SGBM()
    {
        vxReleaseGraph(&main_graph_);
    }

    SGBM::SGBM(vx_context context, const StereoMatchingParams& params,
               vx_image left, vx_image right, vx_image disparity)
    {
        vx_df_image format = VX_DF_IMAGE_VIRT;
        vx_uint32 width = 0;
        vx_uint32 height = 0;
        vx_uint32 D = params.max_disparity - params.min_disparity;

        NVXIO_SAFE_CALL( vxQueryImage(left, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)) );
        NVXIO_SAFE_CALL( vxQueryImage(left, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)) );
        NVXIO_SAFE_CALL( vxQueryImage(left, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)) );

        main_graph_ = vxCreateGraph(context);
        NVXIO_CHECK_REFERENCE(main_graph_);

        vx_image left_gray = vxCreateVirtualImage(main_graph_, width, height, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(left_gray);

        vx_image right_gray = vxCreateVirtualImage(main_graph_, width, height, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(right_gray);

        left_cvt_color_node_ = vxColorConvertNode(main_graph_, left, left_gray);
        NVXIO_CHECK_REFERENCE(left_cvt_color_node_);

        right_cvt_color_node_ = vxColorConvertNode(main_graph_, right, right_gray);
        NVXIO_CHECK_REFERENCE(right_cvt_color_node_);

        vx_image left_census = NULL, right_census = NULL;

        // apply census transform, if requested
        if (params.ct_win_size > 1)
        {
            left_census =  vxCreateVirtualImage(main_graph_, width, height, VX_DF_IMAGE_U32);
            NVXIO_CHECK_REFERENCE(left_census);
            right_census = vxCreateVirtualImage(main_graph_, width, height, VX_DF_IMAGE_U32);
            NVXIO_CHECK_REFERENCE(right_census);

            left_census_node_ = nvxCensusTransformNode(main_graph_, left_gray, left_census, params.ct_win_size);
            NVXIO_CHECK_REFERENCE(left_census_node_);
            right_census_node_ = nvxCensusTransformNode(main_graph_, right_gray, right_census, params.ct_win_size);
            NVXIO_CHECK_REFERENCE(right_census_node_);
        }
        else
        {
            left_census_node_ = NULL;
            right_census_node_ = NULL;
        }

        vx_image convolved_cost = vxCreateVirtualImage(main_graph_, width * D, height, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(convolved_cost);

        // nvxConvolveCostNode can be seen as a simple "filtering" function for
        // the evaluated cost volume. By setting the `sad` parameter to 1 we can
        // omit it overall.
        vx_int32 sad = params.sad;
        if (sad > 1)
        {
            vx_image cost = vxCreateVirtualImage(main_graph_, width * D, height, VX_DF_IMAGE_U8);
            NVXIO_CHECK_REFERENCE(cost);

            // census transformed images should be compared by hamming cost
            if (params.ct_win_size > 1)
            {
                compute_cost_node_ = nvxComputeCostHammingNode(main_graph_, left_census, right_census, cost,
                                                               params.min_disparity, params.max_disparity,
                                                               params.hc_win_size);
            }
            else
            {
                compute_cost_node_ = nvxComputeModifiedCostBTNode(main_graph_, left_gray, right_gray, cost,
                                                                  params.min_disparity, params.max_disparity,
                                                                  params.bt_clip_value);
            }
            NVXIO_CHECK_REFERENCE(compute_cost_node_);

            convolve_cost_node_ = nvxConvolveCostNode(main_graph_, cost, convolved_cost,
                                                      D, sad);
            NVXIO_CHECK_REFERENCE(convolve_cost_node_);

            vxReleaseImage(&cost);
        }
        else
        {
            if (params.ct_win_size > 1)
            {
                compute_cost_node_ = nvxComputeCostHammingNode(main_graph_, left_census, right_census, convolved_cost,
                                                               params.min_disparity, params.max_disparity,
                                                               1);
            }
            else
            {
                compute_cost_node_ = nvxComputeModifiedCostBTNode(main_graph_, left_gray, right_gray, convolved_cost,
                                                                  params.min_disparity, params.max_disparity,
                                                                  params.bt_clip_value);
            }
            NVXIO_CHECK_REFERENCE(compute_cost_node_);

            convolve_cost_node_ = NULL;
        }

        vx_image aggregated_cost = vxCreateVirtualImage(main_graph_, width * D, height, VX_DF_IMAGE_S16);
        NVXIO_CHECK_REFERENCE(aggregated_cost);

        aggregate_cost_scanlines_node_ = nvxAggregateCostScanlinesNode(main_graph_, convolved_cost, aggregated_cost,
                                                                       D, params.P1, params.P2, params.scanlines_mask);
        NVXIO_CHECK_REFERENCE(aggregate_cost_scanlines_node_);

        vx_image disparity_short = vxCreateVirtualImage(main_graph_, width, height, VX_DF_IMAGE_S16);
        NVXIO_CHECK_REFERENCE(disparity_short);

        compute_disparity_node_ = nvxComputeDisparityNode(main_graph_, aggregated_cost, disparity_short,
                                                          params.min_disparity, params.max_disparity,
                                                          params.uniqueness_ratio, params.max_diff);
        NVXIO_CHECK_REFERENCE(compute_disparity_node_);

        vx_int32 shift = 4;
        vx_scalar s_shift = vxCreateScalar(context, VX_TYPE_INT32, &shift);
        NVXIO_CHECK_REFERENCE(s_shift);
        convert_depth_node_ = vxConvertDepthNode(main_graph_, disparity_short, disparity, VX_CONVERT_POLICY_SATURATE, s_shift);
        NVXIO_CHECK_REFERENCE(convert_depth_node_);

        vxReleaseScalar(&s_shift);
        vxReleaseImage(&disparity_short);
        vxReleaseImage(&aggregated_cost);
        vxReleaseImage(&convolved_cost);
        vxReleaseImage(&right_census);
        vxReleaseImage(&left_census);
        vxReleaseImage(&right_gray);
        vxReleaseImage(&left_gray);

        NVXIO_SAFE_CALL( vxVerifyGraph(main_graph_) );
    }

    void SGBM::printPerfs() const
    {
        ovxio::printPerf(main_graph_, "Stereo");
        ovxio::printPerf(left_cvt_color_node_, "Left Color Convert");
        ovxio::printPerf(right_cvt_color_node_, "Right Color Convert");
        if (left_census_node_) ovxio::printPerf(left_census_node_, "Left Census Transform");
        if (right_census_node_) ovxio::printPerf(right_census_node_, "Right Census Transform");
        ovxio::printPerf(compute_cost_node_, "Compute Cost");
        if (convolve_cost_node_) ovxio::printPerf(convolve_cost_node_, "Convolve Cost");
        ovxio::printPerf(aggregate_cost_scanlines_node_, "Aggregate Scanlines");
        ovxio::printPerf(compute_disparity_node_, "Compute Disparity");
        ovxio::printPerf(convert_depth_node_, "Convert Depth");
    }
}

namespace psgm
{
    //
    // This implementation uses the same 4 nodes as the low-level one, but the
    // evaluation is organized in a "pyramidal" scheme to improve performance
    // and reduce memory footprint. Using the pyramidal scheme instead of simple
    // downscaling of the input images makes it possible to do this without
    // drastic accuracy loss. The main idea behind this is to take smaller
    // disparity values from higher resolution images.
    //
    // The implementation uses 2 auxiliary nodes to propagate the disparity from
    // lower resolution levels:
    // - nvxPSGMCostPriorNode
    // - nvxPSGMDisparityMergeNode
    //

    const int pyr_levels = 3;
    const int D_divisors[pyr_levels] = { 4, 2, 1 };

    class SGBM : public StereoMatching
    {
    public:
        SGBM(vx_context context, const StereoMatchingParams& params,
             vx_image left, vx_image right, vx_image disparity);
        ~SGBM();

        virtual void run();

        void printPerfs() const;

    private:
        vx_graph main_graph_;
        vx_node left_cvt_color_node_;
        vx_node right_cvt_color_node_;

        vx_image disparity_short_[pyr_levels];
        vx_image aggregated_cost_[pyr_levels];
        vx_image cost_[pyr_levels];
        vx_image convolved_cost_[pyr_levels];

        vx_image full_left_gray_;
        vx_image full_right_gray_;

        vx_image full_aggregated_cost_;
        vx_image full_cost_;
        vx_image full_convolved_cost_;
    };

    void SGBM::run()
    {
        NVXIO_SAFE_CALL( vxProcessGraph(main_graph_) );
    }

    SGBM::~SGBM()
    {
        vxReleaseGraph(&main_graph_);
    }

    SGBM::SGBM(vx_context context, const StereoMatchingParams& params,
               vx_image left, vx_image right, vx_image disparity)
    {
        vx_df_image format = VX_DF_IMAGE_VIRT;
        vx_uint32 full_width = 0;
        vx_uint32 full_height = 0;
        vx_uint32 full_D = params.max_disparity - params.min_disparity;
        vx_int32 sad = params.sad;

        NVXIO_SAFE_CALL( vxQueryImage(left, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)) );
        NVXIO_SAFE_CALL( vxQueryImage(left, VX_IMAGE_ATTRIBUTE_WIDTH, &full_width, sizeof(full_width)) );
        NVXIO_SAFE_CALL( vxQueryImage(left, VX_IMAGE_ATTRIBUTE_HEIGHT, &full_height, sizeof(full_height)) );

        // We use nvxCreateStreamGraph instead of ordinary vxCreateGraph to
        // indicate that the execution order of nodes should be the same as the
        // creation order. This is necessary to allow for the following trick.
        //
        // For evaluation on the pyramid we need to have a corresponding copy of
        // each temporary buffer. Say, if we have 3 levels of the pyramid, we
        // need to have 3 cost volumes, 3 convolved costs and 3 aggregated
        // costs. But since we don't need them all at the same time, we can
        // allocate the memory only for the largest buffer and create smaller
        // buffers as ROIs on the same memory.
        //
        // Now, if we would do this and use an ordinary vxCreateGraph call,
        // VisionWorks framework could try to reorder the execution of nodes, so
        // that this scheme would no longer work. By using nvxCreateStreamGraph
        // we tell the framework to preserve the execution order.
        main_graph_ = nvxCreateStreamGraph(context);
        NVXIO_CHECK_REFERENCE(main_graph_);

        // allocate full buffers
        {
            full_left_gray_ = vxCreateVirtualImage(main_graph_, full_width, full_height, VX_DF_IMAGE_U8);
            NVXIO_CHECK_REFERENCE(full_left_gray_);

            full_right_gray_ = vxCreateVirtualImage(main_graph_, full_width, full_height, VX_DF_IMAGE_U8);
            NVXIO_CHECK_REFERENCE(full_right_gray_);

            full_convolved_cost_ = vxCreateVirtualImage(main_graph_, full_width * full_D / 4, full_height, VX_DF_IMAGE_U8);
            NVXIO_CHECK_REFERENCE(full_convolved_cost_);

            if (sad > 1)
            {
                full_cost_ = vxCreateVirtualImage(main_graph_, full_width * full_D / 4, full_height, VX_DF_IMAGE_U8);
                NVXIO_CHECK_REFERENCE(full_cost_);
            }

            full_aggregated_cost_ = vxCreateVirtualImage(main_graph_, full_width * full_D / 4, full_height, VX_DF_IMAGE_S16);
            NVXIO_CHECK_REFERENCE(full_aggregated_cost_);

            for (int i = 0; i < pyr_levels; i++)
            {
                int divisor = 1 << i;
                disparity_short_[i] = vxCreateVirtualImage(main_graph_,
                                                           full_width / divisor,
                                                           full_height / divisor,
                                                           VX_DF_IMAGE_S16);
                NVXIO_CHECK_REFERENCE(disparity_short_[i]);
            }
        }

        left_cvt_color_node_ = vxColorConvertNode(main_graph_, left, full_left_gray_);
        NVXIO_CHECK_REFERENCE(left_cvt_color_node_);

        right_cvt_color_node_ = vxColorConvertNode(main_graph_, right, full_right_gray_);
        NVXIO_CHECK_REFERENCE(right_cvt_color_node_);

        for (int i = pyr_levels - 1; i >= 0; i--)
        {
            int divisor = 1 << i;

            int width = full_width / divisor;
            int height = full_height / divisor;
            int D = full_D / D_divisors[i];

            vx_image left_gray = vxCreateVirtualImage(main_graph_, width, height, VX_DF_IMAGE_U8);
            NVXIO_CHECK_REFERENCE(left_gray);

            vx_image right_gray = vxCreateVirtualImage(main_graph_, width, height, VX_DF_IMAGE_U8);
            NVXIO_CHECK_REFERENCE(left_gray);

            vx_node left_downscale_node = vxScaleImageNode(main_graph_, full_left_gray_, left_gray, VX_INTERPOLATION_TYPE_BILINEAR);
            NVXIO_CHECK_REFERENCE(left_downscale_node);

            vx_node right_downscale_node = vxScaleImageNode(main_graph_, full_right_gray_, right_gray, VX_INTERPOLATION_TYPE_BILINEAR);
            NVXIO_CHECK_REFERENCE(right_downscale_node);

            // apply census transform, if requested
            vx_image left_census = NULL, right_census = NULL;
            if (params.ct_win_size > 1)
            {
                left_census =  vxCreateVirtualImage(main_graph_, width, height, VX_DF_IMAGE_U32);
                NVXIO_CHECK_REFERENCE(left_census);
                right_census = vxCreateVirtualImage(main_graph_, width, height, VX_DF_IMAGE_U32);
                NVXIO_CHECK_REFERENCE(right_census);

                vx_node left_census_node = nvxCensusTransformNode(main_graph_, left_gray, left_census, params.ct_win_size);
                NVXIO_CHECK_REFERENCE(left_census_node);
                vx_node right_census_node = nvxCensusTransformNode(main_graph_, right_gray, right_census, params.ct_win_size);
                NVXIO_CHECK_REFERENCE(right_census_node);
            }

            vx_rectangle_t cost_rect { 0, 0, static_cast<vx_uint32>(width * D), static_cast<vx_uint32>(height) };
            convolved_cost_[i] = vxCreateImageFromROI(full_convolved_cost_, &cost_rect);
            NVXIO_CHECK_REFERENCE(convolved_cost_[i]);

            if (sad > 1)
            {
                cost_[i] = vxCreateImageFromROI(full_cost_, &cost_rect);
                NVXIO_CHECK_REFERENCE(cost_[i]);

                // census transformed images should be compared by hamming cost
                vx_node compute_cost_node = NULL;
                if (params.ct_win_size > 1)
                {
                    compute_cost_node = nvxComputeCostHammingNode
                        (main_graph_, left_census, right_census,
                         cost_[i],
                         params.min_disparity / D_divisors[i], params.max_disparity / D_divisors[i],
                         params.hc_win_size);
                }
                else
                {
                    compute_cost_node = nvxComputeModifiedCostBTNode
                        (main_graph_, left_gray, right_gray,
                         cost_[i],
                         params.min_disparity / D_divisors[i], params.max_disparity / D_divisors[i],
                         params.bt_clip_value);
                }
                NVXIO_CHECK_REFERENCE(compute_cost_node);

                vx_node convolve_cost_node = nvxConvolveCostNode
                    (main_graph_,
                     cost_[i], convolved_cost_[i],
                     D, sad);
                NVXIO_CHECK_REFERENCE(convolve_cost_node);
            }
            else
            {
                vx_node compute_cost_node = NULL;
                if (params.ct_win_size > 1)
                {
                    compute_cost_node = nvxComputeCostHammingNode
                        (main_graph_, left_census, right_census,
                         convolved_cost_[i],
                         params.min_disparity / D_divisors[i], params.max_disparity / D_divisors[i],
                         params.hc_win_size);
                }
                else
                {
                    compute_cost_node = nvxComputeModifiedCostBTNode
                        (main_graph_, left_gray, right_gray,
                         convolved_cost_[i],
                         params.min_disparity / D_divisors[i], params.max_disparity / D_divisors[i],
                         params.bt_clip_value);
                }
                NVXIO_CHECK_REFERENCE(compute_cost_node);
            }

            if (i < pyr_levels - 1)
            {
                vx_node cost_prior_node = nvxPSGMCostPriorNode
                    (main_graph_, disparity_short_[i+1],
                     convolved_cost_[i],
                     D);
                NVXIO_CHECK_REFERENCE(cost_prior_node);
            }

            aggregated_cost_[i] = vxCreateImageFromROI(full_aggregated_cost_, &cost_rect);
            NVXIO_CHECK_REFERENCE(aggregated_cost_[i]);

            vx_node aggregate_cost_scanlines_node = nvxAggregateCostScanlinesNode
                (main_graph_,
                 convolved_cost_[i], aggregated_cost_[i],
                 D, params.P1, params.P2, params.scanlines_mask);
            NVXIO_CHECK_REFERENCE(aggregate_cost_scanlines_node);

            vx_node compute_disparity_node = nvxComputeDisparityNode
                (main_graph_,
                 aggregated_cost_[i],
                 disparity_short_[i],
                 params.min_disparity / D_divisors[i], params.max_disparity / D_divisors[i],
                 params.uniqueness_ratio, params.max_diff);
            NVXIO_CHECK_REFERENCE(compute_disparity_node);

            if (i < pyr_levels - 1)
            {
                vx_node disparity_merge_node = nvxPSGMDisparityMergeNode
                    (main_graph_,
                     disparity_short_[i+1],
                     disparity_short_[i], D);
                NVXIO_CHECK_REFERENCE(disparity_merge_node);
            }
        }

        vx_int32 shift = 4;
        vx_scalar s_shift = vxCreateScalar(context, VX_TYPE_INT32, &shift);
        NVXIO_CHECK_REFERENCE(s_shift);
        vx_node convert_depth_node = vxConvertDepthNode
            (main_graph_, disparity_short_[0],
             disparity, VX_CONVERT_POLICY_SATURATE, s_shift);
        vxReleaseScalar(&s_shift);
        NVXIO_CHECK_REFERENCE(convert_depth_node);

        NVXIO_SAFE_CALL( vxVerifyGraph(main_graph_) );
    }

    void SGBM::printPerfs() const
    {
    }
}

StereoMatching* StereoMatching::createStereoMatching(vx_context context, const StereoMatchingParams& params,
                                                     ImplementationType impl,
                                                     vx_image left, vx_image right, vx_image disparity)
{
    switch (impl)
    {
    case HIGH_LEVEL_API:
        return new hlsgm::SGBM(context, params, left, right, disparity);
    case LOW_LEVEL_API:
        return new llsgm::SGBM(context, params, left, right, disparity);
    case LOW_LEVEL_API_PYRAMIDAL:
        return new psgm::SGBM(context, params, left, right, disparity);
    }
    return 0;
}

StereoMatching::StereoMatchingParams::StereoMatchingParams()
{
    min_disparity = 0;
    max_disparity = 64;
    P1 = 8;
    P2 = 109;
    sad = 5;
    bt_clip_value = 31;
    max_diff = 32000;
    uniqueness_ratio = 0;
    scanlines_mask = 85;
    ct_win_size = 0;
    hc_win_size = 1;
    flags = NVX_SGM_PYRAMIDAL_STEREO;
}
