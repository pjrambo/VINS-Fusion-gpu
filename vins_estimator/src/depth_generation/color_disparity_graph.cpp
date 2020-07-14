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
#ifdef WITH_VWORKS

#include <algorithm>

#include "../featureTracker/ovx_replaceheader.hpp"

#include "color_disparity_graph.hpp"

ColorDisparityGraph::ColorDisparityGraph(vx_context context, vx_image disparity, vx_image output, vx_int32 ndisp) :
    graph_(nullptr)
{

    // convert disparity from fixed point to grayscale
    
    
    NVXIO_ASSERT(ndisp <= 256);

    vx_lut r_lut = vxCreateLUT(context, VX_TYPE_UINT8, 256);
    vx_lut g_lut = vxCreateLUT(context, VX_TYPE_UINT8, 256);
    vx_lut b_lut = vxCreateLUT(context, VX_TYPE_UINT8, 256);
    fillLUT(r_lut, g_lut, b_lut, ndisp);

    graph_ = vxCreateGraph(context);

    vx_int32 shift = 4;
    vx_scalar s_shift = vxCreateScalar(context, VX_TYPE_INT32, &shift);
    NVXIO_CHECK_REFERENCE(s_shift);

    vx_uint32 width = 0;
    vx_uint32 height = 0;

    NVXIO_SAFE_CALL( vxQueryImage(disparity, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)) );
    NVXIO_SAFE_CALL( vxQueryImage(disparity, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)) );
    vx_image disparity_short = vxCreateVirtualImage(graph_, width, height, VX_DF_IMAGE_U8);

    convert_depth_node_ = vxConvertDepthNode(graph_, disparity, disparity_short, VX_CONVERT_POLICY_SATURATE, s_shift);
    vxReleaseScalar(&s_shift);
    NVXIO_CHECK_REFERENCE(convert_depth_node_);


    vx_image r_img = vxCreateVirtualImage(graph_, 0, 0, VX_DF_IMAGE_U8);
    vx_image g_img = vxCreateVirtualImage(graph_, 0, 0, VX_DF_IMAGE_U8);
    vx_image b_img = vxCreateVirtualImage(graph_, 0, 0, VX_DF_IMAGE_U8);

    lut_node_[0] = vxTableLookupNode(graph_, disparity_short, r_lut, r_img);
    lut_node_[1] = vxTableLookupNode(graph_, disparity_short, g_lut, g_img);
    lut_node_[2] = vxTableLookupNode(graph_, disparity_short, b_lut, b_img);

    combine_node_ = vxChannelCombineNode(graph_, r_img, g_img, b_img, nullptr, output);

    NVXIO_SAFE_CALL( vxVerifyGraph(graph_) );

    vxReleaseImage(&r_img);
    vxReleaseImage(&g_img);
    vxReleaseImage(&b_img);
    vxReleaseImage(&disparity_short);

    vxReleaseLUT(&r_lut);
    vxReleaseLUT(&g_lut);
    vxReleaseLUT(&b_lut);
}

void ColorDisparityGraph::fillLUT(vx_lut r_lut, vx_lut g_lut, vx_lut b_lut, vx_int32 ndisp)
{
    vx_map_id r_lut_map_id;
    vx_map_id g_lut_map_id;
    vx_map_id b_lut_map_id;
    vx_uint8* r_lut_ptr;
    vx_uint8* g_lut_ptr;
    vx_uint8* b_lut_ptr;
    NVXIO_SAFE_CALL( vxMapLUT(r_lut, &r_lut_map_id, (void **)&r_lut_ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0) );
    NVXIO_SAFE_CALL( vxMapLUT(g_lut, &g_lut_map_id, (void **)&g_lut_ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0) );
    NVXIO_SAFE_CALL( vxMapLUT(b_lut, &b_lut_map_id, (void **)&b_lut_ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0) );

    for (vx_int32 d = 0; d < 256; ++d)
    {
        vx_int32 H = ((ndisp - d) * 240) / ndisp;
        vx_float32 S = 1.0f;
        vx_float32 V = 1.0f;

        vx_int32 hi = (H / 60) % 6;
        vx_float32 f = H / 60.0f - H / 60;
        vx_float32 p = V * (1.0f - S);
        vx_float32 q = V * (1.0f - f * S);
        vx_float32 t = V * (1.0f - (1 - f) * S);

        vx_float32 rval = 0.0f, gval = 0.0f, bval = 0.0f;

        if (hi == 0) //R = V, G = t, B = p
        {
            bval = p;
            gval = t;
            rval = V;
        }
        if (hi == 1) // R = q, G = V, B = p
        {
            bval = p;
            gval = V;
            rval = q;
        }
        if (hi == 2) // R = p, G = V, B = t
        {
            bval = t;
            gval = V;
            rval = p;
        }
        if (hi == 3) // R = p, G = q, B = V
        {
            bval = V;
            gval = q;
            rval = p;
        }
        if (hi == 4) // R = t, G = p, B = V
        {
            bval = V;
            gval = p;
            rval = t;
        }
        if (hi == 5) // R = V, G = p, B = q
        {
            bval = q;
            gval = p;
            rval = V;
        }

        r_lut_ptr[d] = std::max(0.f, std::min(rval, 1.f)) * 255.f;
        g_lut_ptr[d] = std::max(0.f, std::min(gval, 1.f)) * 255.f;
        b_lut_ptr[d] = std::max(0.f, std::min(bval, 1.f)) * 255.f;
    }

    vxUnmapLUT(r_lut, r_lut_map_id);
    vxUnmapLUT(g_lut, g_lut_map_id);
    vxUnmapLUT(b_lut, b_lut_map_id);
}

ColorDisparityGraph::~ColorDisparityGraph()
{
    vxReleaseGraph(&graph_);
}

void ColorDisparityGraph::process()
{
    NVXIO_SAFE_CALL( vxProcessGraph(graph_) );
}

void ColorDisparityGraph::printPerfs()
{
#ifdef OVX
    ovxio::printPerf(graph_, "Color Disparity");
    ovxio::printPerf(lut_node_[0], "Red Channel Table Lookup");
    ovxio::printPerf(lut_node_[1], "Green Channel Table Lookup");
    ovxio::printPerf(lut_node_[2], "Blue Channel Table Lookup");
    ovxio::printPerf(combine_node_, "Channel Combine");
#endif
}

#endif