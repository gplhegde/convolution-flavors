// @file cpp_convnet_layers.h
//
//  \date Created on: Sep 22, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//

#ifndef INC_CONV_LAYERS_H_
#define INC_CONV_LAYERS_H_
#include <stdbool.h>
#include <common_types.h>

void RefConv2dF32(const float *input, const float *weight,
    const float *bias, const int in_c, const int in_h,
    const int in_w, const int out_c, const int out_h, const int out_w,
    const int ker_size, const int group,
    const int pad, const int stride, const int bias_en, float *output);

bool CppConvnetConvLayer(const float *in_data, const float *filters,
                         const float *bias, TensorDim in_dim,
                         TensorDim filt_dim, int stride, int pad, int group,
                         float *output);

bool Kn2RowConvLayer(const float *in_data, const float *filters,
                         const float *bias, TensorDim in_dim,
                         TensorDim filt_dim, int stride, int pad, int group,
                         float *output);

bool Kn2ColConvLayer(const float *in_data, const float *filters,
                         const float *bias, TensorDim in_dim,
                         TensorDim filt_dim, int stride, int pad, int group,
                         float *output);

void Im2ColConvLayer(const float *input, const float *weight,
    const float *bias, float *scratchpad, const TensorDim in_dim,
    const TensorDim out_dim, const int ker_size, const int group,
    const int pad, const int stride, const int bias_en, float *output);

void MatrixShiftAdd(float *base_mat,
                     int base_no_rows, int base_no_cols,
                     float *overlap_mat,
                     int ov_no_rows, int ov_no_cols,
                     int row_shift, int col_shift);

void WinoGradConvHook();
#endif  // INC_CONV_LAYERS_H_
