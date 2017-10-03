// @file im2col_conv.c
//
//  \date Created on: Sep 30, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//

#include <stdbool.h>
#include <cblas.h>
#include "common_types.h"

//From Berkeley Vision's Caffe
//Refer to Caffe's license : https://github.com/BVLC/caffe/blob/master/LICENSE
static inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return (unsigned int)a < (unsigned int)(b);
}

void Im2Col(const float *data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float *data_col) {
  const int output_h = (height + 2 * pad_h -
    ((kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    ((kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}


void Im2ColConvLayer(const float *input, const float *weight,
    const float *bias, float *scratchpad, const TensorDim in_dim,
    const TensorDim out_dim, const int ker_size, const int group,
    const int pad, const int stride, const int bias_en, float *output) {

  int C = in_dim.c;
  int H = in_dim.h;
  int W = in_dim.w;
  int in_ch_per_group = C / group;
  int out_ch_per_group = out_dim.c / group;
  float alpha = 1;
  float beta = 0;

  for (int b = 0; b < in_dim.n; b++) {
    int in_offset = b * C * H * W;
    Im2Col(input + in_offset, C, H, W, ker_size, ker_size, pad, pad, stride,
           stride, scratchpad);
    for (int g = 0; g < group; g++) {
      int weight_offset = g * ker_size * ker_size * in_ch_per_group * out_ch_per_group;
      int data_offset = g * (ker_size * ker_size * in_ch_per_group) *
          (out_dim.h * out_dim.w);
      int out_offset = (b * out_dim.c + g * out_ch_per_group) *
          out_dim.h * out_dim.w;

      int m = out_ch_per_group;
      int k = in_ch_per_group * ker_size * ker_size;
      int n = out_dim.h * out_dim.w;
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 m, n, k, alpha, weight + weight_offset,
                 k, scratchpad + data_offset, n, beta, output + out_offset, n);
    }
  }

}
