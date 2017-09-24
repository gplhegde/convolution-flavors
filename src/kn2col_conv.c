// @file ker2col_conv.c
//
//  \date Created on: Sep 24, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//

#include <stdbool.h>
#include <cblas.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "common_types.h"
#include "data_reshape.h"
#include "utils.h"
#include "conv_layers.h"

bool Kn2ColConvLayer(const float *in_data, const float *filters,
                         const float *bias, TensorDim in_dim,
                         TensorDim filt_dim, int stride, int pad, int group,
                         float *output) {
  // Currently we have limited support.
  assert(group == 1);
  assert((pad == 0) || (pad == filt_dim.w / 2));
  assert(in_dim.n == 1);
  assert(filt_dim.h == filt_dim.w);
  assert(stride == 1);

  // Output dimensions.
  TensorDim out_dim;
  out_dim.w = (in_dim.w + (pad + pad) - filt_dim.w) / stride + 1;
  out_dim.h = (in_dim.h + (pad + pad) - filt_dim.h) / stride + 1;
  out_dim.c = filt_dim.n;
  out_dim.n = in_dim.n;

  // Reshape filters in CHWN (ker_size x ker_size x no_in_ch x no_out_ch) format
  float *kkcm_filters = malloc(filt_dim.n * filt_dim.c * filt_dim.h *
                               filt_dim.w * sizeof(float));
  NCHW2HWCN(filters, filt_dim.n, filt_dim.c, filt_dim.h, filt_dim.w,
            kkcm_filters);

  // Just for convenience
  int H = in_dim.h;
  int W = in_dim.w;
  float alpha = 1.0;
  float beta = 0.0;

  // We need separate buffer because GEMM output will have width = H*W even
  // if there is no padding (pad = 0).
  float *gemm_output = malloc(out_dim.c * H * W * sizeof(float));
  float *nchw_gemm_output = malloc(out_dim.c * H * W * sizeof(float));
  // Prefill output buffer with bias if present else set to zero.
  if (bias) {
    for (int m = 0; m < out_dim.c; ++m) {
      for (int a = 0; a < out_dim.h * out_dim.w; ++a) {
        output[m * out_dim.h * out_dim.w + a] = bias[m];
      }
      // For batch size > 1
      for (int b = 1; b < out_dim.n; ++b) {
        memcpy(output + b * out_dim.c * out_dim.h * out_dim.w,
               output, out_dim.c * out_dim.h * out_dim.w * sizeof(float));
      }
    }
  } else {
    memset(output, 0, out_dim.n * out_dim.c * out_dim.h * out_dim.w *
           sizeof(float));
  }

  for (int kr = 0; kr < filt_dim.h; kr++) {
    int row_shift = pad - kr;
    for (int kc = 0; kc < filt_dim.w; kc++) {
      int group_no = kr * filt_dim.w + kc;
      int col_shift = pad - kc;
      // Matrix dimensions - A -> mxk B -> kxn  C --> mxn
      int m = in_dim.h * in_dim.w;
      int k = filt_dim.c;
      int n = filt_dim.n;

      // output is in H x W x C format.
      cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                  m, n, k, alpha, in_data, m,
                  kkcm_filters + group_no * filt_dim.c * filt_dim.n, n, beta,
                  gemm_output, n);

      // convert to CxHxW format.
      // FIXME: this will be slow. Need to find other ways :(
      NHWC2NCHW(gemm_output, 1, filt_dim.n, H, W, nchw_gemm_output);

      for (int omap = 0; omap < filt_dim.n; omap++) {
        MatrixShiftAdd(output + omap * out_dim.h * out_dim.w,
                        out_dim.h, out_dim.w,
                        nchw_gemm_output + omap * H * W,
                        H, W, row_shift, col_shift);
      }
    }
  }
  free(kkcm_filters);
  free(gemm_output);
  free(nchw_gemm_output);
  return true;
}
