// @file winograd_conv1d.c
//
//  \date Created on: Oct 3, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//

#include "common_types.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include "utils.h"
#include "conv_layers.h"

/* Winograd transformation matrices for F(4, 3)
 *
 */
// Output transformation matrix
float AT[4*6] = {
    1,  1,   1,   1,   1,   0,
    0,  1,  -1,   2,  -2,   0,
    0,  1,   1,   4,   4,   0,
    0,  1,  -1,   8,  -8,   1
  };
// filter transformation matrix
float G[6*3] = {
     1/4,      0,      0,
    -1/6,   -1/6,   -1/6,
    -1/6,    1/6,   -1/6,
    1/24,   1/12,    1/6,
    1/24,  -1/12,    1/6,
       0,      0,      1
    };
// input feature transform matrix
float BT[6*6] = {
    4,   0,  -5,   0,  1,  0,
    0,  -4,  -4,   1,  1,  0,
    0,   4,  -4,  -1,  1,  0,
    0,  -2,  -1,   2,  1,  0,
    0,   2,  -1,  -2,  1,  0,
    0,   4,   0,  -5,  0,  1
  };

float_Wvec EleWiseProd(float_Wvec d1, float_Wvec d2) {
  float_Wvec out;
  for (int i = 0; i < W_VEC; i++) {
    out.d[i] = d1.d[i] * d2.d[i];
  }
  return out;
}

// g = G*f
float_Wvec FilterTransformer(float_Svec filt) {
  float_Wvec g;
  g.d[0] = 0.25 * filt.d[0];
  g.d[1] = -(filt.d[0] + filt.d[1] + filt.d[2]) / 6;
  g.d[2] = (-filt.d[0] + filt.d[1] - filt.d[2]) / 6;
  g.d[3] = filt.d[0] / 24 + filt.d[1] / 12 + filt.d[2] / 6;
  g.d[4] = filt.d[0] / 24 - filt.d[1] / 12 + filt.d[2] / 6;
  g.d[5] = filt.d[2];

  return g;
}

// d = GT * i
float_Wvec DataTransform(float_Wvec i) {
  float_Wvec d;

  d.d[0] = 4*i.d[0] - 5*i.d[2] + i.d[4];
  d.d[1] = -4*i.d[1] -4*i.d[2] + 1*i.d[3] + 1*i.d[4];
  d.d[2] = 4*i.d[1] -4*i.d[2] -i.d[3] + i.d[4];
  d.d[3] = -2*i.d[1] -i.d[2] +2*i.d[3] + i.d[4];
  d.d[4] = 2*i.d[1] -i.d[2] -2*i.d[3] +i.d[4];
  d.d[5] = 4*i.d[1] -5*i.d[3] +i.d[5];
  return d;
}

// o = AT * y
float_Qvec OutputTransform(float_Wvec y) {
  float_Qvec o;

  o.d[0] = y.d[0] + y.d[1] + y.d[2] + y.d[3] + y.d[4];
  o.d[1] = y.d[1] - y.d[2] + 2*(y.d[3] - y.d[4]);
  o.d[2] = y.d[1] + y.d[2] + 4 *(y.d[3] + y.d[4]);
  o.d[3] = y.d[1] - y.d[2] + 8 * (y.d[3] - y.d[4]) + y.d[5];

  return o;
}

float_Qvec DirectConv1D(float_Wvec input, float_Svec filt) {
  float_Qvec out;
  for (int i = 0; i < Q_VEC; i++) {
    out.d[i] = input.d[i] * filt.d[0] + input.d[i+1] *
        filt.d[1] + input.d[i+2] * filt.d[2];
  }
  return out;
}

float_Qvec WinogradConv1D(float_Wvec input, float_Svec filt) {
  float_Wvec tx_data = DataTransform(input);
  float_Wvec tx_filt = FilterTransformer(filt);
  float_Wvec tx_out = EleWiseProd(tx_data, tx_filt);
  float_Qvec out = OutputTransform(tx_out);
  return out;
}

float_Qvec AddPartialOutputs(float_Qvec acc, float_Qvec pp) {
  float_Qvec out;
  for (int i = 0; i < Q_VEC; i++) {
    out.d[i] = acc.d[i] + pp.d[i];
  }
  return out;
}

float_Wvec AddPartialWinogradOutputs(float_Wvec acc, float_Wvec pp) {
  float_Wvec out;
  for (int i = 0; i < W_VEC; i++) {
    out.d[i] = acc.d[i] + pp.d[i];
  }
  return out;
}

void SimpleTest() {
  float_Wvec in_data = {{1, 2, 0, 4, -3, 6}};
  float_Svec filt = {{1, -1, 1}};

  float_Wvec tx_data = DataTransform(in_data);
  float_Wvec tx_filt = FilterTransformer(filt);

  float_Wvec tx_out = EleWiseProd(tx_data, tx_filt);

  float_Qvec out = OutputTransform(tx_out);
  float_Qvec direct_conv = DirectConv1D(in_data, filt);
  for (int i = 0; i < Q_VEC; i++) {
    printf("Ref : %f\tComputed : %f\n", direct_conv.d[i], out.d[i]);
  }
}

void VectoredConvLayer(const float *input, const float *weight,
    const float *bias, const TensorDim in_dim,
    const TensorDim out_dim, const int ker_size, const int group,
    const int pad, const int stride, const int bias_en, float *output) {
  assert(group == 1);
  assert(in_dim.n == 1);
  assert(stride == 1);
  assert(ker_size == S_VEC);
  assert(out_dim.c % K_VEC == 0);
  assert(in_dim.c % C_VEC == 0);

  int no_tiles_x = ceil((float)(in_dim.w + 2*pad - ker_size + 1)/Q_VEC);
  int no_tiles_y = in_dim.h + 2*pad - ker_size + 1;
  int filter_tiles_y = S_VEC;
  int no_Cvec = ceil(in_dim.c / C_VEC);
  int no_Kvec = ceil(out_dim.c / K_VEC);

  for (int v_omap = 0; v_omap < no_Kvec; v_omap++) {
    for (int o_h = 0; o_h < no_tiles_y; o_h++) {
      for (int tile_x = 0; tile_x < no_tiles_x; tile_x++) {
        float_Qvec feat_out[K_VEC] = {0};
        for (int v_imap = 0; v_imap < no_Cvec; v_imap++) {
          for (int imap = 0; imap < C_VEC; imap++) {
            for (int ftile = 0; ftile < filter_tiles_y; ftile++) {
              float_Wvec feat_in;
              float_Svec filt[K_VEC];
              float_Qvec partial_out[K_VEC];
              // read a feature vector of size W_VEC with padding zero if needed.
              int imap_offset = (v_imap * C_VEC + imap) * in_dim.h * in_dim.w;
              int row = -pad + o_h + ftile;
              int col = -pad + tile_x * Q_VEC;

              if (row < 0 || row >= in_dim.h) {
                // all zeros
                for (int f = 0; f < W_VEC; f++) {
                  feat_in.d[f] = 0;
                }
              } else if (col < 0) {
                // zero prefix to the vector
                int f;
                for (f = 0; f < -col; f++) feat_in.d[f] = 0;
                for (; f < W_VEC; f++) {
                  feat_in.d[f] = input[imap_offset + row * in_dim.w + col + f];
                }
              } else if (col > (in_dim.w - W_VEC)) {
                // zero suffix to the vector.
                int f;
                for (f = 0; f < in_dim.w - col; f++) {
                  feat_in.d[f] = input[imap_offset + row * in_dim.w + col + f];
                }
                for (; f < W_VEC; f++) feat_in.d[f] = 0;
              } else {
                // all valid features.
                for (int f = 0; f < W_VEC; f++) {
                  feat_in.d[f] = input[imap_offset + row * in_dim.w + col + f];
                }
              }
              // read the filter vector for this filter tile. Basically read one
              // row of the KxK filter for all K_VEC output maps.
              for (int k = 0; k < K_VEC; k++) {
                int cur_omap = v_omap * K_VEC + k;
                int cur_imap = v_imap * C_VEC + imap;
                int filt_offset = cur_omap * in_dim.c * ker_size * ker_size +
                    cur_imap * ker_size * ker_size + ftile * ker_size;
                for (int coeff = 0; coeff < S_VEC; coeff++) {
                  filt[k].d[coeff] = weight[filt_offset + coeff];
                }
              }
              for (int omap = 0; omap < K_VEC; omap++) {
                partial_out[omap] = DirectConv1D(feat_in, filt[omap]);
                feat_out[omap] = AddPartialOutputs(feat_out[omap],
                                                   partial_out[omap]);
              }
            }
          }
        } // done computing one tile of size Qvec for Kvec maps.
        int out_row = o_h;
        int out_col_start = tile_x * Q_VEC;
        int valid_out_cols = out_dim.w - out_col_start;

        for (int om = 0; om < K_VEC; om++) {
          int out_map_offset = (v_omap * K_VEC + om) * out_dim.h * out_dim.w;
          for (int oc = 0; oc < valid_out_cols; oc++) {
            output[out_map_offset + out_row * out_dim.w + out_col_start + oc] =
                feat_out[om].d[oc];
          }
        }
      }
    }
  }
}

void WinogradConvLayer(const float *input, const float *weight,
    const float *bias, const TensorDim in_dim,
    const TensorDim out_dim, const int ker_size, const int group,
    const int pad, const int stride, const int bias_en, float *output) {
  assert(group == 1);
  assert(in_dim.n == 1);
  assert(stride == 1);
  assert(ker_size == S_VEC);
  assert(out_dim.c % K_VEC == 0);
  assert(in_dim.c % C_VEC == 0);

  int no_tiles_x = ceil((float)(in_dim.w + 2*pad - ker_size + 1)/Q_VEC);
  int no_tiles_y = in_dim.h + 2*pad - ker_size + 1;
  int filter_tiles_y = S_VEC;
  int no_Cvec = ceil(in_dim.c / C_VEC);
  int no_Kvec = ceil(out_dim.c / K_VEC);

  for (int v_omap = 0; v_omap < no_Kvec; v_omap++) {
    for (int o_h = 0; o_h < no_tiles_y; o_h++) {
      for (int tile_x = 0; tile_x < no_tiles_x; tile_x++) {
        float_Wvec win_feat_out[K_VEC] = {0};
        float_Qvec feat_out[K_VEC];
        for (int v_imap = 0; v_imap < no_Cvec; v_imap++) {
          for (int imap = 0; imap < C_VEC; imap++) {
            for (int ftile = 0; ftile < filter_tiles_y; ftile++) {
              float_Wvec feat_in;
              float_Svec filt[K_VEC];
              float_Wvec tx_filt[K_VEC];
              float_Wvec partial_out[K_VEC];
              // read a feature vector of size W_VEC with padding zero if needed.
              int imap_offset = (v_imap * C_VEC + imap) * in_dim.h * in_dim.w;
              int row = -pad + o_h + ftile;
              int col = -pad + tile_x * Q_VEC;

              if (row < 0 || row >= in_dim.h) {
                // all zeros
                for (int f = 0; f < W_VEC; f++) {
                  feat_in.d[f] = 0;
                }
              } else if (col < 0) {
                // zero prefix to the vector
                int f;
                for (f = 0; f < -col; f++) feat_in.d[f] = 0;
                for (; f < W_VEC; f++) {
                  feat_in.d[f] = input[imap_offset + row * in_dim.w + col + f];
                }
              } else if (col > (in_dim.w - W_VEC)) {
                // zero suffix to the vector.
                int f;
                for (f = 0; f < in_dim.w - col; f++) {
                  feat_in.d[f] = input[imap_offset + row * in_dim.w + col + f];
                }
                for (; f < W_VEC; f++) feat_in.d[f] = 0;
              } else {
                // all valid features.
                for (int f = 0; f < W_VEC; f++) {
                  feat_in.d[f] = input[imap_offset + row * in_dim.w + col + f];
                }
              }
              // Transform features to Winograd feaures
              feat_in = DataTransform(feat_in);
              // read the filter vector for this filter tile. Basically read one
              // row of the KxK filter for all K_VEC output maps.
              for (int k = 0; k < K_VEC; k++) {
                int cur_omap = v_omap * K_VEC + k;
                int cur_imap = v_imap * C_VEC + imap;
                int filt_offset = cur_omap * in_dim.c * ker_size * ker_size +
                    cur_imap * ker_size * ker_size + ftile * ker_size;
                for (int coeff = 0; coeff < S_VEC; coeff++) {
                  filt[k].d[coeff] = weight[filt_offset + coeff];
                }
                // Winograd transform filters
                tx_filt[k] = FilterTransformer(filt[k]);
              }
              for (int omap = 0; omap < K_VEC; omap++) {
                partial_out[omap] = EleWiseProd(feat_in, tx_filt[omap]);
                win_feat_out[omap] = AddPartialWinogradOutputs(win_feat_out[omap],
                                                   partial_out[omap]);
              }
            }
          }
        } // done computing one tile of size Qvec for Kvec maps.
        int out_row = o_h;
        int out_col_start = tile_x * Q_VEC;
        int valid_out_cols = out_dim.w - out_col_start;

        for (int om = 0; om < K_VEC; om++) {
          // Winograd inverse transform
          feat_out[om] = OutputTransform(win_feat_out[om]);
          int out_map_offset = (v_omap * K_VEC + om) * out_dim.h * out_dim.w;
          for (int oc = 0; oc < valid_out_cols; oc++) {
            output[out_map_offset + out_row * out_dim.w + out_col_start + oc] =
                feat_out[om].d[oc];
          }
        }
      }
    }
  }
}

void TestVectoredConvLayer() {
  bool winograd = true;
  bool print_outputs = false;
  bool padding_en = false;
  bool bias_en = false;

  int ker_size = 3;
  int group = 1;
  int stride = 1;
  int N = 1;
  int C = 12;
  int H = 4;
  int W = 4;
  int M = 128;
  int pad = 0;
  if (padding_en) {
    pad = ker_size / 2;
  }

  TensorDim in_dim = {N, C, H, W};
  TensorDim filt_dim = {M, C, ker_size, ker_size};
  TensorDim out_dim;
  out_dim.w = (in_dim.w + (pad + pad) - filt_dim.w) / stride + 1;
  out_dim.h = (in_dim.h + (pad + pad) - filt_dim.h) / stride + 1;
  out_dim.c = M;
  out_dim.n = in_dim.n;
  float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
  float *filters = malloc(TensorSize(filt_dim) * sizeof(float));
  float *bias = malloc(out_dim.c * sizeof(float));
  float *output = malloc(TensorSize(out_dim) * sizeof(float));
  float *ref_output = malloc(TensorSize(out_dim) * sizeof(float));

  RandInitF32(in_data, TensorSize(in_dim));
  RandInitF32(filters, TensorSize(filt_dim));
  if (bias_en) {
    RandInitF32(bias, out_dim.c);
  }

  RefConv2dF32(in_data, filters,
      bias, in_dim.c, in_dim.h,
      in_dim.w, out_dim.c, out_dim.h, out_dim.w,
      ker_size, group,
      pad, stride, bias_en, ref_output);

  if (winograd) {
    printf("Using Winograd convolution\n");
    WinogradConvLayer(in_data, filters, bias, in_dim, out_dim, ker_size,
                          group, pad, stride, bias_en, output);
  } else {
    printf("Using vectored convolution\n");
    VectoredConvLayer(in_data, filters, bias, in_dim, out_dim, ker_size,
                      group, pad, stride, bias_en, output);
  }

  if (print_outputs) {
    printf("Output of winograd/vectored method\n");
    PrintTensor(output, out_dim);
    printf("Output of reference implementation\n");
    PrintTensor(ref_output, out_dim);
  }
  if (TensorCompare(output, ref_output, out_dim)) {
    printf("PASS\n");
  } else {
    printf("FAIL\n");
  }
  free(in_data);
  free(bias);
  free(ref_output);
  free(filters);
  free(output);
}

void WinoGradConvHook() {
  //SimpleTest();
  TestVectoredConvLayer();
}


