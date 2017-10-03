// @file main.c
//
//  \date Created on: Sep 22, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//

#include <conv_layers.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "common_types.h"
#include "data_reshape.h"
#include "utils.h"


static const int conv_test_k = 3;
static const int conv_test_in_c = 3;
static const int conv_test_in_h = 5;
static const int conv_test_in_w = 5;
static const int conv_test_out_c = 2;
static const int conv_test_batch = 1;

static const float conv_test_in_data[] = {
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
    3, 3, 3, 3, 3,
    3, 3, 3, 3, 3,
    3, 3, 3, 3, 3,
    3, 3, 3, 3, 3,
};

static const float conv_test_filter[] = {
    1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9,
    1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9,
    1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9,
    1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9,
    1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9,
    1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9
};

static const float conv_test_bias[] = {
    0, 0
};

static const float conv_test_ref_out[] = {
    24.0/9, 36.0/9, 36.0/9, 36.0/9, 24.0/9,
    36.0/9, 54.0/9, 54.0/9, 54.0/9, 36.0/9,
    36.0/9, 54.0/9, 54.0/9, 54.0/9, 36.0/9,
    36.0/9, 54.0/9, 54.0/9, 54.0/9, 36.0/9,
    24.0/9, 36.0/9, 36.0/9, 36.0/9, 24.0/9,
    /*------------------------------------*/
    24.0/9, 36.0/9, 36.0/9, 36.0/9, 24.0/9,
    36.0/9, 54.0/9, 54.0/9, 54.0/9, 36.0/9,
    36.0/9, 54.0/9, 54.0/9, 54.0/9, 36.0/9,
    36.0/9, 54.0/9, 54.0/9, 54.0/9, 36.0/9,
    24.0/9, 36.0/9, 36.0/9, 36.0/9, 24.0/9
};

void TestCppConvnetConvLayer() {
  bool print_outputs = false;
  bool padding_en = true;
  bool bias_en = true;

  int ker_size = 3;
  int group = 2;
  int stride = 1;
  int N = 1;
  int C = 2;
  int H = 5;
  int W = 5;
  int M = 4;
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

  CppConvnetConvLayer(in_data, filters, bias,
                           in_dim, filt_dim, stride,
                           pad, group, output);

  if (print_outputs) {
    printf("Output of kn2xyz method\n");
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

void TestLayoutConverters() {
  // NCHW to HWNC converter.
  int N = 2;
  int C = 4;
  int H = 3;
  int W = 3;
  float *nchw_data = malloc(N * C * H * W * sizeof(float));
  float *hwnc_output = malloc(N * C * H * W * sizeof(float));
  RandInitF32(nchw_data, N * C * H * W);

  NCHW2HWNC(nchw_data, N, C, H, W, hwnc_output);
  TensorDim in_dim = {N, C, H, W};
  TensorDim out_dim = {H, W, N, C};
  PrintTensor(nchw_data, in_dim);
  PrintTensor(hwnc_output, out_dim);

  free(nchw_data);
  free(hwnc_output);
}

void TestKer2RowConvLayerKnownOutput() {
  int pad = conv_test_k/2;
  int group = 1;
  int stride = 1;
  TensorDim in_dim = {conv_test_batch, conv_test_in_c,
      conv_test_in_h, conv_test_in_w};
  TensorDim filt_dim = {conv_test_out_c, conv_test_in_c, conv_test_k,
      conv_test_k};
  TensorDim out_dim;
  out_dim.w = (in_dim.w + (pad + pad) - filt_dim.w) / stride + 1;
  out_dim.h = (in_dim.h + (pad + pad) - filt_dim.h) / stride + 1;
  out_dim.c = conv_test_out_c;
  out_dim.n = in_dim.n;
  const float *in_data = conv_test_in_data;
  const float *filters = conv_test_filter;
  const float *bias = conv_test_bias;
  float *output = malloc(out_dim.n * out_dim.c * out_dim.h * out_dim.w *
                         sizeof(float));
  Kn2RowConvLayer(in_data, filters, bias, in_dim, filt_dim, stride, pad,
                        group, output);

  PrintTensor(output, out_dim);
  TensorCompare(output, conv_test_ref_out, out_dim);
  free(output);
}

void TestKer2FlavorConvLayer() {
  // Configurations
  // Enable kn2row or kn2col
  bool kn2row = true;
  bool print_outputs = true;
  bool padding_en = true;
  bool bias_en = true;

  int ker_size = 3;
  int group = 1;
  int stride = 1;
  int N = 1;
  int C = 2;
  int H = 5;
  int W = 5;
  int M = 2;
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

  if (kn2row) {
    printf("Using Kn2Row convolution\n");
    Kn2RowConvLayer(in_data, filters, bias, in_dim, filt_dim, stride, pad,
                     group, output);
  } else {
    printf("Using Kn2Col convolution\n");
    Kn2ColConvLayer(in_data, filters, bias, in_dim, filt_dim, stride, pad,
                         group, output);
  }

  if (print_outputs) {
    printf("Output of kn2xyz method\n");
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

void TestIm2FlavorConvLayer() {
  // Configurations
  // Enable kn2row or kn2col
  bool im2col = true;
  bool print_outputs = true;
  bool padding_en = true;
  bool bias_en = false;

  int ker_size = 3;
  int group = 3;
  int stride = 1;
  int N = 1;
  int C = 3;
  int H = 5;
  int W = 5;
  int M = 9;
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
  float *spad = malloc(out_dim.h * out_dim.w * in_dim.c *
                       filt_dim.w * filt_dim.h);
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

  if (im2col) {
    printf("Using im2col convolution\n");
    Im2ColConvLayer(in_data, filters, bias, spad, in_dim, out_dim, ker_size,
                    group, pad, stride, bias_en, output);
  } else {

  }

  if (print_outputs) {
    printf("Output of im2xyz method\n");
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
  free(spad);
}
void TestMatShiftAdd() {
  int mat_h = 5;
  int mat_w = 5;
  int row_shift = -1;
  int col_shift = -1;
  float *mat1 = malloc(mat_h * mat_w * sizeof(float));
  float *mat2 = malloc(mat_h * mat_w * sizeof(float));
  RandInitF32(mat1, mat_h * mat_w);
  RandInitF32(mat2, mat_h * mat_w);

  PrintMat("base mat", mat1, mat_h, mat_w, CblasRowMajor);
  PrintMat("overlap mat", mat2, mat_h, mat_w, CblasRowMajor);

  MatrixShiftAdd(mat1, mat_h, mat_w, mat2, mat_h, mat_w, row_shift, col_shift);

  PrintMat("result mat", mat1, mat_h, mat_w, CblasRowMajor);
  free(mat1);
  free(mat2);
}

int main(void) {
  //TestLayoutConverters();
  //TestKer2RowConvLayerKnownOutput();
  //TestKer2FlavorConvLayer();
  //TestMatShiftAdd();
  //TestCppConvnetConvLayer();

  TestIm2FlavorConvLayer();
  return 0;
}
