// @file data_reshape.c
//
//  \date Created on: Sep 24, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//



void NCHW2HWNC(const float *nchw_data, int N, int C, int H, int W,
               float *hwnc_data) {

  for (int n = 0; n < N; n++) {
    int in_batch_offset = n * C * H * W;
    int out_batch_offset = n * C;
    for (int c = 0; c < C; ++c) {
      int in_ch_offset = c * H * W + in_batch_offset;
      int out_ch_offset = out_batch_offset + c;
      for (int h = 0; h < H; ++h) {
        int in_row_offset = h * W + in_ch_offset;
        int out_row_offset = h * C * N * W + out_ch_offset;
        for (int w = 0; w < W; ++w) {
          int in_addr = w + in_row_offset;
          int out_addr = out_row_offset+ w * N * C;
          hwnc_data[out_addr] = nchw_data[in_addr];
        }
      }
    }
  }
}

void NCHW2HWCN(const float *nchw_data, int N, int C, int H, int W,
               float *hwcn_data) {

  for (int n = 0; n < N; n++) {
    int in_batch_offset = n * C * H * W;
    int out_batch_offset = n;
    for (int c = 0; c < C; ++c) {
      int in_ch_offset = c * H * W + in_batch_offset;
      int out_ch_offset = out_batch_offset + c * N;
      for (int h = 0; h < H; ++h) {
        int in_row_offset = h * W + in_ch_offset;
        int out_row_offset = h * C * N * W + out_ch_offset;
        for (int w = 0; w < W; ++w) {
          int in_addr = w + in_row_offset;
          int out_addr = out_row_offset+ w * N * C;
          hwcn_data[out_addr] = nchw_data[in_addr];
        }
      }
    }
  }
}

void NCHW2NHWC(const float *nchw_data, int N, int C, int H, int W,
               float *nhwc_data) {
  for (int n = 0; n < N; n++) {
    int in_batch_offset = n * C * H * W;
    int out_batch_offset = n * H * W * C;
    for (int c = 0; c < C; ++c) {
      int in_ch_offset = c * H * W + in_batch_offset;
      int out_ch_offset = out_batch_offset + c;
      for (int h = 0; h < H; ++h) {
        int in_row_offset = h * W + in_ch_offset;
        int out_row_offset = out_ch_offset + h * W * C;
        for (int w = 0; w < W; ++w) {
          int in_addr = w + in_row_offset;
          int out_addr = out_row_offset + w * C;
          nhwc_data[out_addr] = nchw_data[in_addr];
        }
      }
    }
  }
}

void NCHW2CHWN(const float *nchw_data, int N, int C, int H, int W,
               float *chwn_data) {
  for (int n = 0; n < N; ++n) {
    int in_batch_offset = n * C * H * W;
    int out_batch_offset = n;
    for (int c = 0; c < C; ++c) {
      int in_ch_offset = in_batch_offset + c * H * W;
      int out_ch_offset = out_batch_offset + c * H * W * N;
      for (int h = 0; h < H; ++h) {
        int in_row_offset = in_ch_offset + h * W;
        int out_row_offset = out_ch_offset + h * W * N;
        for (int w = 0; w < W; ++w) {
          int in_addr = in_row_offset + w;
          int out_addr = out_row_offset + w * N;
          chwn_data[out_addr] = nchw_data[in_addr];
        }
      }
    }
  }
}

void NHWC2NCHW(const float *nhwc_data, int N, int C, int H, int W,
               float *nchw_data) {
  for (int n = 0; n < N; ++n) {
    int in_batch_offset = n * H * W * C;
    int out_batch_offset = n * C * H * W;
    for(int h = 0; h < H; ++h) {
      int in_row_offset = in_batch_offset + h * W * C;
      int out_row_offset = out_batch_offset + h * W;
      for (int w = 0; w < W; ++w) {
        int in_col_offset = in_row_offset + w * C;
        int out_col_offset = out_row_offset + w;
        for (int c = 0; c < C; ++c) {
          int in_addr = in_col_offset + c;
          int out_addr = out_col_offset + c * H * W;
          nchw_data[out_addr] = nhwc_data[in_addr];
        }
      }
    }
  }
}
