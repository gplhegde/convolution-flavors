// @file reference_conv.c
//
//  \date Created on: Sep 22, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//

#include <stdint.h>

// Taken from Caffe implementation
// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
static inline uint32_t is_a_ge_zero_and_a_lt_b(int a, int b) {
  return (unsigned int) a < (unsigned int) b;
}

void RefConv2dF32(const float *input, const float *weight,
    const float *bias, const int in_c, const int in_h,
    const int in_w, const int out_c, const int out_h, const int out_w,
    const int ker_size, const int group,
    const int pad, const int stride, const int bias_en, float *output) {
  int imap_offset, omap_offset;

  for (int g = 0; g < group; ++g) {
    imap_offset = g * (in_c / group);
    omap_offset = g * (out_c / group);
      int s = 0;
      while (s < out_c / group) {
        int in_row = -pad;
        for (int out_row = 0; out_row < out_h; ++out_row) {
          int in_col = -pad;
          for (int out_col = 0; out_col < out_w; ++out_col) {
            register float sum = 0.0;
            for (int imap = 0; imap < in_c / group; ++imap) {
              int in_addr_base = (imap_offset + imap) * in_h
                  + in_row;
              int wt_addr_base = ((omap_offset + s) * in_c
                  / group + imap);
              for (int kr = 0; kr < ker_size; ++kr) {

                int wt_addr0 = (wt_addr_base * ker_size + kr)
                    * ker_size;
                int in_addr0 = (in_addr_base + kr) * in_w
                    + in_col;

                for (int kc = 0; kc < ker_size; ++kc) {
                  if (is_a_ge_zero_and_a_lt_b(in_row + kr,
                      in_h)
                      & is_a_ge_zero_and_a_lt_b(
                          in_col + kc, in_w)) {

                    int in_addr = in_addr0 + kc;
                    int wt_addr = wt_addr0 + kc;
                    sum += weight[wt_addr] * input[in_addr];
                  }
                }
              }
            }
            if (bias_en) {
              sum += bias[omap_offset + s];
            }
            int out_addr = ((omap_offset + s) * out_h + out_row)
                * out_w + out_col;
            output[out_addr] = sum;
            in_col += stride;
          }
          in_row += stride;
        }
        s++;
    }
  }
}
