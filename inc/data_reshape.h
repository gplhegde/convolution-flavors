// @file data_reshape.h
//
//  \date Created on: Sep 24, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//

#ifndef INC_DATA_RESHAPE_H_
#define INC_DATA_RESHAPE_H_

void NCHW2HWNC(const float *nchw_data, int N, int C, int H, int W,
               float *hwnc_data);

void NCHW2HWCN(const float *nchw_data, int N, int C, int H, int W,
               float *hwcn_data);

void NCHW2CHWN(const float *nchw_data, int N, int C, int H, int W,
               float *chwn_data);

void NCHW2NHWC(const float *nchw_data, int N, int C, int H, int W,
               float *nhwc_data);

#endif  // INC_DATA_RESHAPE_H_
