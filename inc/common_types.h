// @file common_types.h
//
//  \date Created on: Sep 22, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//

#ifndef INC_COMMON_TYPES_H_
#define INC_COMMON_TYPES_H_

typedef enum {
  LAYOUT_NCHW,
  LAYOUT_NHWC,
  LAYOUT_HWNC,
  LAYOUT_CHWN
}TensorLayout;

typedef struct {
  int n;
  int c;
  int h;
  int w;
}TensorDim;

#endif  // INC_COMMON_TYPES_H_
