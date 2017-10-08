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
#define K_VEC (2)
#define Q_VEC (4)
#define C_VEC (3)
#define S_VEC (3)
#define W_VEC (Q_VEC + S_VEC - 1)

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

typedef struct {
  float d[2];
}float2;

typedef struct {
  float d[3];
}float3;

typedef union {
    struct {
        float2 f0;
        float2 f1;
        float2 f2;
    };
    float2 d[3];
} float2x3;

typedef struct {
  float d[6];
} float6;

typedef struct {
  float d[W_VEC];
}float_Wvec;

typedef struct {
  float d[C_VEC][W_VEC];
}float_CvecXWvec;

typedef struct {
  float d[C_VEC];
}float_Cvec;

typedef struct {
  float d[C_VEC][S_VEC];
}float_CvecXSvec;

typedef struct {
  float d[K_VEC];
}float_Kvec;

typedef struct {
  float d[Q_VEC];
}float_Qvec;

typedef struct {
  float d[K_VEC][W_VEC];
}float_KvecXWvec;

typedef struct {
  float d[K_VEC][Q_VEC];
}float_KvecXQvec;

typedef struct {
  // mean, scale
  float d[2][K_VEC];
}float_2xKvec;

typedef  float3 float_Svec;
#endif  // INC_COMMON_TYPES_H_
