/*
 *
 *
 * The functions in this file are directly taken from
 https://github.com/s5248/cppconvnet
 The license and the copying notice are replicated below.
----------
LICENSE
----------
Copyright (c) 2015, BiaoZhi Huang.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of cppconvnet nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---------------
COPYING
---------------

Copyright (c) 2014 The MatConvNet team.
Copyright (c) 2015 BiaoZhi Huang.
All rights reserved.

Redistribution and use in source and binary forms are permitted
provided that the above copyright notice and this paragraph are
duplicated in all such forms and that any documentation,
advertising materials, and other materials related to such
distribution and use acknowledge that the software was developed
by the <organization>. The name of the
<organization> may not be used to endorse or promote products derived
from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 */


#include "common_types.h"
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>
#include "utils.h"

static inline int floor_divide(int a, int b) {
  if (a >= 0) return a / b;
  else return (a - b + 1) / b;
}

static inline int ceil_divide(int a, int b) {
  if (a >= 0) return (a + b - 1) / b;
  else return a / b;
}

static inline int min (int a, int b) {
  return a < b ? a : b;
}

static void CppConvnetIm2Row(float* stacked, const float* data, int numPatchesX,
                      int numPatchesY, int numRows, const TensorDim input_dim,
                      const TensorDim filter_dim,
                      int stride, int pad)
{

  /*
  Fill a row of the stacked image at a time. Since patches are stored
  aint the columns, scanning a row menas visiting all patche once.
  Each row corresponds to a particular offset within each patch.

  In this manner, as we fill a row
  we tend to access spatially adiacent elements
  in the input image, particulary for small strides.
  */
  float *im2row = stacked;
  // numRows = KxKXC, numPatchesX = H, numPatchesY = W
  for (int row = 0; row < numRows; ++row) {
    /*
    Get the patch offset corresponding to this row of the stacked
    image.
    */
    int u = row;
    int v = u / filter_dim.w;
    int z = v / filter_dim.h; // filter channel no
    u %= filter_dim.w; // filter col no
    v %= filter_dim.h; // filter row no

    /*
    Filling this row amounts to visiting all the pixels in the input
    image that appear at a given offset in the outut patches. Accounting
    for the subsampling of the output patches and input padding,
    these pixels are given by

    x_data(x) = x * m_stride[1] + u - m_pad[2],  0 <= x < numPatchesX
    y_data(y) = y * m_stride[0] + v - m_pad[0],   0 <= y < numPatchesY
    z_data(z) = z.

    Here (x,y) are the spatial indexes of the output patches. Depending
    on the padding, some of these values will read pixels outised
    the input image, which should default to 0. In particular, x lands
    on a x_data(x) within the image if x0 <= x < x1 where:

    x_data(x) >= 0 <=> x >= (m_pad[2] - u) / stride
    <=> x >= ceil((m_pad[2] - u) / stride) = x0
    x_data(x) <= m_inputDims[1]-1 <=> x <= (m_inputDims[1]-1 + m_pad[2] - u) / stride
    <=> x <= floor((m_inputDims[1]-1 + m_pad[2] - u) / stride)
    <=> x <  floor((m_inputDims[1]-1 + m_pad[2] - u) / stride) + 1 = x1

    and the same for y. Note that, while usually x0 <= x1, there are
    special cases for which x1 < x0. This is accounted for in the loops
    below.
    */

    int x0 = min(numPatchesX, (int)ceil_divide(pad - u, stride));
    int y0 = min(numPatchesY, (int)ceil_divide(pad - v, stride));
    int x1 = min(numPatchesX, (int)floor_divide(input_dim.w - 1 + pad - u, stride) + 1);
    int y1 = min(numPatchesY, (int)floor_divide(input_dim.h - 1 + pad - v, stride) + 1);
    int x;
    int y;

    for (y = 0; y < y0; ++y) {
      for (x = 0; x < numPatchesX; ++x) {
        *stacked++ = 0;
      }
    }
    for (; y < y1; ++y) {
      for (x = 0; x < x0; ++x) {
        *stacked++ = 0;
      }
      int y_data = y * stride + v - pad;
      int x_data = x * stride + u - pad;
      float const * b = data + (z * input_dim.h + y_data) * input_dim.w + x_data;
      for (; x < x1; ++x) {
        *stacked++ = *b;
        b += stride;
      }
      for (; x < numPatchesX; ++x) {
        *stacked++ = 0;
      }
    }
    for (; y < numPatchesY; ++y) {
      for (x = 0; x < numPatchesX; ++x) {
        *stacked++ = 0;
      }
    }
  }

  PrintMat("im2row", im2row, 1, numRows*numPatchesX*numPatchesY, CblasRowMajor);

}

// Not working for group != 1
bool CppConvnetConvLayer(const float *in_data, const float *filters,
                         const float *bias, TensorDim in_dim,
                         TensorDim filt_dim, int stride, int pad, int group,
                         float *output)
{
  float dataMult=1.0;
  float outputMult=0.0;
  TensorDim out_dim;
  out_dim.w = (in_dim.w + (pad + pad) - filt_dim.w) / stride + 1;
  out_dim.h = (in_dim.h + (pad + pad) - filt_dim.h) / stride + 1;
  out_dim.c = filt_dim.n;
  out_dim.n = in_dim.n;
  int m_biasDims = out_dim.c;

  //int numGroups = in_dim.c / filt_dim.c;// assume = 1
  bool fullyConnectedMode = (out_dim.w == 1 &&
      out_dim.h == 1 &&
    stride == 1 &&
    pad == 0 &&
    group == 1);
  if (fullyConnectedMode){
    return false;
  }
  printf("---------Inputs------\n");
  PrintTensor(in_data, in_dim);
  printf("---------------------\n");
  printf("-------Filters-------\n");
  PrintTensor(filters, filt_dim);
  int numFiltersPerGroup = filt_dim.n / group; // M
  int numOutputPixels = out_dim.w * out_dim.h; // H*W
  int filtersVolume = filt_dim.w * filt_dim.h * filt_dim.c/group; // K*K*C
  int tempVolume = numOutputPixels * filtersVolume * group; // H*W*K*K*C --> looks to be size of im2row buffer
  float* tempMemory = malloc( sizeof(float) *tempVolume); // im2row buffer?
  float* tempOnes = malloc(sizeof(float)*numOutputPixels); // buffer of size H*W all set to 1
  for (int i = 0; i < numOutputPixels; i++) tempOnes[i] = 1.0;


  for (int image = 0; image < in_dim.n; ++image) {
    int dataOffset = (in_dim.w * in_dim.h * in_dim.c) * image;// pointer to batch of images
    int outputOffset = (out_dim.w * out_dim.h * out_dim.c) * image;// pointer to batch of output maps.
    //under sample
    CppConvnetIm2Row(tempMemory, in_data + dataOffset, out_dim.w, out_dim.h,
                     filtersVolume*group, in_dim, filt_dim, stride, pad);

    for (int g = 0; g < group; ++g) {
      int filterGrpOffset = filtersVolume * numFiltersPerGroup * g;//0
      int tempGrpOffset = numOutputPixels * filtersVolume * g;//0
      int outputGrpOffset = numOutputPixels * numFiltersPerGroup * g;//0
      float alpha = dataMult;
      float beta = outputMult;
      //convolution
      // A -> MxK B-> KxN
      PrintMat("im2row", tempMemory + tempGrpOffset, numOutputPixels, filtersVolume, CblasColMajor);
      PrintMat("filters", filters + filterGrpOffset, filtersVolume, numFiltersPerGroup, CblasColMajor);
      //PrintMat("im2row", tempMemory, 1, filtersVolume* numOutputPixels, CblasColMajor);
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                  numOutputPixels/*m = HxW*/,
                  numFiltersPerGroup/*n = M*/, filtersVolume/*k = KxKxC/group*/,
                  alpha, tempMemory + tempGrpOffset, /*A*/
                  numOutputPixels/*lda*/,
                  filters + filterGrpOffset, // B
                  filtersVolume, // ldb
                  beta,
                  output + outputOffset + outputGrpOffset, // C
                  numOutputPixels // ldc
                  );

    }
    if (bias!=NULL) {
      float alpha = 1;
      float beta = 1;
      int k = 1;
      //add bias
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, numOutputPixels,
                  m_biasDims, k, alpha, tempOnes, numOutputPixels,
                  bias, k, beta, output + outputOffset, numOutputPixels);
    }
  }
  free(tempMemory);
  free(tempOnes);
  return true;
}
