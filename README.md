# Convolution Flavors

This project contains different types of implementations of convolution layer used in Convolutional Neural Networks.

# Notations

N : Batch size

C : No of input channels

H : Input height

W : Input width

M : No of output channels

K : Kernel size

# Description
The current implementations include

1. **Image to Column** - Standard convolution used in frameworks such as Caffe. This method requires an extra buffer space of HxWxKxKxC to store the im2col matrix. The matrix multiplication is done using SGEMM.

2. **Image to Row** - This is same as image to column method except the strided input patches are unrolled and stored in a row of the matrix. The extra buffer space requirement is same as im2col method.

3. **Kernel to Row** - This method is based on the trick that a KxK convolution can be computed using K.K 1x1 convolutions and then shifting and adding the resulting partial outputs. The extra buffer space required for this is MxHxW.

4. **Kernel to Column** - This is the counterpart of kn2row method.

Methods 3 and 4 are introduced in [this](https://arxiv.org/pdf/1709.03395.pdf) research paper. Refer to it for more info.

5. **Vectored Convolution** - This implementation uses direct convolution without using any creation of patch matrix. It vectorizes the computation in C, M, W and K dimensions which can be changed to finetune the performance for different platforms.

6. **1D Winograd Convolution** - This implementation is based on 1D Winograd convolution algorithm. This approach saves some costly multiplications at the expense of few more cheaper addtions.

Methods 5 and 6 are vectorized as per [this](https://arxiv.org/abs/1701.03534) paper from Intel which is on Deep learning accelerator using OpenCL on Arria10 FPGA.

Refer to [Fast Algorithms for Convolutional Neural Networks](https://arxiv.org/abs/1509.09308) for details on Winograd appraoch to realize convolution layers.
