
#ifndef __CNN_SW_H__
#define __CNN_SW_H__

#include "app_params.h"

/**
 * Executes all layers of the CNN sequentially and returns the predicted class for a given sample.
 * @return Predicted class for a given sample
 */
int predict_class_sw(const float * image, addresses * addr);

/**
 * Computes auxiliary matrix to perform convolution as matrix multiplication using GEMM kernel.
 * Matrix A has 4 dimensions:
 * (I, J): Planar coordinates of one OFM
 *      K: OFM
 *     XY: Combination of Planar coordinate of OFM with kernel element
 */
void compute_matrixA(const float* image, float * A);

/**
 * Performs matrix convolution without using GEMM.
 * @param image_in Matrix containing normalized values of the pixels
 * @param weights Matrix containing elements of the kernel
 * @param bias Bias
 * @param image_out Matrix containing resulting convolution OFMs
 */
void sw_convolution_3D(const float *image_in, const float *weights, float bias, float *image_out);

void forward_gemm_layer(const float * image, float * A, const float * fp_params, float * C);

/**
 * Executes convolution layer on input either using GEMM or plain convolution.
 */
void forward_convolutional_layer(const float * fp_params, const float * image, float * conv_out);

/**
 * Executes pooling layer on output of convolutional layer.
 */
void forward_max_pool_layer(const float* A, float *B);

/**
 * Executes fully-connected layer on output of max pooling layer.
 */
void forward_connected_layer(const float * X, const float * fp_params, float * Y);

/**
 * Executes softmax operation on output of fully-connected layer to normalize it to a probability distribution
 * and returns most likely class which corresponds to position of the largest value.
 * @return Position of the largest value in matConnB
 */
int forward_softmax_layer(const float* A, float* B);

#endif // __CNN_SW_H__
