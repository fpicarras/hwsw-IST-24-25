
#ifndef __CNN_SW_HW_H__
#define __CNN_SW_HW_H__

#include "app_params.h"
#include <stdbool.h>

#ifdef EMBEDDED
#include "xaxidma.h"
#endif

/**
 * Executes all layers of the CNN sequentially and returns the predicted class for a given sample.
 * @return Predicted class for a given sample
 */
int predict_class_hw_sw(addresses * addr, XAxiDma *dma);

void predict_images_hw_sw(addresses * addr);

/**
 * Executes convolution + maxpool layer on input using the IP.
 */
void forward_convolutional_layer_hw(addresses * addr, XAxiDma *dma);

/**
 * Executes fully-connected layer on output of max pooling layer.
 */
void forward_connected_layer_int(const int32_t *X, const int16_t * int_params, int64_t * Y);

int forward_softmax_layer_int(const int64_t* A, float* B);

#endif // __CNN_SW_HW_H__
