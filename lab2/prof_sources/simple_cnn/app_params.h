
#ifndef __APP_PARAMS_H__
#define __APP_PARAMS_H__

#include "image.h"
#include <stdint.h>

/* ============================= START OF RUN CONFIGURATION ============================ */
#define EMBEDDED                         /* uncomment to run in Zynq */
//#define USE_GEMM                         /* uncomment to use GEMM */
//#define PRINT_IMAGE                      /* uncomment to print input images to console */
// #define PRINT_TIME_PER_LAYER             /* uncomment to print elapsed time per layer on zynq */
#define PRINT_TOTAL_TIME                 /* uncomment to print elapsed time per layer on zynq */
#define FIRST_IMAGE_TO_CLASSIFY 1        /* first image of the test set to classify */
#define NUMBER_OF_IMAGES_TO_CLASSIFY 10  /* number of images to classify sequentially */
/* ============================== END OF RUN CONFIGURATION ============================= */

/* ============================ START OF MODEL CONFIGURATION =========================== */
#define WEIGHTS_FILENAME "weights.bin"   /* file where the weights are stored */
#define WEIGHTS_Q15_FILENAME "weightsq15.bin" /* file where the q15 weights are stored */
#define CONV_KERNEL_SIZE 3               /* size of the convolution kernel */
#define CONV_OFM_NUMBER 16               /* number of OFMs of convolutional layer */
#define POOL_KERNEL_SIZE 2               /* size of pooling kernel */
#define POOL_STRIDE 2                    /* stride of pooling operation */
/* ============================== END OF RUN CONFIGURATION ============================= */

/* =====================================================================================
 * ================ PARAMETERS AUTOMATICALLY GENERATED BELOW THIS LINE! ================
 * ===================================================================================== */

#define CONV_OUTPUT_HEIGHT (IMAGE_HEIGHT - CONV_KERNEL_SIZE + 1)
#define CONV_OUTPUT_WIDTH (IMAGE_WIDTH - CONV_KERNEL_SIZE + 1)
#define POOL_OUTPUT_HEIGHT CONV_OUTPUT_HEIGHT / POOL_KERNEL_SIZE
#define POOL_OUTPUT_WIDTH CONV_OUTPUT_WIDTH / POOL_KERNEL_SIZE
#define CONV_LAYER_WEIGHTS CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * CONV_OFM_NUMBER * IMAGE_CHANNELS
#define CONV_LAYER_BIASES CONV_OFM_NUMBER
#define CONV_LAYER_PARAMS CONV_LAYER_WEIGHTS + CONV_LAYER_BIASES
#define FC_LAYER_WEIGHTS POOL_OUTPUT_WIDTH * POOL_OUTPUT_HEIGHT * CONV_OFM_NUMBER * N_CLASSES
#define FC_LAYER_0_WEIGHTS POOL_OUTPUT_WIDTH * CONV_OFM_NUMBER * N_CLASSES
#define FC_LAYER_BIASES N_CLASSES
#define FC_LAYER_PARAMS FC_LAYER_WEIGHTS + FC_LAYER_BIASES
#define TOTAL_PARAMS (CONV_LAYER_PARAMS + FC_LAYER_PARAMS)
#define TOTAL_PADDED_PARAMS (TOTAL_PARAMS + FC_LAYER_0_WEIGHTS)
#define CONV_OUTPUT_SIZE CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH * CONV_OFM_NUMBER
#define POOL_OUTPUT_SIZE POOL_OUTPUT_HEIGHT * POOL_OUTPUT_WIDTH * CONV_OFM_NUMBER
#define HW_MATRIX_OUT_HEIGHT POOL_OUTPUT_HEIGHT
#define HW_MATRIX_OUT_WIDTH (POOL_OUTPUT_WIDTH + 1)
#define HW_MATRIX_OUT_SIZE HW_MATRIX_OUT_HEIGHT * HW_MATRIX_OUT_WIDTH * CONV_OFM_NUMBER

#define FLT_MAX 3.402823466e+38F

typedef struct addresses
{
  unsigned char *ch_images; /* images data region */
  float *fp_params;         /* network parameters data region */
  float *matA;              /* auxiliary matrix A to implement 3D convolution as a matrix multiplication */
  float *matCv;             /* output of convolutional layer before adding bias */
  float *matCbias;          /* output of convolutional layer after adding bias */
  float *matCrelu;          /* output of ReLU */
  float *matCpool;          /* output of pooling layer */
  float *matConn;           /* output of fully connected layer before adding bias */
  float *matConnB;          /* output of fully connected layer after adding bias */
  float *matSoftM;          /* output of softmax layer */
  float *vecSoftM;          /* output of softmax layer */
  int16_t *int_params;
  volatile int32_t *matConvPool[2];
  int matConvPoolInd;
  int64_t *matGemm;
  float *matSoftMax;
  float *vecSoftMax;
  float *fp_images;          /* scaled floating-point image to be processed */
  int16_t *int_images;
  uint8_t *nextImage;
} addresses;


#ifdef EMBEDDED
#include "xiltimer.h"
#endif // EMBEDDED

#endif // __APP_PARAMS_H__
