
#ifndef __AXIL_CNN_H__
#define __AXIL_CNN_H__

#include <ap_int.h>
#include <hls_stream.h>
#include "ap_axi_sdata.h"

/* ========================== START OF TEST SET CONFIGURATION ========================== */

#define N_IMAGES 2                   /* number of images in the binary file */
#define IMAGE_HEIGHT 88              /* width of the images */
#define IMAGE_WIDTH IMAGE_HEIGHT     /* height of the images */
#define IMAGE_CHANNELS 3             /* number of channels (red + green + blue) */
#define N_CLASSES 10                 /* number of possible classes */
#define WEIGHTS_FILENAME "/home/joao-pedro/Documents/Tecnico/Y1-P4/HW/Labs/hwsw-IST-24-25/lab2/prof_sources/simple_cnn/weights.bin"   /* file where the weights are stored */
#define IMAGES_FILENAME "/home/joao-pedro/Documents/Tecnico/Y1-P4/HW/Labs/hwsw-IST-24-25/lab2/prof_sources/simple_cnn/images.bin" /* file where the images are stored */
// #define WEIGHTS_FILENAME "weights.bin"   /* file where the weights are stored */
// #define IMAGES_FILENAME  "images.bin" /* file where the images are stored */
/* =========================== END OF TEST SET CONFIGURATION =========================== */

/* ============================ START OF MODEL CONFIGURATION =========================== */
#define CONV_KERNEL_SIZE 3               /* size of the convolution kernel */
#define CONV_OFM_NUMBER 16               /* number of OFMs of convolutional layer */
#define POOL_KERNEL_SIZE 2               /* size of pooling kernel */
#define POOL_STRIDE 2                    /* stride of pooling operation */
/* ============================== END OF RUN CONFIGURATION ============================= */

/* ============================ PL DATA CONFIGURATION =========================== */
#define WEIGHT_BIT_WIDTH 16
#define BIAS_BIT_WIDTH 16
#define IMAGE_BIT_WIDTH 16
#define PIXEL_BIT_WIDTH 16
#define MAXPOOL_BIT_WIDTH 32
#define OUTPUT_BIT_WIDTH 64
#define INPUT_BIT_WIDTH 64
#define INTEGER_BIT_WIDTH 5
#define ACCUM_BIT_WIDTH WEIGHT_BIT_WIDTH + IMAGE_BIT_WIDTH + INTEGER_BIT_WIDTH - 1

typedef ap_int<INPUT_BIT_WIDTH> input_t;
typedef ap_int<IMAGE_BIT_WIDTH> image_t;
typedef ap_int<WEIGHT_BIT_WIDTH> weight_t;
typedef ap_int<BIAS_BIT_WIDTH> bias_t;
typedef ap_int<OUTPUT_BIT_WIDTH> output_t;
typedef ap_int<ACCUM_BIT_WIDTH> accum_t;
typedef ap_int<MAXPOOL_BIT_WIDTH> maxpool_t;
typedef hls::axis<input_t, 0, 0, 0> strmin_t;
typedef hls::axis<output_t, 0, 0, 0> strmout_t;
/* ============================== PL DATA CONFIGURATION ============================= */

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
#define FC_LAYER_BIASES N_CLASSES
#define TOTAL_PARAMS (CONV_LAYER_WEIGHTS + CONV_LAYER_BIASES + FC_LAYER_WEIGHTS + FC_LAYER_BIASES)
#define PIXEL_PER_DATA (INPUT_BIT_WIDTH/PIXEL_BIT_WIDTH)
#define WEIGHTS_PER_DATA (INPUT_BIT_WIDTH/WEIGHT_BIT_WIDTH)
#define BIAS_PER_DATA (INPUT_BIT_WIDTH/BIAS_BIT_WIDTH)
#define MAXPOOLS_PER_DATA (OUTPUT_BIT_WIDTH/MAXPOOL_BIT_WIDTH)
#define IMAGE_SIZE IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS
#define CONV_OUTPUT_SIZE CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH * CONV_OFM_NUMBER
#define POOL_OUTPUT_SIZE POOL_OUTPUT_HEIGHT * POOL_OUTPUT_WIDTH * CONV_OFM_NUMBER
#define HW_MATRIX_OUT_HEIGHT POOL_OUTPUT_HEIGHT
#define HW_MATRIX_OUT_WIDTH (POOL_OUTPUT_WIDTH + 1)
#define HW_MATRIX_OUT_SIZE HW_MATRIX_OUT_HEIGHT * HW_MATRIX_OUT_WIDTH * CONV_OFM_NUMBER
#define FRAC_BIT_WIDTH MAXPOOL_BIT_WIDTH - INTEGER_BIT_WIDTH - 1

void axil_conv3D(hls::stream<strmin_t> &strm_in,
                 hls::stream<strmout_t> &strm_out);
#endif // __AXIL_CNN_H__
