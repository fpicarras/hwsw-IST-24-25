
#ifndef __AXIL_CNN_H__
#define __AXIL_CNN_H__

#include <ap_int.h>
#include <hls_stream.h>
#include "ap_axi_sdata.h"

/* ========================== START OF TEST SET CONFIGURATION ========================== */

#define N_IMAGES 200                 /* number of images in the binary file */
#define IMAGE_HEIGHT 6              /* width of the images */
#define IMAGE_WIDTH IMAGE_HEIGHT     /* height of the images */
#define IMAGE_CHANNELS 3             /* number of channels (red + green + blue) */
#define N_CLASSES 10                 /* number of possible classes */

/* =========================== END OF TEST SET CONFIGURATION =========================== */

/* ============================ START OF MODEL CONFIGURATION =========================== */
#define CONV_KERNEL_SIZE 3               /* size of the convolution kernel */
#define CONV_OFM_NUMBER 2               /* number of OFMs of convolutional layer */
#define POOL_KERNEL_SIZE 2               /* size of pooling kernel */
#define POOL_STRIDE 2                    /* stride of pooling operation */
/* ============================== END OF RUN CONFIGURATION ============================= */

/* ============================ PL DATA CONFIGURATION =========================== */
#define WEIGHT_BIT_WIDTH 16
#define BIAS_BIT_WIDTH 16
#define IMAGE_BIT_WIDTH 8
#define DATA_BIT_WIDTH 32

typedef ap_int<DATA_BIT_WIDTH> data_t;
typedef ap_int<IMAGE_BIT_WIDTH> image_t;
typedef ap_int<WEIGHT_BIT_WIDTH> weight_t;
typedef ap_int<BIAS_BIT_WIDTH> bias_t;
typedef ap_int<WEIGHT_BIT_WIDTH + IMAGE_BIT_WIDTH + 7> accum_t;
typedef hls::axis<data_t, 0, 0, 0> strmio_t;
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
#define FC_LAYER_WEIGHTS POOL_OUTPUT_WIDTH * POOL_OUTPUT_HEIGHT * CONV_OFM_NUMBER * N_CLASSES
#define FC_LAYER_BIASES N_CLASSES
#define TOTAL_PARAMS (CONV_LAYER_WEIGHTS + CONV_LAYER_BIASES + FC_LAYER_WEIGHTS + FC_LAYER_BIASES)
#define IMAGES_PER_DATA (DATA_BIT_WIDTH/IMAGE_BIT_WIDTH)
#define WEIGHTS_PER_DATA (DATA_BIT_WIDTH/WEIGHT_BIT_WIDTH)
#define BIAS_PER_DATA (DATA_BIT_WIDTH/BIAS_BIT_WIDTH)

void axil_conv3D(hls::stream<strmio_t> &strm_in,
                 hls::stream<strmio_t> &strm_out);
#endif // __AXIL_CNN_H__
