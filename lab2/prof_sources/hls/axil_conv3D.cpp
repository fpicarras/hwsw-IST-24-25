
#include "axil_conv3D.h"

void axil_conv3D(hls::stream<strmin_t> &strm_in,
                 hls::stream<strmout_t> &strm_out) {

#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS interface axis port=strm_in
#pragma HLS INTERFACE axis port=strm_out

static image_t image_red[IMAGE_HEIGHT*IMAGE_WIDTH + 2];
static image_t image_green[IMAGE_HEIGHT*IMAGE_WIDTH + 2];
static image_t image_blue[IMAGE_HEIGHT*IMAGE_WIDTH + 2];
static weight_t weights[IMAGE_CHANNELS*CONV_OFM_NUMBER*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE];
static bias_t bias[CONV_OFM_NUMBER];
static bool weights_ready = false;

  strmin_t tmp;

    /* Bias Stream */
  if(!weights_ready) {
    loop_bias:
    for(int i = 0; i < CONV_OFM_NUMBER; i += BIAS_PER_DATA) {
      tmp = strm_in.read();
      bias[i]   = (bias_t)tmp.data.range(15, 0);
      bias[i+1] = (bias_t)tmp.data.range(31, 16);
    }

    /* Weights Stream */
    loop_weights:
    for(int i = 0; i < IMAGE_CHANNELS*CONV_OFM_NUMBER*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE; i += WEIGHTS_PER_DATA) {
      tmp = strm_in.read();
      weights[i]   = (weight_t)tmp.data.range(15, 0);
      weights[i+1] = (weight_t)tmp.data.range(31, 16);
    }
  }

  weights_ready = true;
  /* Input Image Stream */
  loop_red: 
  for(int i = 0; i < IMAGE_HEIGHT*IMAGE_WIDTH; i += PIXEL_PER_DATA) {
    tmp = strm_in.read();
    image_red[i] = tmp.data.range(15, 0);
    image_red[i+1] = tmp.data.range(31, 16);
  }
  loop_green: 
  for(int i = 0; i < IMAGE_HEIGHT*IMAGE_WIDTH; i += PIXEL_PER_DATA) {
    tmp = strm_in.read();
    image_green[i] = tmp.data.range(15, 0);
    image_green[i+1] = tmp.data.range(31, 16);
  }
  loop_blue: 
  for(int i = 0; i < IMAGE_HEIGHT*IMAGE_WIDTH; i += PIXEL_PER_DATA) {
    tmp = strm_in.read();
    image_blue[i] = tmp.data.range(15, 0);
    image_blue[i+1] = tmp.data.range(31, 16);
  }

  loop_conv:
  for(int l = 0; l < CONV_OFM_NUMBER; l++) {
    loop_i:
    for (int i = 0; i < CONV_OUTPUT_HEIGHT; i+=2) {
      loop_j:
      for (int j = 0; j < CONV_OUTPUT_WIDTH; j+=4) {
        // Accumulators for the 8 different positions
        accum_t acc0_r = 0, acc1_r = 0, acc2_r = 0, acc3_r = 0, acc4_r = 0, acc5_r = 0, acc6_r = 0, acc7_r = 0;
        accum_t acc0_g = 0, acc1_g = 0, acc2_g = 0, acc3_g = 0, acc4_g = 0, acc5_g = 0, acc6_g = 0, acc7_g = 0;
        accum_t acc0_b = 0, acc1_b = 0, acc2_b = 0, acc3_b = 0, acc4_b = 0, acc5_b = 0, acc6_b = 0, acc7_b = 0;

        loop_k:
        for (int k = 0; k < CONV_KERNEL_SIZE; k++) {
  #pragma HLS PIPELINE
          // Indices are initialized here to highlight
          // the incremental counting in the inner loop
          int kernel_1d_idx = k * CONV_KERNEL_SIZE + l*(CONV_KERNEL_SIZE*CONV_KERNEL_SIZE*IMAGE_CHANNELS); /* start of kernel row */
          int image_1d_idx_base = (i + k) * IMAGE_WIDTH + j; /* start of input row */
          loop_x:
          for (int x = 0; x < CONV_KERNEL_SIZE; x++) {
            // Kernel values
            int kernel_1d_idx_r = (kernel_1d_idx + x);
            int kernel_1d_idx_g = (kernel_1d_idx + x) + CONV_KERNEL_SIZE*CONV_KERNEL_SIZE;
            int kernel_1d_idx_b = (kernel_1d_idx + x) + 2*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE;
            weight_t weight_r  = weights[kernel_1d_idx_r];
            weight_t weight_g  = weights[kernel_1d_idx_g];
            weight_t weight_b  = weights[kernel_1d_idx_b];
            // Image values
            // x - - -
            // - - - -
            int image_1d_idx0 = image_1d_idx_base + x;
            image_t image0_r = image_red[image_1d_idx0];
            image_t image0_g = image_green[image_1d_idx0];
            image_t image0_b = image_blue[image_1d_idx0];
            acc0_r += weight_r * image0_r;
            acc0_g += weight_g * image0_g;
            acc0_b += weight_b * image0_b;
            
            // - x - -
            // - - - -
            int image_1d_idx1 = image_1d_idx_base + x +1;
            image_t image1_r = image_red[image_1d_idx1];
            image_t image1_g = image_green[image_1d_idx1];
            image_t image1_b = image_blue[image_1d_idx1];
            acc1_r += weight_r * image1_r;
            acc1_g += weight_g * image1_g;
            acc1_b += weight_b * image1_b;
            
            // - - - -
            // x - - -
            int image_1d_idx2 = image_1d_idx_base + x + IMAGE_WIDTH;
            image_t image2_r = image_red[image_1d_idx2];
            image_t image2_g = image_green[image_1d_idx2];
            image_t image2_b = image_blue[image_1d_idx2];
            acc2_r += weight_r * image2_r;
            acc2_g += weight_g * image2_g;
            acc2_b += weight_b * image2_b;
            
            // - - - -
            // - x - -
            int image_1d_idx3 = image_1d_idx_base + x + IMAGE_WIDTH + 1;
            image_t image3_r = image_red[image_1d_idx3];
            image_t image3_g = image_green[image_1d_idx3];
            image_t image3_b = image_blue[image_1d_idx3];
            acc3_r += weight_r * image3_r;
            acc3_g += weight_g * image3_g;
            acc3_b += weight_b * image3_b;

            // - - x -
            // - - - -
            int image_1d_idx4 = image_1d_idx_base + x + 2;
            image_t image4_r = image_red[image_1d_idx4];
            image_t image4_g = image_green[image_1d_idx4];
            image_t image4_b = image_blue[image_1d_idx4];
            acc4_r += weight_r * image4_r;
            acc4_g += weight_g * image4_g;
            acc4_b += weight_b * image4_b;

            // - - - x
            // - - - -
            int image_1d_idx5 = image_1d_idx_base + x + 3;
            image_t image5_r = image_red[image_1d_idx5];
            image_t image5_g = image_green[image_1d_idx5];
            image_t image5_b = image_blue[image_1d_idx5];
            acc5_r += weight_r * image5_r;
            acc5_g += weight_g * image5_g;
            acc5_b += weight_b * image5_b;

            // - - - -
            // - - x -
            int image_1d_idx6 = image_1d_idx_base + x + IMAGE_WIDTH + 2;
            image_t image6_r = image_red[image_1d_idx6];
            image_t image6_g = image_green[image_1d_idx6];
            image_t image6_b = image_blue[image_1d_idx6];
            acc6_r += weight_r * image6_r;
            acc6_g += weight_g * image6_g;
            acc6_b += weight_b * image6_b;

            // - - - -
            // - - - x
            int image_1d_idx7 = image_1d_idx_base + x + IMAGE_WIDTH + 3;
            image_t image7_r = image_red[image_1d_idx7];
            image_t image7_g = image_green[image_1d_idx7];
            image_t image7_b = image_blue[image_1d_idx7];
            acc7_r += weight_r * image7_r;
            acc7_g += weight_g * image7_g;
            acc7_b += weight_b * image7_b;
          }
        }

        accum_t bia = ((accum_t)bias[l] << (ACCUM_BIT_WIDTH - INTEGER_BIT_WIDTH - WEIGHT_BIT_WIDTH));
        accum_t acc0 = acc0_r + acc0_g + acc0_b + bia;
        accum_t acc1 = acc1_r + acc1_g + acc1_b + bia;
        accum_t acc2 = acc2_r + acc2_g + acc2_b + bia;
        accum_t acc3 = acc3_r + acc3_g + acc3_b + bia;
        accum_t acc4 = acc4_r + acc4_g + acc4_b + bia;
        accum_t acc5 = acc5_r + acc5_g + acc5_b + bia;
        accum_t acc6 = acc6_r + acc6_g + acc6_b + bia;
        accum_t acc7 = acc7_r + acc7_g + acc7_b + bia;

        accum_t acc_first = (acc0 > acc1) ? acc0 : acc1;
        acc_first = (acc_first > acc2) ? acc_first : acc2;
        acc_first = (acc_first > acc3) ? acc_first : acc3;

        accum_t acc_second = (acc4 > acc5) ? acc4 : acc5;
        acc_second = (acc_second > acc6) ? acc_second : acc6;
        acc_second = (acc_second > acc7) ? acc_second : acc7;

        maxpool_t acc_sat0, acc_sat1;
        acc_first = acc_first >> (ACCUM_BIT_WIDTH - MAXPOOL_BIT_WIDTH);
        /* Relu */
        if (acc_first < 0)
          acc_sat0 = 0;
        else
          acc_sat0 = acc_first;

        acc_second = acc_second >> (ACCUM_BIT_WIDTH - MAXPOOL_BIT_WIDTH);
        /* Relu */
        if (acc_second < 0)
          acc_sat1 = 0;
        else
          acc_sat1 = acc_second;

        strmout_t chunk_out;
        chunk_out.last = ((i == CONV_OUTPUT_HEIGHT - 2) && (j == CONV_OUTPUT_WIDTH - 2) && (l == CONV_OFM_NUMBER - 1));
        chunk_out.data.range(31, 0)  = acc_sat0;
        chunk_out.data.range(63, 32) = acc_sat1;
        chunk_out.keep = 0xFF;
        chunk_out.strb = 0xFF;
        strm_out.write(chunk_out);
      }
    }
  }
}
