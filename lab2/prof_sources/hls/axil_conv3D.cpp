
#include "axil_conv3D.h"

void axil_conv3D(hls::stream<strmin_t> &strm_in,
                 hls::stream<strmout_t> &strm_out) {

#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS interface axis port=strm_in
#pragma HLS INTERFACE axis port=strm_out

static image_t image_red[IMAGE_HEIGHT*IMAGE_WIDTH];
static image_t image_green[IMAGE_HEIGHT*IMAGE_WIDTH];
static image_t image_blue[IMAGE_HEIGHT*IMAGE_WIDTH];
static data_t weights[IMAGE_CHANNELS*CONV_OFM_NUMBER*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE/WEIGHTS_PER_DATA];
static data_t bias[CONV_OFM_NUMBER/BIAS_PER_DATA];
static bool weights_ready = false;

  float tmp0, tmp1, tmp2, tmp3;
  int tmp0_i, tmp1_i, tmp2_i, tmp3_i;

  strmin_t tmp;
  /* Input Image Stream */

    /* Bias Stream */
  if(!weights_ready) {
    loop_bias:
    for(int i = 0; i < CONV_OFM_NUMBER/BIAS_PER_DATA; i ++) {
      tmp = strm_in.read();
      bias[i] = tmp.data;
    }

    /* Weights Stream */
    loop_weights:
    for(int i = 0; i < IMAGE_CHANNELS*CONV_OFM_NUMBER*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE/WEIGHTS_PER_DATA; i ++) {
      tmp = strm_in.read();
      weights[i] = tmp.data;
    }
  }

  weights_ready = true;

  loop_red: 
  for(int i = 0; i < IMAGE_HEIGHT*IMAGE_WIDTH; i +=PIXEL_PER_DATA) {
    tmp = strm_in.read();

    tmp0 = ((float)tmp.data.range(7, 0) / 255.0f - 0.5f) / 0.5f;
    tmp1 = ((float)tmp.data.range(15, 8) / 255.0f - 0.5f) / 0.5f;
    tmp2 = ((float)tmp.data.range(23, 16) / 255.0f - 0.5f) / 0.5f;
    tmp3 = ((float)tmp.data.range(31, 24) / 255.0f - 0.5f) / 0.5f;

    image_red[i]   = (image_t)(tmp0 * (float)(1 << 15) + 0.5f);
    image_red[i+1] = (image_t)(tmp1 * (float)(1 << 15) + 0.5f);
    image_red[i+2] = (image_t)(tmp2 * (float)(1 << 15) + 0.5f);
    image_red[i+3] = (image_t)(tmp3 * (float)(1 << 15) + 0.5f);
  }
  loop_green: 
  for(int i = 0; i < IMAGE_HEIGHT*IMAGE_WIDTH; i +=PIXEL_PER_DATA) {
    tmp = strm_in.read();

    tmp0 = ((float)tmp.data.range(7, 0) / 255.0f - 0.5f) / 0.5f;
    tmp1 = ((float)tmp.data.range(15, 8) / 255.0f - 0.5f) / 0.5f;
    tmp2 = ((float)tmp.data.range(23, 16) / 255.0f - 0.5f) / 0.5f;
    tmp3 = ((float)tmp.data.range(31, 24) / 255.0f - 0.5f) / 0.5f;

    image_green[i]   = (image_t)(tmp0 * (float)(1 << 15) + 0.5f);
    image_green[i+1] = (image_t)(tmp1 * (float)(1 << 15) + 0.5f);
    image_green[i+2] = (image_t)(tmp2 * (float)(1 << 15) + 0.5f);
    image_green[i+3] = (image_t)(tmp3 * (float)(1 << 15) + 0.5f);
  }
  loop_blue: 
  for(int i = 0; i < IMAGE_HEIGHT*IMAGE_WIDTH; i +=PIXEL_PER_DATA) {
    tmp = strm_in.read();

    tmp0 = ((float)tmp.data.range(7, 0) / 255.0f - 0.5f) / 0.5f;
    tmp1 = ((float)tmp.data.range(15, 8) / 255.0f - 0.5f) / 0.5f;
    tmp2 = ((float)tmp.data.range(23, 16) / 255.0f - 0.5f) / 0.5f;
    tmp3 = ((float)tmp.data.range(31, 24) / 255.0f - 0.5f) / 0.5f;

    image_blue[i]   = (image_t)(tmp0 * (float)(1 << 15) + 0.5f);
    image_blue[i+1] = (image_t)(tmp1 * (float)(1 << 15) + 0.5f);
    image_blue[i+2] = (image_t)(tmp2 * (float)(1 << 15) + 0.5f);
    image_blue[i+3] = (image_t)(tmp3 * (float)(1 << 15) + 0.5f);
  }

  loop_conv:
  for(int l = 0; l < CONV_OFM_NUMBER; l++) {
    loop_i:
    for (int i = 0; i < CONV_OUTPUT_HEIGHT; i+=2) {
      loop_j:
      for (int j = 0; j < CONV_OUTPUT_WIDTH; j+=2) {
        // Accumulators for the 4 different positions
        accum_t acc0_r = 0, acc1_r = 0, acc2_r = 0, acc3_r = 0;
        accum_t acc0_g = 0, acc1_g = 0, acc2_g = 0, acc3_g = 0;
        accum_t acc0_b = 0, acc1_b = 0, acc2_b = 0, acc3_b = 0;

        image_t image_r, image_g, image_b;
        output_t acc_sat;
        data_t tmp_out;
        weight_t weight_r, weight_g, weight_b;

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
            int kernel_1d_idx2_r = kernel_1d_idx_r & 0x1;
            int kernel_1d_idx2_g = kernel_1d_idx_g & 0x1;
            int kernel_1d_idx2_b = kernel_1d_idx_b & 0x1;
            // Image values
            // x -
            // - -
            int image_1d_idx = image_1d_idx_base + x;
            image_r = image_red[image_1d_idx];
            image_g = image_green[image_1d_idx];
            image_b = image_blue[image_1d_idx];
            weight_r  = (weights[kernel_1d_idx_r >> 1] >> (kernel_1d_idx2_r << 4)) & 0xFFFF;
            weight_g  = (weights[kernel_1d_idx_g >> 1] >> (kernel_1d_idx2_g << 4)) & 0xFFFF;
            weight_b  = (weights[kernel_1d_idx_b >> 1] >> (kernel_1d_idx2_b << 4)) & 0xFFFF;
            acc0_r += weight_r * image_r;
            acc0_g += weight_g * image_g;
            acc0_b += weight_b * image_b;
            
            // - x
            // - -
            image_1d_idx = image_1d_idx_base + x +1;
            image_r = image_red[image_1d_idx];
            image_g = image_green[image_1d_idx];
            image_b = image_blue[image_1d_idx];
            acc1_r += weight_r * image_r;
            acc1_g += weight_g * image_g;
            acc1_b += weight_b * image_b;
            
            // - -
            // x -
            image_1d_idx = image_1d_idx_base + x + IMAGE_WIDTH;
            image_r = image_red[image_1d_idx];
            image_g = image_green[image_1d_idx];
            image_b = image_blue[image_1d_idx];
            acc2_r += weight_r * image_r;
            acc2_g += weight_g * image_g;
            acc2_b += weight_b * image_b;
            
            // - -
            // - x
            image_1d_idx = image_1d_idx_base + x + IMAGE_WIDTH + 1;
            image_r = image_red[image_1d_idx];
            image_g = image_green[image_1d_idx];
            image_b = image_blue[image_1d_idx];
            acc3_r += weight_r * image_r;
            acc3_g += weight_g * image_g;
            acc3_b += weight_b * image_b;
          }
        }

        int bias_idx2 = l & 0x1;
        bias_t bia = (bias_t)((bias[l >> 1] >> (bias_idx2 << 4)) & 0xFFFF);
        accum_t acc0 = acc0_r + acc0_g + acc0_b + ((accum_t)bia << (ACCUM_BIT_WIDTH - INTEGER_BIT_WIDTH - WEIGHT_BIT_WIDTH));
        accum_t acc1 = acc1_r + acc1_g + acc1_b + ((accum_t)bia << (ACCUM_BIT_WIDTH - INTEGER_BIT_WIDTH - WEIGHT_BIT_WIDTH));
        accum_t acc2 = acc2_r + acc2_g + acc2_b + ((accum_t)bia << (ACCUM_BIT_WIDTH - INTEGER_BIT_WIDTH - WEIGHT_BIT_WIDTH));
        accum_t acc3 = acc3_r + acc3_g + acc3_b + ((accum_t)bia << (ACCUM_BIT_WIDTH - INTEGER_BIT_WIDTH - WEIGHT_BIT_WIDTH));

        accum_t acc = (acc0 > acc1) ? acc0 : acc1;
        acc = (acc > acc2) ? acc : acc2;
        acc = (acc > acc3) ? acc : acc3;

        acc = acc >> (ACCUM_BIT_WIDTH - OUTPUT_BIT_WIDTH);
        /* Relu */
        if (acc < 0)
          acc_sat = 0;
        else
          acc_sat = acc;

        strmout_t chunk_out;
        chunk_out.last = ((i == CONV_OUTPUT_HEIGHT - 2) && (j == CONV_OUTPUT_WIDTH - 2) && (l == CONV_OFM_NUMBER - 1));
        chunk_out.data = acc_sat;
        chunk_out.keep = 0xF;
        chunk_out.strb = 0xF;
        strm_out.write(chunk_out);
      }
    }
  }
}
