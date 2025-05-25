
#include "axil_conv3D.h"

void axil_conv3D(hls::stream<strmio_t> &strm_in,
                 hls::stream<strmio_t> &strm_out) {

#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS interface axis port=strm_in
#pragma HLS INTERFACE axis port=strm_out

static data_t image_red[IMAGE_HEIGHT*IMAGE_WIDTH/IMAGES_PER_DATA];
static data_t image_green[IMAGE_HEIGHT*IMAGE_WIDTH/IMAGES_PER_DATA];
static data_t image_blue[IMAGE_HEIGHT*IMAGE_WIDTH/IMAGES_PER_DATA];
static data_t weights[IMAGE_CHANNELS*CONV_OFM_NUMBER*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE/WEIGHTS_PER_DATA];
static data_t bias[CONV_OFM_NUMBER/BIAS_PER_DATA];

  strmio_t tmp;
  /* Input Image Stream */
  loop_red: 
  for(int i = 0; i < IMAGE_HEIGHT*IMAGE_WIDTH/IMAGES_PER_DATA; i ++) {
    tmp = strm_in.read();
    image_red[i] = tmp.data;
  }
  loop_green: 
  for(int i = 0; i < IMAGE_HEIGHT*IMAGE_WIDTH/IMAGES_PER_DATA; i ++) {
    tmp = strm_in.read();
    image_green[i] = tmp.data;
  }
  loop_blue: 
  for(int i = 0; i < IMAGE_HEIGHT*IMAGE_WIDTH/IMAGES_PER_DATA; i ++) {
    tmp = strm_in.read();
    image_blue[i] = tmp.data;
  }

  /* Weights Stream */
  loop_weights:
  for(int i = 0; i < IMAGE_CHANNELS*CONV_OFM_NUMBER*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE/WEIGHTS_PER_DATA; i ++) {
    tmp = strm_in.read();
    weights[i] = tmp.data;
  }

  loop_bias:
  for(int i = 0; i < CONV_OFM_NUMBER/BIAS_PER_DATA; i ++) {
    tmp = strm_in.read();
    bias[i] = tmp.data;
  }

  output_t conv [OUTPUTS_PER_DATA];
  loop_conv:
  for(int l = 0; l < CONV_OFM_NUMBER; l++) {
    loop_i:
    for (int i = 0; i < CONV_OUTPUT_HEIGHT; i++) {
      loop_j:
      for (int j = 0; j < CONV_OUTPUT_WIDTH; j++) {
        accum_t acc_r = 0;
        accum_t acc_g = 0;
        accum_t acc_b = 0;

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
            int image_1d_idx = image_1d_idx_base + x;
            int image_1d_idx2 = image_1d_idx & 0x3;
            int kernel_1d_idx_r = (kernel_1d_idx + x);
            int kernel_1d_idx_g = (kernel_1d_idx + x) + CONV_KERNEL_SIZE*CONV_KERNEL_SIZE;
            int kernel_1d_idx_b = (kernel_1d_idx + x) + 2*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE;
            int kernel_1d_idx2_r = kernel_1d_idx_r & 0x1;
            int kernel_1d_idx2_g = kernel_1d_idx_g & 0x1;
            int kernel_1d_idx2_b = kernel_1d_idx_b & 0x1;
            image_r = (image_red[image_1d_idx >> 2] >> (image_1d_idx2 << 3)) & 0xFF;
            image_g = (image_green[image_1d_idx >> 2] >> (image_1d_idx2 << 3)) & 0xFF;
            image_b = (image_blue[image_1d_idx >> 2] >> (image_1d_idx2 << 3)) & 0xFF;
            weight_r  = (weights[kernel_1d_idx_r >> 1] >> (kernel_1d_idx2_r << 4)) & 0xFFFF;
            weight_g  = (weights[kernel_1d_idx_g >> 1] >> (kernel_1d_idx2_g << 4)) & 0xFFFF;
            weight_b  = (weights[kernel_1d_idx_b >> 1] >> (kernel_1d_idx2_b << 4)) & 0xFFFF;
            acc_r += weight_r * image_r;
            acc_g += weight_g * image_g;
            acc_b += weight_b * image_b;
          }
        }

        int bias_idx2 = l & 0x1;
        bias_t bia = (bias_t)((bias[l >> 1] >> (bias_idx2 << 4)) & 0xFFFF);
        accum_t acc = acc_r + acc_g + acc_b + ((accum_t)bia << (ACCUM_BIT_WIDTH - INTEGER_BIT_WIDTH - WEIGHT_BIT_WIDTH));

        acc = acc >> (ACCUM_BIT_WIDTH - OUTPUT_BIT_WIDTH);
        /* Normalize */
        if (acc < 0)
          acc_sat = 0;
        else
          acc_sat = acc;

        int ind_out = (i * CONV_OUTPUT_WIDTH + j) & 0x1;
        conv[ind_out] = acc_sat;

        if(ind_out == 0x1) {
          strmio_t chunk_out;
          chunk_out.last = ((i == CONV_OUTPUT_HEIGHT - 1) && (j == CONV_OUTPUT_WIDTH - 1) && (l == CONV_OFM_NUMBER - 1));
          chunk_out.data(15,0) = conv[0];
          chunk_out.data(31,16) = conv[1];
          chunk_out.keep = 0xF;
          chunk_out.strb = 0xF;
          strm_out.write(chunk_out);
        }
      }
    }
  }
}
