
#include "axil_conv3D.h"

void axil_conv3D(hls::stream<strmio_t> &strm_in,
                 hls::stream<strmio_t> &strm_out) {

#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS interface axis port=strm_in
#pragma HLS INTERFACE axis port=strm_out

static data_t image_red[IMAGE_HEIGHT*IMAGE_WIDTH/IMAGES_PER_DATA];
static data_t image_green[IMAGE_HEIGHT*IMAGE_WIDTH/IMAGES_PER_DATA];
static data_t image_blue[IMAGE_HEIGHT*IMAGE_WIDTH/IMAGES_PER_DATA];
static data_t weights_red[CONV_OFM_NUMBER*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE/WEIGHTS_PER_DATA];
static data_t weights_green[CONV_OFM_NUMBER*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE/WEIGHTS_PER_DATA];
static data_t weights_blue[CONV_OFM_NUMBER*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE/WEIGHTS_PER_DATA];
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
  loop_weights_red:
  for(int i = 0; i < CONV_OFM_NUMBER*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE/WEIGHTS_PER_DATA; i ++) {
    tmp = strm_in.read();
    weights_red[i] = tmp.data;
  }
  loop_weights_green:
  for(int i = 0; i < CONV_OFM_NUMBER*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE/WEIGHTS_PER_DATA; i ++) {
    tmp = strm_in.read();
    weights_green[i] = tmp.data;
  }
  loop_weights_blue:
  for(int i = 0; i < CONV_OFM_NUMBER*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE/WEIGHTS_PER_DATA; i ++) {
    tmp = strm_in.read();
    weights_blue[i] = tmp.data;
  }

  loop_bias:
  for(int i = 0; i < CONV_OFM_NUMBER/BIAS_PER_DATA; i ++) {
    tmp = strm_in.read();
    bias[i] = tmp.data;
  }

  loop_conv:
  for(int l = 0; l < CONV_OFM_NUMBER; l++) {
    loop_i:
    for (int i = 0; i < CONV_OUTPUT_HEIGHT; i++) {
      loop_j:
      for (int j = 0; j < CONV_OUTPUT_WIDTH; j++) {
        accum_t acc_r = 0;
        accum_t acc_g = 0;
        accum_t acc_b = 0;

        image_t image_r, image_g, image_b, acc_sat;
        data_t tmp_out;
        weight_t weight_r, weight_g, weight_b;

        loop_k:
        for (int k = 0; k < CONV_KERNEL_SIZE; k++) {
  #pragma HLS PIPELINE
          // Indices are initialized here to highlight
          // the incremental counting in the inner loop
          int kernel_1d_idx = k * CONV_KERNEL_SIZE + l*(CONV_KERNEL_SIZE*CONV_KERNEL_SIZE); /* start of kernel row */
          int image_1d_idx = (i + k) * IMAGE_WIDTH + j; /* start of input row */
          int image_1d_idx2, kernel_1d_idx2;
          loop_x:
          for (int x = 0; x < CONV_KERNEL_SIZE; x++, kernel_1d_idx++, image_1d_idx++) {
            image_1d_idx2 = image_1d_idx & 0x3;
            kernel_1d_idx2 = kernel_1d_idx & 0x1;
            image_r = (image_t)((image_red[image_1d_idx >> 2] >> (image_1d_idx2 << 3)) & 0xFF);
            image_g = (image_t)((image_green[image_1d_idx >> 2] >> (image_1d_idx2 << 3)) & 0xFF);
            image_b = (image_t)((image_blue[image_1d_idx >> 2] >> (image_1d_idx2 << 3)) & 0xFF);
            weight_r  = (weight_t)((weights_red[kernel_1d_idx >> 1] >> (kernel_1d_idx2 << 4)) & 0xFFFF);
            weight_g  = (weight_t)((weights_green[kernel_1d_idx >> 1] >> (kernel_1d_idx2 << 4)) & 0xFFFF);
            weight_b  = (weight_t)((weights_blue[kernel_1d_idx >> 1] >> (kernel_1d_idx2 << 4)) & 0xFFFF);
            acc_r += weight_r * image_r;
            acc_g += weight_g * image_g;
            acc_b += weight_b * image_b;
          }
        }

        int bias_idx2 = l & 0x1;
        bias_t bia = (bias_t)((bias[l >> 1] >> (bias_idx2 << 4)) & 0xFFFF);
        accum_t acc = acc_r + acc_g + acc_b + bia;

        acc = acc >> WEIGHT_BIT_WIDTH;
        /* Normalize */
        if (acc > 127)
          acc_sat = 127;
        else if (acc < 0)
          acc_sat = 0;
        else
          acc_sat = acc;

        tmp_out = tmp_out >> 8;
        tmp_out(31, 24) = acc_sat;

        if(((i * CONV_OUTPUT_WIDTH + j) & 0x3) == 0x3) {
          strmio_t chunk_out;
          chunk_out.last = ((i == CONV_OUTPUT_HEIGHT - 1) && (j == CONV_OUTPUT_WIDTH - 1) && (l == CONV_OFM_NUMBER - 1));
          chunk_out.data = tmp_out;
          chunk_out.keep = 0xF;
          chunk_out.strb = 0xF;
          strm_out.write(chunk_out);
        }
      }
    }
  }
}
