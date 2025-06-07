
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

  if(!weights_ready) {
    /* Bias Stream */
    loop_bias:
    for(int i = 0; i < CONV_OFM_NUMBER; i += BIAS_PER_DATA) {
      tmp = strm_in.read();
      bias[i]   = (bias_t)tmp.data.range(15, 0);
      bias[i+1] = (bias_t)tmp.data.range(31, 16);
      bias[i+2] = (bias_t)tmp.data.range(47, 32);
      bias[i+3] = (bias_t)tmp.data.range(63, 48);
    }

    /* Weights Stream */
    loop_weights:
    for(int i = 0; i < IMAGE_CHANNELS*CONV_OFM_NUMBER*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE; i += WEIGHTS_PER_DATA) {
      tmp = strm_in.read();
      weights[i]   = (weight_t)tmp.data.range(15, 0);
      weights[i+1] = (weight_t)tmp.data.range(31, 16);
      weights[i+2] = (weight_t)tmp.data.range(47,32);
      weights[i+3] = (weight_t)tmp.data.range(63, 48);
    }
  }

  weights_ready = true;
  /* Input Image Stream */
  loop_red: 
  for(int i = 0; i < IMAGE_HEIGHT*IMAGE_WIDTH; i += PIXEL_PER_DATA) {
    tmp = strm_in.read();
    image_red[i] = tmp.data.range(15, 0);
    image_red[i+1] = tmp.data.range(31, 16);
    image_red[i+2] = tmp.data.range(47, 32);
    image_red[i+3] = tmp.data.range(63, 48);
  }
  loop_green: 
  for(int i = 0; i < IMAGE_HEIGHT*IMAGE_WIDTH; i += PIXEL_PER_DATA) {
    tmp = strm_in.read();
    image_green[i] = tmp.data.range(15, 0);
    image_green[i+1] = tmp.data.range(31, 16);
    image_green[i+2] = tmp.data.range(47, 32);
    image_green[i+3] = tmp.data.range(63, 48);
  }
  loop_blue: 
  for(int i = 0; i < IMAGE_HEIGHT*IMAGE_WIDTH; i += PIXEL_PER_DATA) {
    tmp = strm_in.read();
    image_blue[i] = tmp.data.range(15, 0);
    image_blue[i+1] = tmp.data.range(31, 16);
    image_blue[i+2] = tmp.data.range(47, 32);
    image_blue[i+3] = tmp.data.range(63, 48);
  }

  loop_conv:
  for(int l = 0; l < CONV_OFM_NUMBER; l++) {
    // Loading Kernels
    weight_t kernel_r[CONV_KERNEL_SIZE*CONV_KERNEL_SIZE], kernel_g[CONV_KERNEL_SIZE*CONV_KERNEL_SIZE], kernel_b[CONV_KERNEL_SIZE*CONV_KERNEL_SIZE];
    int kernel_base_idx;
    for(int k = 0; k < CONV_KERNEL_SIZE; k++){

      kernel_base_idx = k * CONV_KERNEL_SIZE + l*(CONV_KERNEL_SIZE*CONV_KERNEL_SIZE*IMAGE_CHANNELS);
      for(int x = 0; x < CONV_KERNEL_SIZE; x++){
        int kernel_idx_r = (kernel_base_idx + x);
        int kernel_idx_g = (kernel_base_idx + x) + CONV_KERNEL_SIZE*CONV_KERNEL_SIZE;
        int kernel_idx_b = (kernel_base_idx + x) + 2*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE;
        kernel_r[k*CONV_KERNEL_SIZE+x] = weights[kernel_idx_r];
        kernel_g[k*CONV_KERNEL_SIZE+x] = weights[kernel_idx_g];
        kernel_b[k*CONV_KERNEL_SIZE+x] = weights[kernel_idx_b];
      }
    }

    accum_t bia = ((accum_t)bias[l] << (ACCUM_BIT_WIDTH - INTEGER_BIT_WIDTH - WEIGHT_BIT_WIDTH));
    
    loop_i:
    for (int i = 0; i < CONV_OUTPUT_HEIGHT; i+=2) {
      loop_j:
      for (int j = 0; j < CONV_OUTPUT_WIDTH; j+=2) {
        maxpool_t maxpool_result = 0;
        accum_t best_max = 0;

        for(int ml = 0; ml < POOL_KERNEL_SIZE; ml++)              // BUG OUT OF BOUNDRIES
          for(int mc = 0; mc < POOL_KERNEL_SIZE; mc++) {
            accum_t acc_r, acc_g, acc_b;

            #pragma HLS PIPELINE
            // Base image idx
            int image_idx = i*IMAGE_WIDTH + j + CONV_KERNEL_SIZE*(mc + IMAGE_WIDTH*ml);
            // Aux idx
            int curr_image_idx;
            
            // [0][0]
            curr_image_idx = image_idx;
            acc_r = kernel_r[0]*image_red[curr_image_idx];
            acc_g = kernel_r[0]*image_green[curr_image_idx];
            acc_b = kernel_r[0]*image_blue[curr_image_idx];

            // [0][1]
            curr_image_idx = image_idx + 1;
            acc_r += kernel_r[1]*image_red[curr_image_idx];
            acc_g += kernel_r[1]*image_green[curr_image_idx];
            acc_b += kernel_r[1]*image_blue[curr_image_idx];

            // [0][2]
            curr_image_idx = image_idx + 2;
            acc_r += kernel_r[2]*image_red[curr_image_idx];
            acc_g += kernel_r[2]*image_green[curr_image_idx];
            acc_b += kernel_r[2]*image_blue[curr_image_idx];

            // [1][0]
            curr_image_idx = image_idx + IMAGE_WIDTH;
            acc_r += kernel_r[3]*image_red[curr_image_idx];
            acc_g += kernel_r[3]*image_green[curr_image_idx];
            acc_b += kernel_r[3]*image_blue[curr_image_idx];

            // [1][1]
            curr_image_idx = image_idx + IMAGE_WIDTH + 1;
            acc_r += kernel_r[4]*image_red[curr_image_idx];
            acc_g += kernel_r[4]*image_green[curr_image_idx];
            acc_b += kernel_r[4]*image_blue[curr_image_idx];

            // [1][2]
            curr_image_idx = image_idx + IMAGE_WIDTH + 2;
            acc_r += kernel_r[5]*image_red[curr_image_idx];
            acc_g += kernel_r[5]*image_green[curr_image_idx];
            acc_b += kernel_r[5]*image_blue[curr_image_idx];

            // [2][0]
            curr_image_idx = image_idx + 2*IMAGE_WIDTH;
            acc_r += kernel_r[6]*image_red[curr_image_idx];
            acc_g += kernel_r[6]*image_green[curr_image_idx];
            acc_b += kernel_r[6]*image_blue[curr_image_idx];

            // [2][1]
            curr_image_idx = image_idx + 2*IMAGE_WIDTH + 1;
            acc_r += kernel_r[7]*image_red[curr_image_idx];
            acc_g += kernel_r[7]*image_green[curr_image_idx];
            acc_b += kernel_r[7]*image_blue[curr_image_idx];

            // [2][2]
            curr_image_idx = image_idx + 2*IMAGE_WIDTH + 2;
            acc_r += kernel_r[8]*image_red[curr_image_idx];
            acc_g += kernel_r[8]*image_green[curr_image_idx];
            acc_b += kernel_r[8]*image_blue[curr_image_idx];

            accum_t acc = acc_r + acc_g + acc_b + bia;
            best_max = (best_max < acc) ? acc : best_max;
          }       
        
        // End of a maxpool cycle
        best_max = best_max >> (ACCUM_BIT_WIDTH - MAXPOOL_BIT_WIDTH); 
        // ReLU
        if(best_max > 0)
          maxpool_result = best_max;
        else
          maxpool_result = 0;

        strmout_t chunk_out;
        chunk_out.last = ((i == CONV_OUTPUT_HEIGHT - 2) && (j == CONV_OUTPUT_WIDTH - 2) && (l == CONV_OFM_NUMBER - 1));
        chunk_out.data = maxpool_result;
        chunk_out.keep = 0xF; // Fix for 32bit
        chunk_out.strb = 0xF;
        strm_out.write(chunk_out);
      }
    }
  }
}
