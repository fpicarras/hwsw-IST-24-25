/**
 * @file axil_conv2D.cpp
 * @date May, 2023
 *
 * Support file for lab1 of Hardware-Software Co-Design 2023
 *
 * Contains the HLS implementation of a simple matrix convolution IP.
 */

#include "axil_conv2D.h"

void axil_conv2D(hls::stream<strmio_t> &strm_in,
                 hls::stream<strmio_t> &strm_out) {

#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS interface axis port=strm_in
#pragma HLS INTERFACE axis port=strm_out

    static input_image_t image_in[IMAGE_HEIGHT * IMAGE_WIDTH];
    static weight_t weights[KERNEL_PADDED];
    static output_image_t image_out_buff[STREAM_BIT_WIDTH/IMAGE_BIT_WIDTH];
    static bias_t bias;

    strmio_t tmp_in;
    loop_init: for(int i = 0; ; i += 4) {
        tmp_in = strm_in.read();
        if(i < IMAGE_HEIGHT * IMAGE_WIDTH) {
            image_in[i] = tmp_in.data.range(7, 0);
            image_in[i + 1] = tmp_in.data.range(15, 8);
            image_in[i + 2] = tmp_in.data.range(23, 16);
            image_in[i + 3] = tmp_in.data.range(31, 24);
        } else if (i < IMAGE_HEIGHT * IMAGE_WIDTH + KERNEL_PADDED) {
            int i2 = i - IMAGE_HEIGHT * IMAGE_WIDTH;
            weights[i2] = tmp_in.data.range(7, 0);
            weights[i2 + 1] = tmp_in.data.range(15, 8);
            weights[i2 + 2] = tmp_in.data.range(23, 16);
            weights[i2 + 3] = tmp_in.data.range(31, 24);
        } else {
            bias = tmp_in.data;
        }
        if(tmp_in.last == 1) {
            break;
        }
    }

    strmio_t tmp_out;
    loop_i:
    for (int i = 0; i < OUTPUT_HEIGHT; i++) {
        loop_j:
        for (int j = 0; j < OUTPUT_WIDTH; j++) {
            accum_t acc = (accum_t) bias;
            output_image_t acc_sat;

            loop_k:
            for (int k = 0; k < KERNEL_SIZE; k++) {
#pragma HLS PIPELINE
            	// Indices are initialized here to highlight
            	// the incremental counting in the inner loop
            	int kernel_1d_idx = k * KERNEL_SIZE ; /* start of kernel row */
            	int image_1d_idx = (i + k) * IMAGE_WIDTH + j ; /* start of input row */
                loop_x:
                for (int x = 0; x < KERNEL_SIZE; x++, kernel_1d_idx++, image_1d_idx++) {
                    acc += weights[kernel_1d_idx] * image_in[image_1d_idx];
                }
            }

            /* Normalize */
            if (acc > 255)
                acc_sat = 255;
            else if (acc < 0)
                acc_sat = 0;
            else
                acc_sat = acc;

            tmp_out.data.range(7, 0) = tmp_out.data.range(15, 8);
            tmp_out.data.range(15, 8) = tmp_out.data.range(23, 16);
            tmp_out.data.range(23, 16) = tmp_out.data.range(31, 24);
            tmp_out.data.range(31, 24) = acc_sat;

            if(((i * OUTPUT_WIDTH + j) & 0x3) == 0x3) {
                tmp_out.last = ((i == OUTPUT_HEIGHT - 1) && (j == OUTPUT_WIDTH - 1));
                tmp_out.keep = 0xF;
                tmp_out.strb = 0xF;
                strm_out.write(tmp_out);
            }
        }
    }
}
