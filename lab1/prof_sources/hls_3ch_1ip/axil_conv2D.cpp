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
                 hls::stream<strmio_t> &strm_out,
                 weight_t weights[KERNEL_SIZE * KERNEL_SIZE],
                 bias_t bias) {

#pragma HLS INTERFACE s_axilite port=return bundle=BUS1
#pragma HLS INTERFACE axis port=strm_in
#pragma HLS INTERFACE axis port=strm_out
#pragma HLS INTERFACE s_axilite port=weights bundle=BUS1
#pragma HLS INTERFACE s_axilite port=bias bundle=BUS1

    static input_image_t image_in[IMAGE_HEIGHT * IMAGE_WIDTH];

    strmio_t chunk_in;
    loop_init: for(int i = 0; ; i ++) {
        chunk_in = strm_in.read();
        image_in[i] = chunk_in.data;
        if(chunk_in.last == 1) {
            break;
        }
    }

    loop_i:
    for (count_t i = 0; i < OUTPUT_HEIGHT; i++) {
        loop_j:
        for (count_t j = 0; j < OUTPUT_WIDTH; j++) {
            accum_t acc_r = (accum_t) bias;
            accum_t acc_g = (accum_t) bias;
            accum_t acc_b = (accum_t) bias;
            output_image_t acc_sat_r;
            output_image_t acc_sat_g;
            output_image_t acc_sat_b;

            input_image_t tmp_in;
            output_image_t tmp_out;
            weight_t tmp_w;

            loop_k:
            for (count_t k = 0; k < KERNEL_SIZE; k++) {
#pragma HLS PIPELINE
            	// Indices are initialized here to highlight
            	// the incremental counting in the inner loop
            	count_t kernel_1d_idx = k * KERNEL_SIZE ; /* start of kernel row */
            	count_t image_1d_idx = (i + k) * IMAGE_WIDTH + j ; /* start of input row */
                loop_x:
                for (count_t x = 0; x < KERNEL_SIZE; x++, kernel_1d_idx++, image_1d_idx++) {
                    tmp_in = image_in[image_1d_idx];
                    tmp_w  = weights[kernel_1d_idx];
                    acc_r += tmp_w * tmp_in(7,0);
                    acc_g += tmp_w * tmp_in(15,8);
                    acc_b += tmp_w * tmp_in(23,16);
                }
            }

            /* Normalize */
            // Red channel
            if (acc_r > 255)
                acc_sat_r = 255;
            else if (acc_r < 0)
                acc_sat_r = 0;
            else
                acc_sat_r = acc_r;
            
            // Green channel
            if (acc_g > 255)
                acc_sat_g = 255;
            else if (acc_g < 0)
                acc_sat_g = 0;
            else
                acc_sat_g = acc_g;

            // Blue channel
            if (acc_b > 255)
                acc_sat_b = 255;
            else if (acc_b < 0)
                acc_sat_b = 0;
            else
                acc_sat_b = acc_b;

            tmp_out(7,0)   = acc_sat_r(7,0);
            tmp_out(15,8)  = acc_sat_g(7,0);
            tmp_out(23,16) = acc_sat_b(7,0);
            tmp_out(31,24) = 0;

            strmio_t chunk_out;
            chunk_out.last = ((i == OUTPUT_HEIGHT - 1) && (j == OUTPUT_WIDTH - 1));
            chunk_out.data = tmp_out;
            chunk_out.keep = 0xF;
            chunk_out.strb = 0xF;
            strm_out.write(chunk_out);
        }
    }
}
