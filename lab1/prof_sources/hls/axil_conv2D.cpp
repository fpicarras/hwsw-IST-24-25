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

    stream_t *image_in_str = (stream_t*)image_in;
    stream_t *weights_str = (stream_t*)weights;
    stream_t *image_out_buff_str = (stream_t*)image_out_buff;

    strmio_t tmp_in;
    loop_init: for(count_t i = 0; ; i ++) {
        tmp_in = strm_in.read();
        if(i < IMAGE_HEIGHT * IMAGE_WIDTH/4) {
            image_in_str[i] = tmp_in.data;
        } else if (i < (IMAGE_HEIGHT * IMAGE_WIDTH + KERNEL_PADDED)/4) {
            weights_str[i - (IMAGE_HEIGHT * IMAGE_WIDTH/4)] = tmp_in.data;
        } else {
            bias = tmp_in.data;
        }
        if(tmp_in.last == 1) {
            break;
        }
    }

    strmio_t tmp_out;
    loop_i:
    for (count_t i = 0; i < OUTPUT_HEIGHT; i++) {
        loop_j:
        for (count_t j = 0; j < OUTPUT_WIDTH; j++) {
            accum_t acc = (accum_t) bias;
            output_image_t acc_sat;

            loop_k:
            for (count_t k = 0; k < KERNEL_SIZE; k++) {
#pragma HLS PIPELINE
            	// Indices are initialized here to highlight
            	// the incremental counting in the inner loop
            	count_t kernel_1d_idx = k * KERNEL_SIZE ; /* start of kernel row */
            	count_t image_1d_idx = (i + k) * IMAGE_WIDTH + j ; /* start of input row */
                loop_x:
                for (count_t x = 0; x < KERNEL_SIZE; x++, kernel_1d_idx++, image_1d_idx++) {
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

            count_t image_out_idx = (i * OUTPUT_WIDTH + j) & 0x3;
            image_out_buff[image_out_idx] = acc_sat;

            if(image_out_idx == 0x3) {
                tmp_out.last = ((i == OUTPUT_HEIGHT - 1) && (j == OUTPUT_WIDTH - 1));
                tmp_out.data = image_out_buff_str[0];
                tmp_out.keep = 0xF;
                tmp_out.strb = 0xF;
                strm_out.write(tmp_out);
            }
        }
    }
}
