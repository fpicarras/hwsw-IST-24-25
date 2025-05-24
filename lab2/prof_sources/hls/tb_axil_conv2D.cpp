
#include "axil_conv2D.h"
#include "tb_axil_conv2D_int.h"
#include <stdint.h>

static int8_t image_in[IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH];
static int16_t kernel[CONV_OFM_NUMBER*CONV_KERNEL_SIZE * CONV_KERNEL_SIZE];
static int16_t bias[CONV_OFM_NUMBER];

static int8_t hw_image_out[CONV_OFM_NUMBER * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH];
static int8_t sw_image_out[CONV_OFM_NUMBER * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH];

/**
 * Performs software-only matrix convolution.
 * @param matrix_in Input matrix
 * @param matrix_out Output matrix after convolution
 */

int main() {
    init_inputs(image_in, kernel, bias);

    hls::stream<strmio_t> str_in;
    hls::stream<strmio_t> str_out;
    strmio_t tmp_in, tmp_out;
    for (int i = 0; i < IMAGE_CHANNELS*IMAGE_HEIGHT*IMAGE_WIDTH; i+=4) {
        tmp_in.data(7, 0) = image_in[i];
        tmp_in.data(15, 8) = image_in[i + 1];
        tmp_in.data(23, 16) = image_in[i + 2];
        tmp_in.data(31, 24) = image_in[i + 3];
        tmp_in.last = (ap_int<1>)0;
        str_in.write(tmp_in);
    }
    for (int i = 0; i < CONV_OFM_NUMBER*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE; i+=2) {
        tmp_in.data(15, 0) = kernel[i];
        tmp_in.data(31, 16) = kernel[i + 1];
        tmp_in.last = (ap_int<1>)0;
        str_in.write(tmp_in);
    }
    for (int i = 0; i < CONV_OFM_NUMBER; i+=2) {
        tmp_in.data(15, 0) = bias[i];
        tmp_in.data(31, 16) = bias[i + 1];
        tmp_in.last = (ap_int<1>)(i == (CONV_OFM_NUMBER - 2));
        str_in.write(tmp_in);
    }
    axil_conv2D(str_in, str_out);
    for (int i = 0; i < CONV_OFM_NUMBER*CONV_OUTPUT_HEIGHT*CONV_OUTPUT_WIDTH; i+=4) {
        tmp_out = str_out.read();
        hw_image_out[i] = tmp_out.data(7, 0);
        hw_image_out[i + 1] = tmp_out.data(15, 8);
        hw_image_out[i + 2] = tmp_out.data(23, 16);
        hw_image_out[i + 3] = tmp_out.data(31, 24);
    }

    sw_convolution_2D(image_in, kernel, bias, sw_image_out);

    return check_output(sw_image_out, hw_image_out);
}
