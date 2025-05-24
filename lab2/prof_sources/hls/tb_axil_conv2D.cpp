/**
 * @file tb_axil_conv2D.cpp
 * @date May, 2022
 *
 * Support file for lab1 of Hardware-Software Co-Design 2022
 *
 * Contains the HLS test bench for a simple matrix convolution IP.
 */

#include "axil_conv2D.h"

#define HW_IP

static unsigned char image_in[IMAGE_HEIGHT * IMAGE_WIDTH];
static signed char kernel[KERNEL_SIZE * KERNEL_SIZE] = {-2, -1, 0,
                                                        -1, 1, 1,
                                                        0, 1, 2};
static int bias = 5;

static unsigned char hw_image_out[OUTPUT_HEIGHT * OUTPUT_WIDTH];
static unsigned char sw_image_out[OUTPUT_HEIGHT * OUTPUT_WIDTH];

/**
 * Performs software-only matrix convolution.
 * @param matrix_in Input matrix
 * @param matrix_out Output matrix after convolution
 */
void sw_convolution_2D(const unsigned char *matrix_in, unsigned char *matrix_out) {
    for (int i = 0; i < OUTPUT_HEIGHT; i++)
        for (int j = 0; j < OUTPUT_WIDTH; j++) {
            int accum = bias;                       /* initialize result with bias */

            for (int k = 0; k < KERNEL_SIZE; k++)
                for (int x = 0; x < KERNEL_SIZE; x++) {
                    /* Kernel index */
                    int kernel_1d_idx =
                            k * KERNEL_SIZE +       /* kernel row */
                            x;                      /* kernel column */

                    /* Input matrix index */
                    int input_1d_idx =
                            (i + k) * IMAGE_WIDTH + /* input row */
                            j + x;                  /* input column */

                    accum += kernel[kernel_1d_idx] * matrix_in[input_1d_idx];
                }

            /* Normalize result */
            if (accum > 127)
                accum = 127;
            else if (accum < 0)
                accum = 0;

            matrix_out[i * OUTPUT_WIDTH + j] = accum;
        }
}

int main() {
    printf("Input Image\n\r");
    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        for (int j = 0; j < IMAGE_WIDTH; j++) {
            image_in[i * IMAGE_WIDTH + j] = (i + 1) * 10 + (j + 1);
            printf("%d ", image_in[i * IMAGE_WIDTH + j]);
        }
        printf("\n\r");
    }

#ifdef HW_IP
    hls::stream<strmio_t> str_in;
    hls::stream<strmio_t> str_out;
    strmio_t tmp_in, tmp_out;
    weight_t weights [KERNEL_PADDED];
    stream_t *image_in_str = (stream_t*)image_in;
    stream_t *weights_str = (stream_t*)weights;
    for(int i = 0; i < KERNEL_PADDED; i++) {
        if(i < KERNEL_SIZE * KERNEL_SIZE) {
            weights[i] = kernel[i];
        } else {
            weights[i] = 0;
        }
    }
    for (int i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH/4; i++) {
        tmp_in.data = image_in_str[i];
        tmp_in.last = (ap_int<1>)0;
        str_in.write(tmp_in);
    }
    for (int i = 0; i < KERNEL_PADDED/4; i++) {
        tmp_in.data = weights_str[i];
        tmp_in.last = (ap_int<1>)0;
        str_in.write(tmp_in);
    }
    tmp_in.data = bias;
    tmp_in.last = 1;
    str_in.write(tmp_in);
    axil_conv2D(str_in, str_out);
    for (int i = 0; i < OUTPUT_HEIGHT*OUTPUT_WIDTH; i+=4) {
        tmp_out = str_out.read();
        hw_image_out[i] = tmp_out.data.range(7, 0);
        hw_image_out[i + 1] = tmp_out.data.range(15, 8);
        hw_image_out[i + 2] = tmp_out.data.range(23, 16);
        hw_image_out[i + 3] = tmp_out.data.range(31, 24);
    }
#endif

    sw_convolution_2D(image_in, sw_image_out);

    printf("Output Image SW\n\r");
    for (int i = 0; i < OUTPUT_HEIGHT; i++) {
        for (int j = 0; j < OUTPUT_WIDTH; j++) {
            printf("%4d ", sw_image_out[i * OUTPUT_WIDTH + j]);
        }
        printf("\n\r");
    }

    printf("Output Image HW\n\r");
    for (int i = 0; i < OUTPUT_HEIGHT; i++) {
        for (int j = 0; j < OUTPUT_WIDTH; j++) {
            printf("%4d ", hw_image_out[i * OUTPUT_WIDTH + j]);
        }
        printf("\n\r");
    }

#ifdef HW_IP
    int err_cnt = 0;
    for (int i = 0; i < OUTPUT_HEIGHT; i++)
        for (int j = 0; j < OUTPUT_WIDTH; j++)
            if (hw_image_out[i * OUTPUT_WIDTH + j] != sw_image_out[i * OUTPUT_WIDTH + j]) {
                err_cnt++;
                // printf("%d,%d: %d != %d\n\r",
                //        i, j, hw_image_out[i * OUTPUT_WIDTH + j], sw_image_out[i * OUTPUT_WIDTH + j]);
            }

    return err_cnt;
#else
    return 0;
#endif
}
