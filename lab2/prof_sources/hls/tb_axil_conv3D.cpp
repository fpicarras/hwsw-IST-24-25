
#include "axil_conv3D.h"
#include <stdint.h>

static int8_t image_in[IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH];
static int16_t kernel[IMAGE_CHANNELS*CONV_OFM_NUMBER*CONV_KERNEL_SIZE * CONV_KERNEL_SIZE];
static int16_t bias[CONV_OFM_NUMBER];

static int8_t hw_image_out[CONV_OFM_NUMBER * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH];
static int8_t sw_image_out[CONV_OFM_NUMBER * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH];

void init_inputs(int8_t *image_in, int16_t * kernel, int16_t * bias) {
  printf("Input Image\n\r");
  for(int k = 0; k < IMAGE_CHANNELS; k++) {
      for (int i = 0; i < IMAGE_HEIGHT; i++) {
          for (int j = 0; j < IMAGE_WIDTH; j++) {
              image_in[k*IMAGE_HEIGHT*IMAGE_WIDTH + i * IMAGE_WIDTH + j] = (i + 1) * 30 + (j + 1)*3 + k;
              printf("%4d ", image_in[k*IMAGE_HEIGHT*IMAGE_WIDTH + i * IMAGE_WIDTH + j]);
          }
          printf("\n");
      }
      printf("\n");
  }
  printf("Kernel\n\r");
  for (int i = 0; i < CONV_OFM_NUMBER; i++) {
    for(int l = 0; l < IMAGE_CHANNELS; l++) {
        for (int j = 0; j < CONV_KERNEL_SIZE * CONV_KERNEL_SIZE; j++) {
            kernel[(l + i*IMAGE_CHANNELS) * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE + j] = (1 << (14 + i)) + (1 << 13)*(j + l);
            printf("%6d ", kernel[(l + i*IMAGE_CHANNELS) * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE + j]);
        }
        printf("\n");
    }
    printf("\n");
  }
  printf("Bias\n\r");
  for (int i = 0; i < CONV_OFM_NUMBER; i++) {
      bias[i] = (1 << 14);
      printf("%6d ", bias[i]);
  }
  printf("\n");
}

void sw_convolution_3D(const int8_t *matrix_in, const int16_t * kernel, const int16_t * bias, int8_t *matrix_out) {
    for(int l = 0; l < CONV_OFM_NUMBER; l ++)
        for (int i = 0; i < CONV_OUTPUT_HEIGHT; i++)
            for (int j = 0; j < CONV_OUTPUT_WIDTH; j++) {
                int accum = bias[l];                       /* initialize result with bias */

                for (int k = 0; k < CONV_KERNEL_SIZE; k++)
                    for (int x = 0; x < CONV_KERNEL_SIZE; x++) {
                        /* Kernel index */
                        int kernel_1d_idx = l*(CONV_KERNEL_SIZE*CONV_KERNEL_SIZE*IMAGE_CHANNELS) + // kernel number
                                k * CONV_KERNEL_SIZE +       /* kernel row */
                                x;                      /* kernel column */

                        /* Input matrix index */
                        int input_1d_idx =
                                (i + k) * IMAGE_WIDTH + /* input row */
                                j + x;                  /* input column */

                        accum += kernel[kernel_1d_idx] * matrix_in[input_1d_idx];
                        accum += kernel[kernel_1d_idx + CONV_KERNEL_SIZE*CONV_KERNEL_SIZE] * matrix_in[input_1d_idx + IMAGE_HEIGHT * IMAGE_WIDTH];
                        accum += kernel[kernel_1d_idx + 2*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE] * matrix_in[input_1d_idx + 2 * IMAGE_HEIGHT * IMAGE_WIDTH];
                    }

                /* Normalize result */
                accum >>= 16;
                if (accum > 127)
                    accum = 127;
                else if (accum < 0)
                    accum = 0;

                matrix_out[l*CONV_OUTPUT_HEIGHT*CONV_OUTPUT_WIDTH + i * CONV_OUTPUT_WIDTH + j] = accum;
            }
}

int check_output(const int8_t *sw_matrix_out, int8_t *hw_matrix_out) {
    printf("Output Image SW\n\r");
    for(int k = 0; k < CONV_OFM_NUMBER; k ++)
        for (int i = 0; i < CONV_OUTPUT_HEIGHT; i++) {
            for (int j = 0; j < CONV_OUTPUT_WIDTH; j++) {
                printf("%4d ", sw_matrix_out[k * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH + i * CONV_OUTPUT_WIDTH + j]);
            }
            printf("\n\r");
        }

    printf("Output Image HW\n\r");
    for(int k = 0; k < CONV_OFM_NUMBER; k ++)
        for (int i = 0; i < CONV_OUTPUT_HEIGHT; i++) {
            for (int j = 0; j < CONV_OUTPUT_WIDTH; j++) {
                printf("%4d ", hw_matrix_out[k * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH + i * CONV_OUTPUT_WIDTH + j]);
            }
            printf("\n\r");
        }

    int err_cnt = 0;
    for(int k = 0; k < CONV_OFM_NUMBER; k ++)
        for (int i = 0; i < CONV_OUTPUT_HEIGHT; i++)
            for (int j = 0; j < CONV_OUTPUT_WIDTH; j++) {
                int ind = k * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH + i * CONV_OUTPUT_WIDTH + j;
                if (hw_matrix_out[ind] != sw_matrix_out[ind]) {
                    err_cnt++;
                    printf("%d,%d: %d != %d\n\r", i, j, hw_matrix_out[ind], sw_matrix_out[ind]);
                }
            }

    return err_cnt;
}

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
    for (int i = 0; i < IMAGE_CHANNELS*CONV_OFM_NUMBER*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE; i+=2) {
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
    axil_conv3D(str_in, str_out);
    for (int i = 0; i < CONV_OFM_NUMBER*CONV_OUTPUT_HEIGHT*CONV_OUTPUT_WIDTH; i+=4) {
        tmp_out = str_out.read();
        hw_image_out[i] = tmp_out.data(7, 0);
        hw_image_out[i + 1] = tmp_out.data(15, 8);
        hw_image_out[i + 2] = tmp_out.data(23, 16);
        hw_image_out[i + 3] = tmp_out.data(31, 24);
    }

    sw_convolution_3D(image_in, kernel, bias, sw_image_out);

    return check_output(sw_image_out, hw_image_out);
}
