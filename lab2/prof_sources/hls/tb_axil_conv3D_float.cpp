
#include "tb_axil_conv3D_float.h"
#include "axil_conv3D.h"
#include <stdio.h>

static float image_in_f[IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH];
static float kernel_f[IMAGE_CHANNELS*CONV_OFM_NUMBER*CONV_KERNEL_SIZE * CONV_KERNEL_SIZE];
static float bias_f[CONV_OFM_NUMBER];
static float image_out_f[CONV_OFM_NUMBER * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH];

int float2fixed(float f, int scale) {
  return (int)(f * (float)(1 << scale) + 0.5F);
}

float fixed2float(int i, int scale) {
  return (float)i / (float)(1 << scale);
}

void init_inputs_f(int8_t *image_in, int16_t * kernel, int16_t * bias) {

  float fp_params [CONV_LAYER_PARAMS];
  FILE *weights_file = fopen(WEIGHTS_FILENAME, "rb");
  assert(weights_file);
  fread((float *) fp_params,
        CONV_LAYER_PARAMS,
        sizeof(float),
        weights_file);
  fclose(weights_file);

  FILE *images_file = fopen(IMAGES_FILENAME, "rb");
  assert(images_file);
  fread(image_in,
        IMAGE_SIZE,
        sizeof(unsigned char),
        images_file);
  fclose(images_file);

  printf("Input Image\n\r");
  for(int k = 0; k < IMAGE_CHANNELS; k++) {
      for (int i = 0; i < IMAGE_HEIGHT; i++) {
          for (int j = 0; j < IMAGE_WIDTH; j++) {
              int ind = k*IMAGE_HEIGHT*IMAGE_WIDTH + i * IMAGE_WIDTH + j;
              image_in_f[ind] = fixed2float(image_in[ind], 7);
              printf("%4d ", image_in[ind]);
          }
          printf("\n");
      }
      printf("\n");
  }

  printf("Kernel\n\r");
  for (int i = 0; i < CONV_OFM_NUMBER; i++) {
    for(int l = 0; l < IMAGE_CHANNELS; L++){
        for (int j = 0; j < CONV_KERNEL_SIZE * CONV_KERNEL_SIZE; j++) {
          int ind = (l+i*IMAGE_CHANNELS) * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE + j;
          kernel_f[ind] = fp_params[ind + CONV_OFM_NUMBER];
          kernel[ind] = float2fixed(kernel_f[ind], 15);
          printf("%6d ", kernel[ind]);
      }
      printf("\n");
    }
    printf("\n");
  }

  printf("Bias\n\r");
  for (int i = 0; i < CONV_OFM_NUMBER; i++) {
      bias_f[i] = fp_params[i];
      bias[i] = float2fixed(bias_f[i], 15);
      printf("%6d ", bias[i]);
  }
  printf("\n");
}

void sw_convolution_3D_f() {
    for(int l = 0; l < CONV_OFM_NUMBER; l ++)
        for (int i = 0; i < CONV_OUTPUT_HEIGHT; i++)
            for (int j = 0; j < CONV_OUTPUT_WIDTH; j++) {
                int accum = bias_f[l];                       /* initialize result with bias */

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

                        accum += kernel_f[kernel_1d_idx] * image_in_f[input_1d_idx];
                        accum += kernel_f[kernel_1d_idx + CONV_KERNEL_SIZE*CONV_KERNEL_SIZE] * image_in_f[input_1d_idx + IMAGE_HEIGHT * IMAGE_WIDTH];
                        accum += kernel_f[kernel_1d_idx + 2*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE] * image_in_f[input_1d_idx + 2 * IMAGE_HEIGHT * IMAGE_WIDTH];
                    }

                /* Normalize result */
                if (accum < 0)
                    accum = 0;

                image_out_f[l*CONV_OUTPUT_HEIGHT*CONV_OUTPUT_WIDTH + i * CONV_OUTPUT_WIDTH + j] = accum;
            }
}

int check_output_f(const float *sw_matrix_out, const int8_t *hw_matrix_out) {
    printf("Output Image SW\n\r");
    for(int k = 0; k < CONV_OFM_NUMBER; k ++)
        for (int i = 0; i < CONV_OUTPUT_HEIGHT; i++) {
            for (int j = 0; j < CONV_OUTPUT_WIDTH; j++) {
                printf("%f ", sw_matrix_out[k * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH + i * CONV_OUTPUT_WIDTH + j]);
            }
            printf("\n\r");
        }

    printf("Output Image HW\n\r");
    for(int k = 0; k < CONV_OFM_NUMBER; k ++)
        for (int i = 0; i < CONV_OUTPUT_HEIGHT; i++) {
            for (int j = 0; j < CONV_OUTPUT_WIDTH; j++) {
                printf("%f ", fixed2float(hw_matrix_out[k * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH + i * CONV_OUTPUT_WIDTH + j], 7));
            }
            printf("\n\r");
        }

    int err_cnt = 0;
    // for(int k = 0; k < CONV_OFM_NUMBER; k ++)
    //     for (int i = 0; i < CONV_OUTPUT_HEIGHT; i++)
    //         for (int j = 0; j < CONV_OUTPUT_WIDTH; j++) {
    //             int ind = k * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH + i * CONV_OUTPUT_WIDTH + j;
    //             if (hw_matrix_out[ind] != sw_matrix_out[ind]) {
    //                 err_cnt++;
    //                 printf("%d,%d: %d != %d\n\r", i, j, hw_matrix_out[ind], sw_matrix_out[ind]);
    //             }
    //         }

    return err_cnt;
}
