
#include "axil_conv3D.h"
#include <stdint.h>

#define FLT_MAX 3.402823466e+38F

static int8_t image_in_i[IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH];
static int16_t kernel_i[IMAGE_CHANNELS*CONV_OFM_NUMBER*CONV_KERNEL_SIZE * CONV_KERNEL_SIZE];
static int16_t bias_i[CONV_OFM_NUMBER];
static int16_t image_out_i[CONV_OFM_NUMBER * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH];
static int16_t maxpool_i[CONV_OFM_NUMBER * POOL_OUTPUT_HEIGHT * POOL_OUTPUT_WIDTH];

static float image_in_f[IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH];
static float kernel_f[IMAGE_CHANNELS*CONV_OFM_NUMBER*CONV_KERNEL_SIZE * CONV_KERNEL_SIZE];
static float bias_f[CONV_OFM_NUMBER];
static float image_out_f[CONV_OFM_NUMBER * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH];
static float maxpool_f[CONV_OFM_NUMBER * POOL_OUTPUT_HEIGHT * POOL_OUTPUT_WIDTH];

static int16_t hw_matrix_out[CONV_OFM_NUMBER * HW_MATRIX_OUT_HEIGHT * HW_MATRIX_OUT_WIDTH];

int float2fixed(float f, int scale) {
  return (int)(f * (float)(1 << scale) + 0.5F);
}

float fixed2float(int i, int scale) {
  return (float)i / (float)(1 << scale);
}

void init_inputs() {

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
  fread(image_in_i,
        IMAGE_SIZE,
        sizeof(unsigned char),
        images_file);
  fclose(images_file);

  printf("Input Image\n\r");
  for(int k = 0; k < IMAGE_CHANNELS; k++) {
      for (int i = 0; i < IMAGE_HEIGHT; i++) {
          for (int j = 0; j < IMAGE_WIDTH; j++) {
              int ind = k*IMAGE_HEIGHT*IMAGE_WIDTH + i * IMAGE_WIDTH + j;
              image_in_f[ind] = fixed2float(image_in_i[ind], 7);
              printf("%f %4d ", image_in_f[ind], image_in_i[ind]);
          }
          printf("\n");
      }
      printf("\n");
  }

  printf("Kernel\n\r");
  for (int i = 0; i < CONV_OFM_NUMBER; i++) {
    for(int l = 0; l < IMAGE_CHANNELS; l++){
        for (int j = 0; j < CONV_KERNEL_SIZE * CONV_KERNEL_SIZE; j++) {
          int ind = (l+i*IMAGE_CHANNELS) * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE + j;
          kernel_f[ind] = fp_params[ind + CONV_OFM_NUMBER];
          kernel_i[ind] = float2fixed(kernel_f[ind], 15);
          printf("%f %4d ", kernel_f[ind], kernel_i[ind]);
      }
      printf("\n");
    }
    printf("\n");
  }

  printf("Bias\n\r");
  for (int i = 0; i < CONV_OFM_NUMBER; i++) {
      bias_f[i] = fp_params[i];
      bias_i[i] = float2fixed(bias_f[i], 15);
      printf("%f %4d ", bias_f[i], bias_i[i]);
  }
  printf("\n");
}

void sw_convolution_3D_f() {
    for(int l = 0; l < CONV_OFM_NUMBER; l ++)
        for (int i = 0; i < CONV_OUTPUT_HEIGHT; i++)
            for (int j = 0; j < CONV_OUTPUT_WIDTH; j++) {
                float accum = bias_f[l];                       /* initialize result with bias */

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

                        float w = kernel_f[kernel_1d_idx];
                        float im = image_in_f[input_1d_idx];
                        accum +=  w * im; 
                        w = kernel_f[kernel_1d_idx + CONV_KERNEL_SIZE*CONV_KERNEL_SIZE];
                        im = image_in_f[input_1d_idx + IMAGE_HEIGHT * IMAGE_WIDTH];
                        accum += w * im;
                        w = kernel_f[kernel_1d_idx + 2*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE];
                        im = image_in_f[input_1d_idx + 2 * IMAGE_HEIGHT * IMAGE_WIDTH];
                        accum += w * im; 
                    }

                /* Normalize result */
                if (accum < 0)
                    accum = 0;

                image_out_f[l*CONV_OUTPUT_HEIGHT*CONV_OUTPUT_WIDTH + i * CONV_OUTPUT_WIDTH + j] = accum;
            }
}

void forward_max_pool_layer_f() {
    for (int i = 0; i < POOL_OUTPUT_HEIGHT; i++)
        for (int j = 0; j < POOL_OUTPUT_WIDTH; j++)
            for (int k = 0; k < CONV_OFM_NUMBER; k++) {
                float max = -FLT_MAX;
                for (int x = 0; x < POOL_KERNEL_SIZE; x++) {
                    for (int y = 0; y < POOL_KERNEL_SIZE; y++) {
                        /* Index of element of convolution output */
                        int conv_out_1d_idx =
                                k * (CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH) + /* OFM */
                                (i * POOL_STRIDE + x) * CONV_OUTPUT_WIDTH +    /* Row */
                                j * POOL_STRIDE + y;                           /* Column */

                        max = (image_out_f[conv_out_1d_idx] > max) ? image_out_f[conv_out_1d_idx] : max;
                    }
                }
                /* Index of element of the pooling output */
                int pool_out_1d_idx =
                        k * (POOL_OUTPUT_HEIGHT * POOL_OUTPUT_WIDTH) + /* OFM */
                        i * POOL_OUTPUT_WIDTH +                        /* Row */
                        j;                                             /* Column */

                maxpool_f[pool_out_1d_idx] = max;
            }
}

void sw_convolution_3D_i() {
    for(int l = 0; l < CONV_OFM_NUMBER; l ++)
        for (int i = 0; i < CONV_OUTPUT_HEIGHT; i++)
            for (int j = 0; j < CONV_OUTPUT_WIDTH; j++) {
                int accum = (int)bias_i[l] << (ACCUM_BIT_WIDTH - INTEGER_BIT_WIDTH - WEIGHT_BIT_WIDTH); /* initialize result with bias */

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

                        accum += kernel_i[kernel_1d_idx] * image_in_i[input_1d_idx];
                        accum += kernel_i[kernel_1d_idx + CONV_KERNEL_SIZE*CONV_KERNEL_SIZE] * image_in_i[input_1d_idx + IMAGE_HEIGHT * IMAGE_WIDTH];
                        accum += kernel_i[kernel_1d_idx + 2*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE] * image_in_i[input_1d_idx + 2 * IMAGE_HEIGHT * IMAGE_WIDTH];
                    }

                /* Normalize result */
                accum >>= (ACCUM_BIT_WIDTH - OUTPUT_BIT_WIDTH);
                if (accum < 0)
                    accum = 0;

                image_out_i[l*CONV_OUTPUT_HEIGHT*CONV_OUTPUT_WIDTH + i * CONV_OUTPUT_WIDTH + j] = accum;
            }
}

void forward_max_pool_layer_i() {
    for (int i = 0; i < POOL_OUTPUT_HEIGHT; i++)
        for (int j = 0; j < POOL_OUTPUT_WIDTH; j++)
            for (int k = 0; k < CONV_OFM_NUMBER; k++) {
                int max = -1;
                for (int x = 0; x < POOL_KERNEL_SIZE; x++) {
                    for (int y = 0; y < POOL_KERNEL_SIZE; y++) {
                        /* Index of element of convolution output */
                        int conv_out_1d_idx =
                                k * (CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH) + /* OFM */
                                (i * POOL_STRIDE + x) * CONV_OUTPUT_WIDTH +    /* Row */
                                j * POOL_STRIDE + y;                           /* Column */
                        int conv = image_out_i[conv_out_1d_idx]; 
                        max = (conv > max) ? conv : max;
                    }
                }
                /* Index of element of the pooling output */
                int pool_out_1d_idx =
                        k * (POOL_OUTPUT_HEIGHT * POOL_OUTPUT_WIDTH) + /* OFM */
                        i * POOL_OUTPUT_WIDTH +                        /* Row */
                        j;                                             /* Column */

                maxpool_i[pool_out_1d_idx] = max;
            }
}

int check_output(const int16_t *sw_matrix_out_i, const float *sw_matrix_out_f) {
    printf("SW Int Output Image\n\r");
    for(int k = 0; k < CONV_OFM_NUMBER; k ++)
        for (int i = 0; i < HW_MATRIX_OUT_HEIGHT; i++) {
            for (int j = 0; j < HW_MATRIX_OUT_WIDTH; j++) {
                printf("%4d ", sw_matrix_out_i[k * HW_MATRIX_OUT_HEIGHT * HW_MATRIX_OUT_WIDTH + i * HW_MATRIX_OUT_WIDTH + j]);
            }
            printf("\n\r");
        }

    printf("SW Float Output Image\n\r");
    for(int k = 0; k < CONV_OFM_NUMBER; k ++)
        for (int i = 0; i < HW_MATRIX_OUT_HEIGHT; i++) {
            for (int j = 0; j < HW_MATRIX_OUT_WIDTH; j++) {
                printf("%f ", sw_matrix_out_f[k * HW_MATRIX_OUT_HEIGHT * HW_MATRIX_OUT_WIDTH + i * HW_MATRIX_OUT_WIDTH + j]);
            }
            printf("\n\r");
        }

    printf("HW Output Image\n\r");
    for(int k = 0; k < CONV_OFM_NUMBER; k ++)
        for (int i = 0; i < HW_MATRIX_OUT_HEIGHT; i++) {
            for (int j = 0; j < HW_MATRIX_OUT_WIDTH; j++) {
                printf("%4d ", hw_matrix_out[k * HW_MATRIX_OUT_HEIGHT * HW_MATRIX_OUT_WIDTH + i * HW_MATRIX_OUT_WIDTH + j]);
            }
            printf("\n\r");
        }

    int err_cnt = 0;
    for(int k = 0; k < CONV_OFM_NUMBER; k ++)
        for (int i = 0; i < HW_MATRIX_OUT_HEIGHT; i++)
            for (int j = 0; j < HW_MATRIX_OUT_WIDTH; j++) {
                int ind = k * HW_MATRIX_OUT_HEIGHT * HW_MATRIX_OUT_WIDTH + i * HW_MATRIX_OUT_WIDTH + j;
                float diff = fixed2float(hw_matrix_out[ind], (OUTPUT_BIT_WIDTH - INTEGER_BIT_WIDTH - 1)) - sw_matrix_out_f[ind];
                diff = diff > 0 ? diff : -diff;
                if (hw_matrix_out[ind] != sw_matrix_out_i[ind]) {
                    err_cnt++;
                    printf("Int: %d,%d: %d != %d\n\r", i, j, hw_matrix_out[ind], sw_matrix_out_i[ind]);
                } else if (diff > 1E-3) {
                    err_cnt++;
                    printf("Float: %d,%d: diff = %f\n\r", i, j, diff);
                }
            }

    return err_cnt;
}

int main() {
    init_inputs();
    hls::stream<strmio_t> str_in;
    hls::stream<strmio_t> str_out;
    strmio_t tmp_in, tmp_out;
    for (int i = 0; i < IMAGE_CHANNELS*IMAGE_HEIGHT*IMAGE_WIDTH; i+=4) {
        tmp_in.data(7, 0) = image_in_i[i];
        tmp_in.data(15, 8) = image_in_i[i + 1];
        tmp_in.data(23, 16) = image_in_i[i + 2];
        tmp_in.data(31, 24) = image_in_i[i + 3];
        tmp_in.last = (ap_int<1>)0;
        str_in.write(tmp_in);
    }
    for (int i = 0; i < IMAGE_CHANNELS*CONV_OFM_NUMBER*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE; i+=2) {
        tmp_in.data(15, 0) = kernel_i[i];
        tmp_in.data(31, 16) = kernel_i[i + 1];
        tmp_in.last = (ap_int<1>)0;
        str_in.write(tmp_in);
    }
    for (int i = 0; i < CONV_OFM_NUMBER; i+=2) {
        tmp_in.data(15, 0) = bias_i[i];
        tmp_in.data(31, 16) = bias_i[i + 1];
        tmp_in.last = (ap_int<1>)(i == (CONV_OFM_NUMBER - 2));
        str_in.write(tmp_in);
    }
    axil_conv3D(str_in, str_out);
    for (int i = 0; i < CONV_OFM_NUMBER*HW_MATRIX_OUT_HEIGHT*HW_MATRIX_OUT_WIDTH; i+=2) {
        tmp_out = str_out.read();
        hw_matrix_out[i] = tmp_out.data(15, 0);
        hw_matrix_out[i + 1] = tmp_out.data(31, 16);
    }
    int err_cnt = 0;

    sw_convolution_3D_i();
    forward_max_pool_layer_i();
    sw_convolution_3D_f();
    forward_max_pool_layer_f();

    return check_output(maxpool_i, maxpool_f);
}
