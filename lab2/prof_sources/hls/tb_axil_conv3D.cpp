
#include "axil_conv3D.h"
#include <stdint.h>

#define FLT_MAX 3.402823466e+38F

static int8_t image_in[N_IMAGES*IMAGE_SIZE];
static int16_t image_in_i[IMAGE_SIZE];
static int16_t kernel_i[CONV_LAYER_WEIGHTS];
static int16_t bias_i[CONV_LAYER_BIASES];
static int32_t image_out_i[CONV_OUTPUT_SIZE];
static int32_t maxpool_i[POOL_OUTPUT_SIZE];

static float image_in_f[IMAGE_SIZE];
static float kernel_f[CONV_LAYER_WEIGHTS];
static float bias_f[CONV_OFM_NUMBER];
static float image_out_f[CONV_OUTPUT_SIZE];
static float maxpool_f[POOL_OUTPUT_SIZE];

static int32_t hw_matrix_out[HW_MATRIX_OUT_SIZE];

int float2fixed(float f, int scale) {
    f = f * (float)(1 << scale);
    f += 0.5F;
  return (int)(f);
}

float fixed2float(int i, int scale) {
  return (float)i / (float)(1 << scale);
}

void normalize_image(const unsigned char *rgb_image) {
    for (int i = 0; i < IMAGE_SIZE; i++) {
        image_in_f[i] = ((float) rgb_image[i] / 255 - 0.5F) / 0.5F;
        image_in_i[i] = float2fixed(image_in_f[i], 15);
    }   
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
  fread(image_in,
        N_IMAGES*IMAGE_SIZE,
        sizeof(unsigned char),
        images_file);
  fclose(images_file);

  for (int i = 0; i < CONV_OFM_NUMBER; i++) {
        bias_f[i] = fp_params[i];
        bias_i[i] = float2fixed(bias_f[i], 15);
  }

  for (int i = 0; i < CONV_LAYER_WEIGHTS; i++) {
        kernel_f[i] = fp_params[i + CONV_OFM_NUMBER];
        kernel_i[i] = float2fixed(kernel_f[i], 15);
  }
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
                int64_t accum = (int64_t)bias_i[l] << (ACCUM_BIT_WIDTH - INTEGER_BIT_WIDTH - WEIGHT_BIT_WIDTH); /* initialize result with bias */

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

                        int w = kernel_i[kernel_1d_idx];
                        int im = image_in_i[input_1d_idx];
                        accum +=  w * im; 
                        w = kernel_i[kernel_1d_idx + CONV_KERNEL_SIZE*CONV_KERNEL_SIZE];
                        im = image_in_i[input_1d_idx + IMAGE_HEIGHT * IMAGE_WIDTH];
                        accum += w * im;
                        w = kernel_i[kernel_1d_idx + 2*CONV_KERNEL_SIZE*CONV_KERNEL_SIZE];
                        im = image_in_i[input_1d_idx + 2 * IMAGE_HEIGHT * IMAGE_WIDTH];
                        accum += w * im; 
                    }

                /* Normalize result */
                accum >>= (ACCUM_BIT_WIDTH - MAXPOOL_BIT_WIDTH);
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

int check_output(const int32_t *sw_matrix_out_i, const float *sw_matrix_out_f) {
    // printf("SW Int Output Image\n\r");
    // for(int k = 0; k < CONV_OFM_NUMBER; k ++)
    //     for (int i = 0; i < POOL_OUTPUT_HEIGHT; i++) {
    //         for (int j = 0; j < POOL_OUTPUT_WIDTH; j++) {
    //             printf("%f ", fixed2float(sw_matrix_out_i[k * HW_MATRIX_OUT_HEIGHT * HW_MATRIX_OUT_WIDTH + i * HW_MATRIX_OUT_WIDTH + j], FRAC_BIT_WIDTH));
    //         }
    //         printf("\n\r");
    //     }

    // printf("SW Float Output Image\n\r");
    // for(int k = 0; k < CONV_OFM_NUMBER; k ++)
    //     for (int i = 0; i < POOL_OUTPUT_HEIGHT; i++) {
    //         for (int j = 0; j < POOL_OUTPUT_WIDTH; j++) {
    //             printf("%f ", sw_matrix_out_f[k * HW_MATRIX_OUT_HEIGHT * HW_MATRIX_OUT_WIDTH + i * HW_MATRIX_OUT_WIDTH + j]);
    //         }
    //         printf("\n\r");
    //     }

    // printf("HW Output Image\n\r");
    // for(int k = 0; k < CONV_OFM_NUMBER; k ++)
    //     for (int i = 0; i < HW_MATRIX_OUT_HEIGHT; i++) {
    //         for (int j = 0; j < HW_MATRIX_OUT_WIDTH; j++) {
    //             printf("%f ", fixed2float(hw_matrix_out[k * HW_MATRIX_OUT_HEIGHT * HW_MATRIX_OUT_WIDTH + i * HW_MATRIX_OUT_WIDTH + j], FRAC_BIT_WIDTH));
    //         }
    //         printf("\n\r");
    //     }

    int err_cnt = 0;
    for(int k = 0; k < CONV_OFM_NUMBER; k ++)
        for (int i = 0; i < POOL_OUTPUT_HEIGHT; i++)
            for (int j = 0; j < POOL_OUTPUT_WIDTH; j++) {
                int ind_hw = k * HW_MATRIX_OUT_HEIGHT * HW_MATRIX_OUT_WIDTH + i * HW_MATRIX_OUT_WIDTH + j;
                int ind_sw = k * POOL_OUTPUT_HEIGHT * POOL_OUTPUT_WIDTH + i * POOL_OUTPUT_WIDTH + j;
                float diff = fixed2float(hw_matrix_out[ind_hw], FRAC_BIT_WIDTH) - sw_matrix_out_f[ind_sw];
                diff = diff > 0 ? diff : -diff;
                if (hw_matrix_out[ind_hw] != sw_matrix_out_i[ind_sw]) {
                    err_cnt++;
                    // printf("Int: %d,%d,%d: %d != %d\n\r", k, i, j, hw_matrix_out[ind_hw], sw_matrix_out_i[ind_sw]);
                } else if (diff > 1E-3) {
                    err_cnt++;
                    // printf("Float: %d,%d,%d: diff = %f\n\r", k, i, j, diff);
                }
            }
    return err_cnt;
}

int main() {
    init_inputs();
    hls::stream<strmin_t> str_in;
    hls::stream<strmout_t> str_out;
    strmin_t tmp_in;
    strmout_t tmp_out;
    int err_cnt = 0;
    for(int k = 0; k < N_IMAGES; k ++) {
        if(k == 0) {
            for (int i = 0; i < CONV_LAYER_BIASES; i+=BIAS_PER_DATA) {
                for(int j = 0; j < BIAS_PER_DATA; j ++) {
                    tmp_in.data((j+1)*(BIAS_BIT_WIDTH) - 1, j*BIAS_BIT_WIDTH) = bias_i[i + j];
                }
                tmp_in.last = (ap_int<1>)0;
                str_in.write(tmp_in);
            }
            for (int i = 0; i < CONV_LAYER_WEIGHTS; i+=WEIGHTS_PER_DATA) {
                for(int j = 0; j < WEIGHTS_PER_DATA; j ++) {
                    tmp_in.data((j+1)*(WEIGHT_BIT_WIDTH) - 1, j*WEIGHT_BIT_WIDTH) = kernel_i[i + j];
                }
                tmp_in.last = (ap_int<1>)0;
                str_in.write(tmp_in);
            }
        }
        normalize_image((unsigned char *) &image_in[k*IMAGE_SIZE]);

        for (int i = 0; i < IMAGE_SIZE; i+=PIXEL_PER_DATA) {
            for(int j = 0; j < PIXEL_PER_DATA; j ++) {
                tmp_in.data((j+1)*(PIXEL_BIT_WIDTH) - 1, j*PIXEL_BIT_WIDTH) = image_in_i[i + j];
            }
            tmp_in.last = (ap_int<1>)(i == IMAGE_SIZE - PIXEL_PER_DATA);
            str_in.write(tmp_in);
        }
        axil_conv3D(str_in, str_out);
        for (int i = 0; ; i+=MAXPOOLS_PER_DATA) {
            tmp_out = str_out.read();
            // hw_matrix_out[i] = tmp_out.data;
            for(int j = 0; j < MAXPOOLS_PER_DATA; j ++) {
                hw_matrix_out[i + j] = tmp_out.data.range((j+1)*(MAXPOOL_BIT_WIDTH) - 1, j*MAXPOOL_BIT_WIDTH);
            }
            if(tmp_out.last) {
                break;
            }
        }

        sw_convolution_3D_i();
        forward_max_pool_layer_i();
        sw_convolution_3D_f();
        forward_max_pool_layer_f();

        err_cnt += check_output(maxpool_i, maxpool_f);
    }
    return err_cnt;
}
