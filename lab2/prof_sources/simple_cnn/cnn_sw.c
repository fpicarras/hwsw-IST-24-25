
#define __IMAGES_CLASSES__

#include "cnn_sw.h"
#include "app_params.h"
#include "utils.h"
#include "gemm.h"
#include "simple_cnn.h"
#include <math.h>

void compute_matrixA(const float* image, float * A) {
    for (int i = 0; i < CONV_OUTPUT_HEIGHT; i++)
        for (int j = 0; j < CONV_OUTPUT_WIDTH; j++)
            for (int k = 0; k < IMAGE_CHANNELS; k++)
                for (int x = 0; x < CONV_KERNEL_SIZE; x++)
                    for (int y = 0; y < CONV_KERNEL_SIZE; y++) {
                        /* Auxiliary matrix index */
                        int matA_1d_idx =
                                (i * CONV_OUTPUT_WIDTH + j) *               /* planar coordinate */
                                CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * IMAGE_CHANNELS +
                                k * (CONV_KERNEL_SIZE * CONV_KERNEL_SIZE) + /* OFM */
                                x * CONV_KERNEL_SIZE + y;                   /* kernel element */

                        /* Input image index */
                        int image_1d_idx =
                                k * (IMAGE_HEIGHT * IMAGE_WIDTH) +  /* OFM */
                                (x + i) * IMAGE_HEIGHT + (y + j);   /* pixel */

                        A[matA_1d_idx] = image[image_1d_idx];
                    }
}

void sw_convolution_3D(const float *image_in, const float *weights, float bias, float *image_out) {
    for (int i = 0; i < CONV_OUTPUT_HEIGHT; i++) {
        for (int j = 0; j < CONV_OUTPUT_WIDTH; j++) {
            float accum = 0;
            for (int k = 0; k < IMAGE_CHANNELS; k++) {
                for (int x = 0; x < CONV_KERNEL_SIZE; x++) {
                    for (int y = 0; y < CONV_KERNEL_SIZE; y++) {
                        /* Weights matrix index */
                        int weight_1d_idx =
                                k * (CONV_KERNEL_SIZE * CONV_KERNEL_SIZE) + /* OFM */
                                x * CONV_KERNEL_SIZE + y;                   /* planar coordinate */

                        /* Input image index */
                        int image_1d_idx =
                                k * (IMAGE_HEIGHT * IMAGE_WIDTH) +          /* OFM */
                                (x + i) * IMAGE_HEIGHT + (y + j);           /* pixel */

                        accum += weights[weight_1d_idx] * image_in[image_1d_idx];
                    }
                }
            }
            image_out[i * CONV_OUTPUT_WIDTH + j] = accum + bias;
        }
    }
}

void forward_gemm_layer(const float * image, float * A, const float * fp_params, float * C) {
    compute_matrixA(image, A);

    float *W = (float *) fp_params + CONV_OFM_NUMBER;
    float *b = (float *) fp_params;

    gemmBTCT((float *) A,
             (float *) W,
             (float *) C,
             CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH,
             CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * IMAGE_CHANNELS,
             CONV_OFM_NUMBER);

    add_bias((float *) C,
             CONV_OFM_NUMBER,
             CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH,
             (float *) b,
             (float *) C);
}

void forward_convolutional_layer(const float * fp_params, const float * image, float * conv_out) {
    for (int i = 0; i < CONV_OFM_NUMBER; i++) {
        float bias = fp_params[i];
        /* Address where the convolutional weights are stored */
        float *fp_weights =
                (float *) fp_params +                                       /* start address of params */
                CONV_OFM_NUMBER +
                i * (IMAGE_CHANNELS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE); /* kernel of OFM(i) */

        float *image_out =
                (float *) conv_out +                                        /* base address */
                i * (CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);               /* offset (number of images) */

        sw_convolution_3D((float *) image, fp_weights, bias, image_out);
    }
    ReLU((float *) conv_out,
         CONV_OFM_NUMBER * CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH,
         (float *) conv_out);
}

void forward_max_pool_layer(const float* A, float *B) {
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

                        max = (A[conv_out_1d_idx] > max) ? A[conv_out_1d_idx] : max;
                    }
                }
                /* Index of element of the pooling output */
                int pool_out_1d_idx =
                        k * (POOL_OUTPUT_HEIGHT * POOL_OUTPUT_WIDTH) + /* OFM */
                        i * POOL_OUTPUT_WIDTH +                        /* Row */
                        j;                                             /* Column */

                B[pool_out_1d_idx] = max;
            }
}

void forward_connected_layer(const float * X, const float * fp_params, float * Y) {
    float *b =
            (float *) fp_params +
            CONV_OFM_NUMBER +
            IMAGE_CHANNELS * CONV_OFM_NUMBER * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE;

    float *W =
            (float *) b +
            N_CLASSES;

    gemm(W,
         (float *) X,
         (float *) Y,
         N_CLASSES,
         POOL_OUTPUT_HEIGHT * POOL_OUTPUT_WIDTH * CONV_OFM_NUMBER,
         1);

    add_bias((float *) Y,
             N_CLASSES,
             1,
             b,
             (float *) Y);
}

int forward_softmax_layer(const float* A, float* B) {
    int best = 0;
    float sum = 0;

    /* Determine the most likely class */
    for (int i = 0; i < N_CLASSES; i++)
        if (A[i] > A[best])
            best = i;

    /* Exponential */
    for (int i = 0; i < N_CLASSES; i++) {
        B[i] = exp(A[i] - A[best]);
        sum += B[i];
    }

    /* Normalize */
    for (int i = 0; i < N_CLASSES; i++)
        B[i] /= sum;

    return best;
}

int predict_class_sw(float* image, addresses * addr) {
#if defined(EMBEDDED) && defined(PRINT_TIME_PER_LAYER)
    double t_start = xilGetMilliseconds();
#endif
#ifdef USE_GEMM
    forward_gemm_layer(image, (float*) addr->matA, (float*) addr->fp_params, (float*) addr->matCbias);
#else
    forward_convolutional_layer((float*) addr->fp_params, image, (float*) addr->matCrelu);
#endif
#if defined(EMBEDDED) && defined(PRINT_TIME_PER_LAYER)
    double t_conv = xilGetMilliseconds();
#endif
    forward_max_pool_layer((float*) addr->matCrelu, (float*) addr->matCpool);
#if defined(EMBEDDED) && defined(PRINT_TIME_PER_LAYER)
    double t_pool = xilGetMilliseconds();
#endif
    forward_connected_layer((float*) addr->matCpool, (float*) addr->fp_params, (float *) addr->matConnB);
#if defined(EMBEDDED) && defined(PRINT_TIME_PER_LAYER)
    double t_conn = xilGetMilliseconds();
#endif
    int predicted_class = forward_softmax_layer((float*) addr->matConnB, (float*) addr->matSoftM);
#if defined(EMBEDDED) && defined(PRINT_TIME_PER_LAYER)
    double t_end = xilGetMilliseconds();
#endif

#if defined(EMBEDDED) && defined(PRINT_TIME_PER_LAYER)
    printf("SW Layer 1 (Convolutional) took %.0f ms.\n\r", t_conv - t_start);
    printf("SW Layer 2 (Pooling) took %.0f ms.\n\r", t_pool - t_conv);
    printf("SW Layer 3 (Fully-Connected) took %.0f ms.\n\r", t_conn - t_pool);
    printf("SW Layer 4 (Soft-max) took %.3f ms.\n\r", t_end - t_conn);
    printf("SW Prediction took %.3f ms.\n\r", t_end - t_start);
#endif // PRINT_TIME_PER_LAYER
    return predicted_class;
}

void predict_images_sw(addresses * addr) {
    #if defined(EMBEDDED) && (defined(PRINT_TIME_PER_LAYER) || defined(PRINT_TOTAL_TIME))
        double t_start = xilGetMilliseconds();
    #endif
    for (int i = FIRST_IMAGE_TO_CLASSIFY - 1; i < FIRST_IMAGE_TO_CLASSIFY + NUMBER_OF_IMAGES_TO_CLASSIFY - 1; i++) {
        /* normalize to [-1, 1] */
        normalize_image((unsigned char *) &addr->ch_images[i * IMAGE_SIZE], (float *) &addr->fp_images[i*IMAGE_SIZE]);
    }
    /* Classify first NUMBER_OF_IMAGES_TO_CLASSIFY from the dataset */
    for (int i = FIRST_IMAGE_TO_CLASSIFY - 1; i < FIRST_IMAGE_TO_CLASSIFY + NUMBER_OF_IMAGES_TO_CLASSIFY - 1; i++) {
#ifdef PRINT_IMAGE
        print_ppm(image_in);
#endif // PRINT_IMAGE

        int prediction = predict_class_sw((float*) &addr->fp_images[i*IMAGE_SIZE], addr);

        printf("# Image    SW %03d -> Class=%d (%8s) %3.0f%% [ ",
               i + 1, prediction,
               image_class[prediction],
               addr->matSoftM[prediction] * 100);

        for (int i = 0; i < N_CLASSES; i++)
            printf("%3.0f%% ", addr->matSoftM[i] * 100);

        printf(prediction == i % N_CLASSES ? "] OK\n\r" : "] Prediction Error\n\r");
    }
    #if defined(EMBEDDED) && (defined(PRINT_TIME_PER_LAYER) || defined(PRINT_TOTAL_TIME))
        double t_end = xilGetMilliseconds();
    #endif

    #if defined(EMBEDDED) && (defined(PRINT_TIME_PER_LAYER) || defined(PRINT_TOTAL_TIME))
        printf("SW    Images Prediction took %.3f ms.\n\r", t_end - t_start);
    #endif // PRINT_TIME_PER_LAYER
}
