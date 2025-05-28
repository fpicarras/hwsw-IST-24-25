/**
 * @file simple_cnn.c
 * @date May, 2022
 *
 * Support file for lab2 of Hardware-Software Co-Design 2022
 *
 * Contains the variables, constants, and routines related to the CNN model, as well as the main routine.
 *
 * NOTES:
 * 1) With the default memory management scheme, images.bin and weights.bin should
 *    be stored in the following memory locations:
 *    images.bin  -> 0x10000000
 *    weights.bin -> 0x11000000
 * 2) When generating the linker script, define heap and stack size at least:
 *    Heap Size  -> 10 kB
 *    Stack Size -> 10 kB
 */

#include <stdio.h>

#include "simple_cnn.h"
#include "cnn_sw.h"
#include "cnn_hw_sw.h"
#include "utils.h"
#ifdef EMBEDDED
#include "xaxidma.h"
#endif

static char image_class[10][9] = {"Airplane", "Bird", "Car", "Cat", "Deer", "Dog",
                                  "Horse", "Monkey", "Ship", "Truck"};

void init_memory(addresses * addr) {
    /* Check if memory reserved for loading files is enough */
    assert(MEM_BIN_IMAGES >= MEM_CH_IMAGES);
    assert(MEM_BIN_PARAMS >= MEM_FP_PARAMS);

    /* Assign memory regions */
    addr->ch_images = (unsigned char *) MEM_BASE_ADDR;
    addr->fp_params = (float *) ((unsigned char *) addr->ch_images + MEM_BIN_IMAGES);
    addr->fp_image = (float *) ((unsigned char *) addr->fp_params + MEM_BIN_PARAMS);
    addr->matA = (float *) ((unsigned char *) addr->fp_image + MEM_FP_IMAGE);
    addr->matCv = (float *) ((unsigned char *) addr->matA + MEM_MAT_A);
    addr->matCbias = (float *) ((unsigned char *) addr->matCv + MEM_MAT_C_V);
    addr->matCrelu = (float *) ((unsigned char *) addr->matCbias + MEM_MAT_C_BIAS);
    addr->matCpool = (float *) ((unsigned char *) addr->matCrelu + MEM_MAT_C_RELU);
    addr->matConn = (float *) ((unsigned char *) addr->matCpool + MEM_MAT_C_POOL);
    addr->matConnB = (float *) ((unsigned char *) addr->matConn + MEM_MAT_CONN);
    addr->matSoftM = (float *) ((unsigned char *) addr->matConnB + MEM_MAT_CONN_B);

    addr->int_params = (int16_t*) (MEM_HW_BASE_ADDR);
    addr->matConvPool = (int16_t*) ((unsigned char *) addr->int_params + MEM_BIN_PARAMS);
    addr->matGemm = (float*) ((unsigned char*) addr->matConvPool + MEM_MAT_C_POOL);
    addr->matSoftMax = (float*) ((unsigned char*) addr->matGemm + MEM_MAT_CONN);
    addr->image_ip = (int16_t *) IMAGE_IP_BASE_ADDR;

    /* Load images and weights from files if running in PC */
#ifndef EMBEDDED
    FILE *weights_file = fopen(WEIGHTS_FILENAME, "rb");
    assert(weights_file);
    fread((float *) addr->fp_params,
          TOTAL_PARAMS,
          sizeof(float),
          weights_file);
    fclose(weights_file);

    weights_file = fopen(WEIGHTS_Q15_FILENAME, "rb");
    assert(weights_file);
    fread((int16_t *) addr->int_params,
          TOTAL_PARAMS,
          sizeof(int16_t),
          weights_file);
    fclose(weights_file);

    FILE *images_file = fopen(IMAGES_FILENAME, "rb");
    assert(images_file);
    fread((float *) addr->ch_images,
          N_IMAGES * IMAGE_SIZE,
          sizeof(unsigned char),
          images_file);
    fclose(images_file);
#else
    for (int i = 0; i < TOTAL_PARAMS; i++) {
        addr->int_params[i] = float2fixed(addr->fp_params[i], 15);
    }
    Xil_DCacheFlushRange((INTPTR)addr->ch_images, MEM_BIN_IMAGES);
    Xil_DCacheFlushRange((INTPTR)addr->int_params, MEM_BIN_PARAMS);
#endif // EMBEDDED
}

int main() {
    printf("Start!\n");
    addresses addr;
    /* Performs memory assignment */
    init_memory(&addr);

    /* Classify first NUMBER_OF_IMAGES_TO_CLASSIFY from the dataset */
    for (int i = FIRST_IMAGE_TO_CLASSIFY - 1; i < FIRST_IMAGE_TO_CLASSIFY + NUMBER_OF_IMAGES_TO_CLASSIFY - 1; i++) {
        unsigned char *image_in = (unsigned char *) addr.ch_images + i * IMAGE_SIZE;
        /* normalize to [-1, 1] */
        normalize_image((unsigned char *) image_in, (float *) addr.fp_image);
        image_to_ip((float *) addr.fp_image, (int16_t * ) addr.image_ip);

#ifdef PRINT_IMAGE
        print_ppm(image_in);
#endif // PRINT_IMAGE

        int prediction_sw = predict_class_sw((float*) addr.fp_image, &addr);

        int prediction = predict_class_sw_hw((int16_t *) addr.image_ip, &addr);

        printf("# Image    SW %03d -> Class=%d (%8s) %3.0f%% [ ",
               i + 1, prediction_sw,
               image_class[prediction_sw],
               addr.matSoftM[prediction_sw] * 100);

        for (int i = 0; i < N_CLASSES; i++)
            printf("%3.0f%% ", addr.matSoftM[i] * 100);

        printf(prediction_sw == i % N_CLASSES ? "] OK\n\r" : "] Prediction Error\n\r");

        printf("# Image HW-SW %03d -> Class=%d (%8s) %3.0f%% [ ",
               i + 1, prediction,
               image_class[prediction],
               addr.matSoftMax[prediction] * 100);

        for (int i = 0; i < N_CLASSES; i++)
            printf("%3.0f%% ", addr.matSoftMax[i] * 100);

        printf(prediction == i % N_CLASSES ? "] OK\n\r" : "] Prediction Error\n\r");
    }
}
