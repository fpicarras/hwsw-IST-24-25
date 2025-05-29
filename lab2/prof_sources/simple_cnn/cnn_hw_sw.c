
#define __IMAGES_CLASSES__

#include "cnn_hw_sw.h"
#include "gemm.h"
#include "image.h"
#include "utils.h"
#include "cnn_sw.h"
#include "simple_cnn.h"

int predict_class_hw_sw(int16_t * image, addresses * addr, bool first) {
#if defined(EMBEDDED) && defined(PRINT_TIME_PER_LAYER)
    double t_start = xilGetMilliseconds();
#endif
    forward_convolutional_layer_hw(image, (int16_t *)addr->int_params, (int32_t *)addr->matConvPool, first);
#if defined(EMBEDDED) && defined(PRINT_TIME_PER_LAYER)
    double t_conv = xilGetMilliseconds();
#endif
    forward_connected_layer_int((int32_t *)addr->matConvPool, (int16_t *)addr->int_params, (float * ) addr->matGemm);
#if defined(EMBEDDED) && defined(PRINT_TIME_PER_LAYER)
    double t_conn = xilGetMilliseconds();
#endif
    int predicted_class = forward_softmax_layer((float * ) addr->matGemm, (float*) addr->matSoftMax);
#if defined(EMBEDDED) && defined(PRINT_TIME_PER_LAYER)
    double t_end = xilGetMilliseconds();
#endif

#if defined(EMBEDDED) && defined(PRINT_TIME_PER_LAYER)
    printf("SW-HW Layer 1 (Convolutional+Pooling) took %.0f ms.\n\r", t_conv - t_start);
    printf("SW-HW Layer 3 (Fully-Connected) took %.0f ms.\n\r", t_conn - t_conv);
    printf("SW-HW Layer 4 (Soft-max) took %.3f ms.\n\r", t_end - t_conn);
    printf("SW-HW Prediction took %.3f ms.\n\r", t_end - t_start);
#endif // PRINT_TIME_PER_LAYER
    return predicted_class;
}

void forward_convolutional_layer_hw(const int16_t * image, const int16_t *int_params, volatile int32_t * matConvPool, bool first) {  
    Xil_DCacheFlushRange((INTPTR)image, sizeof(int16_t)*IMAGE_SIZE);  
    // Initialize DMA
    XAxiDma dma;
    XAxiDma_Config *cfg_dma;
    cfg_dma = XAxiDma_LookupConfig(0x40400000);
    XAxiDma_CfgInitialize(&dma, cfg_dma);

    XAxiDma_IntrDisable(&dma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
    XAxiDma_IntrDisable(&dma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

    if(first) {
        XAxiDma_SimpleTransfer(&dma, (UINTPTR) int_params, sizeof(int16_t)*(CONV_LAYER_WEIGHTS + CONV_LAYER_BIASES), XAXIDMA_DMA_TO_DEVICE);
    }

    XAxiDma_SimpleTransfer(&dma, (UINTPTR) matConvPool, sizeof(int32_t)*HW_MATRIX_OUT_SIZE, XAXIDMA_DEVICE_TO_DMA);

    while (first && XAxiDma_Busy(&dma,XAXIDMA_DMA_TO_DEVICE));

    XAxiDma_SimpleTransfer(&dma, (UINTPTR) image, sizeof(int16_t)*IMAGE_SIZE, XAXIDMA_DMA_TO_DEVICE);

    while (XAxiDma_Busy(&dma, XAXIDMA_DEVICE_TO_DMA));

    Xil_DCacheInvalidateRange((UINTPTR) matConvPool, sizeof(int32_t)*HW_MATRIX_OUT_SIZE);
}

void forward_connected_layer_int(const int32_t *X, const int16_t * int_params, float * Y) {
    int16_t *mbias =
            (int16_t *) int_params +
            CONV_OFM_NUMBER +
            IMAGE_CHANNELS * CONV_OFM_NUMBER * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE;

    int16_t *matW =
            (int16_t *) mbias +
            N_CLASSES;

    gemmBias(matW,
         (int32_t *) X,
         mbias,
         (float *) Y,
         N_CLASSES,
         POOL_OUTPUT_SIZE,
         1);
}

void predict_images_hw_sw(addresses * addr) {
    #if defined(EMBEDDED) && (defined(PRINT_TIME_PER_LAYER) || defined(PRINT_TOTAL_TIME))
        double t_start = xilGetMilliseconds();
    #endif
    for (int i = FIRST_IMAGE_TO_CLASSIFY - 1; i < FIRST_IMAGE_TO_CLASSIFY + NUMBER_OF_IMAGES_TO_CLASSIFY - 1; i++) {
        normalize_image((unsigned char *) &addr->ch_images[i * IMAGE_SIZE], (float *) &addr->fp_images[i*IMAGE_SIZE]);
        image_to_ip((float *) &addr->fp_images[i * IMAGE_SIZE], (int16_t * ) &addr->int_images[i * IMAGE_SIZE]);
    }
    /* Classify first NUMBER_OF_IMAGES_TO_CLASSIFY from the dataset */
    for (int i = FIRST_IMAGE_TO_CLASSIFY - 1; i < FIRST_IMAGE_TO_CLASSIFY + NUMBER_OF_IMAGES_TO_CLASSIFY - 1; i++) {
#ifdef PRINT_IMAGE
        print_ppm(image_in);
#endif // PRINT_IMAGE

        int prediction = predict_class_hw_sw((int16_t *) &addr->int_images[i*IMAGE_SIZE], addr, i == 0);

        printf("# Image HW-SW %03d -> Class=%d (%8s) %3.0f%% [ ",
               i + 1, prediction,
               image_class[prediction],
               addr->matSoftMax[prediction] * 100);

        for (int i = 0; i < N_CLASSES; i++)
            printf("%3.0f%% ", addr->matSoftMax[i] * 100);

        printf(prediction == i % N_CLASSES ? "] OK\n\r" : "] Prediction Error\n\r");
    }
    #if defined(EMBEDDED) && (defined(PRINT_TIME_PER_LAYER) || defined(PRINT_TOTAL_TIME))
        double t_end = xilGetMilliseconds();
    #endif

    #if defined(EMBEDDED) && (defined(PRINT_TIME_PER_LAYER) || defined(PRINT_TOTAL_TIME))
        printf("SW-HW Images Prediction took %.3f ms.\n\r", t_end - t_start);
    #endif // PRINT_TIME_PER_LAYER
}
