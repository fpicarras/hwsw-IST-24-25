
#define __IMAGES_CLASSES__

#include "cnn_hw_sw.h"
#include "gemm.h"
#include "image.h"
#include "utils.h"
#include "cnn_sw.h"
#include "simple_cnn.h"

void init_sw_pipeline(XAxiDma *dma, addresses * addr) {
    // Initialize DMA
    XAxiDma_Config *cfg_dma;
    cfg_dma = XAxiDma_LookupConfig(0x40400000);
    XAxiDma_CfgInitialize(dma, cfg_dma);

    XAxiDma_IntrDisable(dma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
    XAxiDma_IntrDisable(dma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

    XAxiDma_SimpleTransfer(dma, (UINTPTR) addr->int_params, sizeof(int16_t)*(CONV_LAYER_WEIGHTS + CONV_LAYER_BIASES), XAXIDMA_DMA_TO_DEVICE);

    normalize_image16((unsigned char *) &addr->ch_images[FIRST_IMAGE_TO_CLASSIFY - 1], (int16_t * ) addr->int_images);
    Xil_DCacheFlushRange((INTPTR)addr->int_images, sizeof(int16_t)*IMAGE_SIZE);

    while (XAxiDma_Busy(dma, XAXIDMA_DMA_TO_DEVICE));

    XAxiDma_SimpleTransfer(dma, (UINTPTR) addr->int_images, sizeof(int16_t)*IMAGE_SIZE, XAXIDMA_DMA_TO_DEVICE);

    XAxiDma_SimpleTransfer(dma, (UINTPTR) addr->matConvPool[addr->matConvPoolInd], sizeof(int32_t)*HW_MATRIX_OUT_SIZE, XAXIDMA_DEVICE_TO_DMA);

    while (XAxiDma_Busy(dma, XAXIDMA_DMA_TO_DEVICE));
}

int predict_class_hw_sw(addresses * addr, XAxiDma *dma) {
#if defined(EMBEDDED) && defined(PRINT_TIME_PER_LAYER)
    double t_start = xilGetMilliseconds();
#endif
    forward_convolutional_layer_hw(addr, dma);
#if defined(EMBEDDED) && defined(PRINT_TIME_PER_LAYER)
    double t_conv = xilGetMilliseconds();
#endif
    forward_connected_layer_int((int32_t *)addr->matConvPool[addr->matConvPoolInd], (int16_t *)addr->int_params, (int64_t * ) addr->matGemm);
#if defined(EMBEDDED) && defined(PRINT_TIME_PER_LAYER)
    double t_conn = xilGetMilliseconds();
#endif
    int predicted_class = forward_softmax_layer_int((int64_t * ) addr->matGemm, (float*) addr->vecSoftMax);
#if defined(EMBEDDED) && defined(PRINT_TIME_PER_LAYER)
    double t_end = xilGetMilliseconds();
#endif

#if defined(EMBEDDED) && defined(PRINT_TIME_PER_LAYER)
    printf("SW-HW Layer 1 (Convolutional+Pooling) took %.3f ms.\n\r", t_conv - t_start);
    printf("SW-HW Layer 3 (Fully-Connected) took %.3f ms.\n\r", t_conn - t_conv);
    printf("SW-HW Layer 4 (Soft-max) took %.3f ms.\n\r", t_end - t_conn);
    printf("SW-HW Prediction took %.3f ms.\n\r", t_end - t_start);
#endif // PRINT_TIME_PER_LAYER
    return predicted_class;
}

void forward_convolutional_layer_hw(addresses * addr, XAxiDma *dma) {

    normalize_image16((unsigned char *) addr->nextImage, (int16_t * ) addr->int_images);

    Xil_DCacheFlushRange((INTPTR)addr->int_images, sizeof(int16_t)*IMAGE_SIZE);

    XAxiDma_SimpleTransfer(dma, (UINTPTR) addr->int_images, sizeof(int16_t)*IMAGE_SIZE, XAXIDMA_DMA_TO_DEVICE);

    while (XAxiDma_Busy(dma, XAXIDMA_DEVICE_TO_DMA));

    XAxiDma_SimpleTransfer(dma, (UINTPTR) addr->matConvPool[(addr->matConvPoolInd + 1) & 0x1], sizeof(int32_t)*HW_MATRIX_OUT_SIZE, XAXIDMA_DEVICE_TO_DMA);

    Xil_DCacheInvalidateRange((UINTPTR) addr->matConvPool[addr->matConvPoolInd], sizeof(int32_t)*HW_MATRIX_OUT_SIZE);

    while (XAxiDma_Busy(dma, XAXIDMA_DMA_TO_DEVICE));
}

void forward_connected_layer_int(const int32_t *X, const int16_t * int_params, int64_t * Y) {
    int16_t *mbias =
            (int16_t *) int_params +
            CONV_OFM_NUMBER +
            IMAGE_CHANNELS * CONV_OFM_NUMBER * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE;

    int16_t *matW =
            (int16_t *) mbias +
            N_CLASSES;

    gemvOpt(matW, X, mbias, Y);
}

int forward_softmax_layer_int(const int64_t* A, float* B) {
    float A2[N_CLASSES];

    for (int i = 0; i < N_CLASSES; i++)
        A2[i] = fixed2float(A[i], 41);

    return forward_softmax_layer(A2, B);
}

void predict_images_hw_sw(addresses * addr) {
    int predictions[N_CLASSES];
    #if defined(EMBEDDED) && (defined(PRINT_TIME_PER_LAYER) || defined(PRINT_TOTAL_TIME))
        double t_start = xilGetMilliseconds();
    #endif
    XAxiDma dma;
    init_sw_pipeline(&dma, addr);
    /* Classify first NUMBER_OF_IMAGES_TO_CLASSIFY from the dataset */
    for (int i = FIRST_IMAGE_TO_CLASSIFY - 1; i < FIRST_IMAGE_TO_CLASSIFY + NUMBER_OF_IMAGES_TO_CLASSIFY - 1; i++) {
#ifdef PRINT_IMAGE
        print_ppm(image_in);
#endif // PRINT_IMAGE
        int ind = (i == (FIRST_IMAGE_TO_CLASSIFY + NUMBER_OF_IMAGES_TO_CLASSIFY - 1)) ? 0 : (i + 1);
        addr->nextImage = &addr->ch_images[ind*IMAGE_SIZE];
        addr->vecSoftMax = &addr->matSoftMax[i*N_CLASSES];
        predictions[i] = predict_class_hw_sw(addr, &dma);
        addr->matConvPoolInd = (addr->matConvPoolInd + 1) & 0x1;
    }
    #if defined(EMBEDDED) && (defined(PRINT_TIME_PER_LAYER) || defined(PRINT_TOTAL_TIME))
        double t_end = xilGetMilliseconds();
    #endif

    #if defined(EMBEDDED) && (defined(PRINT_TIME_PER_LAYER) || defined(PRINT_TOTAL_TIME))
        printf("SW-HW Images Prediction took %.3f ms.\n\r", t_end - t_start);
    #endif // PRINT_TIME_PER_LAYER
    for (int i = FIRST_IMAGE_TO_CLASSIFY - 1; i < FIRST_IMAGE_TO_CLASSIFY + NUMBER_OF_IMAGES_TO_CLASSIFY - 1; i++) {
        printf("# Image HW-SW %03d -> Class=%d (%8s) %3.0f%% [ ",
               i + 1, predictions[i],
               image_class[predictions[i]],
               addr->matSoftMax[i*N_CLASSES + predictions[i]] * 100);

        for (int j = 0; j < N_CLASSES; j++)
            printf("%3.0f%% ", addr->matSoftMax[i*N_CLASSES + j] * 100);

        printf(predictions[i] == i % N_CLASSES ? "] OK\n\r" : "] Prediction Error\n\r");
    }
}
