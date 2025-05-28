
#include "cnn_hw_sw.h"
#include "gemm.h"
#include "utils.h"
#include "cnn_sw.h"

void check_output(const int16_t *hw_matrix_out, const float *sw_matrix_out) {
    int err_cnt = 0;
    for(int k = 0; k < CONV_OFM_NUMBER; k ++)
        for (int i = 0; i < HW_MATRIX_OUT_HEIGHT; i++)
            for (int j = 0; j < HW_MATRIX_OUT_WIDTH; j++) {
                int ind = k * HW_MATRIX_OUT_HEIGHT * HW_MATRIX_OUT_WIDTH + i * HW_MATRIX_OUT_WIDTH + j;
                float diff = fixed2float(hw_matrix_out[ind], 10) - sw_matrix_out[ind];
                diff = diff > 0 ? diff : -diff;
                if (diff > 2E-3) {
                    err_cnt ++;
                }
            }
    printf("Number of errors: %d\n", err_cnt);
}

int predict_class_sw_hw(const int8_t * image, addresses * addr) {
#if defined(EMBEDDED) && defined(PRINT_TIME_PER_LAYER)
    double t_start = xilGetMilliseconds();
#endif
    forward_convolutional_layer_hw(image, (int16_t *)addr->int_params, (int16_t *)addr->matConvPool);
    check_output((int16_t *) addr->matConvPool, (float *) addr->matCpool);
#if defined(EMBEDDED) && defined(PRINT_TIME_PER_LAYER)
    double t_conv = xilGetMilliseconds();
#endif
    forward_connected_layer_int((int16_t *)addr->matConvPool, (int16_t *)addr->int_params, (float * ) addr->matGemm);
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

void forward_convolutional_layer_hw(const int8_t * image, const int16_t *int_params, volatile int16_t * matConvPool) {    
    // Initialize DMA
    XAxiDma dma;
    XAxiDma_Config *cfg_dma;
    cfg_dma = XAxiDma_LookupConfig(0x40400000);
    XAxiDma_CfgInitialize(&dma, cfg_dma);

    XAxiDma_IntrDisable(&dma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
    XAxiDma_IntrDisable(&dma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

    // Set Inputs on all channels
    XAxiDma_SimpleTransfer(&dma, (UINTPTR) image, IMAGE_SIZE, XAXIDMA_DMA_TO_DEVICE);

    XAxiDma_SimpleTransfer(&dma, (UINTPTR) matConvPool, sizeof(int16_t)*HW_MATRIX_OUT_SIZE, XAXIDMA_DEVICE_TO_DMA);

    while (XAxiDma_Busy(&dma,XAXIDMA_DMA_TO_DEVICE));

    XAxiDma_SimpleTransfer(&dma, (UINTPTR) int_params, sizeof(int16_t)*(CONV_LAYER_WEIGHTS + CONV_LAYER_BIASES), XAXIDMA_DMA_TO_DEVICE);

    while (XAxiDma_Busy(&dma, XAXIDMA_DEVICE_TO_DMA));

    Xil_DCacheInvalidateRange((UINTPTR) matConvPool, sizeof(int16_t)*HW_MATRIX_OUT_SIZE);
}

void forward_connected_layer_int(const int16_t *X, const int16_t * int_params, float * Y) {
    int16_t *mbias =
            (int16_t *) int_params +
            CONV_OFM_NUMBER +
            IMAGE_CHANNELS * CONV_OFM_NUMBER * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE;

    int16_t *matW =
            (int16_t *) mbias +
            N_CLASSES;

    gemmBias(matW,
         (int16_t *) X,
         mbias,
         (float *) Y,
         N_CLASSES,
         POOL_OUTPUT_HEIGHT * POOL_OUTPUT_WIDTH * CONV_OFM_NUMBER,
         1);
}
