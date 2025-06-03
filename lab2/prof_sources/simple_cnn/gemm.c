/**
 * @file gemm.c
 * @date May, 2022
 *
 * Support file for lab2 of Hardware-Software Co-Design 2022
 *
 * Contains routines related with General Matrix Multiplication kernel.
 */
#include "gemm.h"
#include "utils.h"
#include "app_params.h"

void gemm(const float *A, const float *B, float *C, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++)
        for (int j = 0; j < colsB; j++) {
            C[i * colsB + j] = 0;
            for (int k = 0; k < colsA; k++)
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
        }
}

void gemmBias(const int16_t *A, const int32_t *B, const int16_t* bias, float *C, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++)
        for (int j = 0; j < colsB; j++) {
            int64_t acc = (int64_t) bias[i] << 26;
            for (int k = 0; k < colsA; k++)
                acc += (int64_t) A[i * colsA + k] * B[k * colsB + j];
            C[i * colsB + j] = fixed2float(acc, 41);
        }
}

void gemvOpt(const int16_t *A, const int32_t *B, const int16_t* bias, int64_t *C) {
    for (int i = 0; i < N_CLASSES; i++) {
        int64_t acc = (int64_t) bias[i] << 26;
        for (int k = 0; k < HW_MATRIX_OUT_SIZE; k++)
            acc += (int64_t) A[i * HW_MATRIX_OUT_SIZE + k] * B[k];
        C[i] = acc;
    }
}

void gemvOptT(const int16_t *A, const int32_t *B, const int16_t* bias, int64_t *C) {
    for(int i = 0; i < N_CLASSES; i++) {
        C[i] =  (int64_t) bias[i] << 26;
    }
    for (int k = 0; k < POOL_OUTPUT_SIZE; k++) {
        for (int i = 0; i < N_CLASSES; i++)
             C[i] += (int64_t) A[k * N_CLASSES + i] * B[k];
    }
}

void gemmBT(const float *A, const float *BT, float *C, int rowsA, int colsA, int rowsBT) {
    for (int i = 0; i < rowsA; i++)
        for (int j = 0; j < rowsBT; j++) {
            C[i * rowsBT + j] = 0;
            for (int k = 0; k < colsA; k++)
                C[i * rowsBT + j] += A[i * colsA + k] * BT[j * colsA + k];
        }
}

void gemmBTCT(const float *A, const float *BT, float *CT, int rowsA, int colsA, int rowsBT) {
    for (int i = 0; i < rowsBT; i++)
        /*
         *          address         size
         * dma_send(BT + i * colsA, colsA);         dma_wait_for_complete();
         * dma_send(A,              rowsA * colsA); dma_wait_for_complete();
         * dma_recv(CT + i * rowsA, colsA);         dma_wait_for_complete();
         */
        for (int j = 0; j < rowsA; j++) {
            CT[i * rowsA + j] = 0;
            for (int k = 0; k < colsA; k++)
                CT[i * rowsA + j] += A[j * colsA + k] * BT[i * colsA + k];
        }
}

void transpose(const float *C, int rows, int cols, float *CT) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            CT[j * rows + i] = C[i * cols + j];
}

void transpose_int(const int16_t *C, int rows, int cols, int16_t * CT) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            CT[j * rows + i] = C[i * cols + j];
}

void add_bias(const float *C, int rows, int cols, const float *bias, float *Cbias) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            Cbias[i * cols + j] = C[i * cols + j] + bias[i];
}

void ReLU(const float *C, int size, float *Crl) {
    for (int i = 0; i < size; i++)
        Crl[i] = C[i] < 0 ? 0 : C[i];
}
