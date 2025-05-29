
#include "utils.h"
#include "app_params.h"

int float2fixed(float f, int scale) {
  return (int)(f * (float)(1 << scale) + 0.5F);
}

float fixed2float(int64_t i, int64_t scale) {
  return (float)i / (float)((int64_t) 1 << scale);
}

void print_fp(float *matrix, int n, char *description) {
    printf("%s\n\r", description);
    for (int i = 0; i < n; i++) {
        if ((i % N_CLASSES) == 0)
            printf("%03d: ", i);
        printf("%f ", matrix[i]);

        if ((i % N_CLASSES) == 9)
            printf("\n\r");
    }
    printf("\n\r");
}

void print_fp_mat(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("%f ", matrix[i * cols + j]);
        printf("\n\r");
    }
}

#ifdef EMBEDDED
double xilGetMilliseconds() {
    XTime time;
    XTime_GetTime(&time);
    return (double) time * 1000 / COUNTS_PER_SECOND;
}
#endif // EMBEDDED
