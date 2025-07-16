
#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdint.h>

int float2fixed(float f, int scale);

float fixed2float(int64_t i, int64_t scale);

/**
 * Prints first n elements of floating-point matrix for debug purposes.
 * @param matrix Input flattened floating-point matrix
 * @param n Size to print
 * @param description Description string
 */
void print_fp(float *matrix, int n, char *description);

/**
 * Prints entire floating-point matrix for debug purposes.
 * @param matrix Input flattened floating-point matrix
 * @param rows Number of rows
 * @param cols Number of columns
 */
void print_fp_mat(float *matrix, int rows, int cols);

#ifdef EMBEDDED
/**
 * Gets elapsed time in milliseconds in zynq device.
 */
double xilGetMilliseconds();
#endif // EMBEDDED

#endif // __UTILS_H__
