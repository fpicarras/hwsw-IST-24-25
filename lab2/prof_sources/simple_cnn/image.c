/**
 * @file image.c
 * @date May, 2022
 *
 * Support file for lab2 of Hardware-Software Co-Design 2022
 *
 * Contains variables, constants, and routines related to the test set.
 */

#include "image.h"
#include "utils.h"

void normalize_image(const unsigned char *rgb_image, float *norm_image) {
    /* Scales image pixels to be floating-point values in range [-1, 1] */
    for (int i = 0; i < IMAGE_SIZE; i++)
        norm_image[i] = ((float) rgb_image[i] / 255 - 0.5F) / 0.5F;
}

void normalize_image16(const unsigned char *rgb_image, int16_t *image_ip) {
    /* Scales image pixels to be floating-point values in range [-1, 1] */
    for (int i = 0; i < IMAGE_SIZE; i++) {
        float tmp = 2*((float) rgb_image[i] / 255 - 0.5F);
        image_ip[i] = (int16_t)(tmp * (float)(1UL << 15UL) + 0.5F);
    }
}

void print_ppm(unsigned char *rgb_image) {
    printf("P3\r\n%d %d 255\r\n", IMAGE_WIDTH, IMAGE_HEIGHT);
    for (int i = 0, k = 0; i < IMAGE_WIDTH; i++)
        for (int j = 0; j < IMAGE_HEIGHT; j++) {
            printf("%3d ", IMAGE_IN_R(i, j));
            printf("%3d ", IMAGE_IN_G(i, j));
            printf("%3d ", IMAGE_IN_B(i, j));
            if ((++k % 8) == 0) printf("\r\n");
        }
    printf("\r\n");
}

void print_fp_image(float *norm_image) {
    for (int z = 0; z < 3; z++)
        for (int y = 0; y < IMAGE_HEIGHT; y++) {
            for (int x = 0; x < IMAGE_WIDTH; x++)
                printf("%f ", norm_image[z * (IMAGE_WIDTH * IMAGE_HEIGHT) + y * IMAGE_WIDTH + x]);
            printf("\n\r");
        }
}
