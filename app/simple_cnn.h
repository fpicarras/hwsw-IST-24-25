#ifndef __SIMPLE_CNN__
#define __SIMPLE_CNN__

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "app_params.h"
#include "gemm.h"

/* ============================= START OF MEMORY ASSIGNMENT ============================ */
/* Align memory regions for PS-PL data transference purposes */
#define ALIGN_CONSTANT 0x1000                          /* ALIGN_CONSTANT = 4 kB */
#define ALIGN(MEM_REGION) (                            /* aligns memory region to ALIGN_CONSTANT if not aligned */ \
    MEM_REGION % ALIGN_CONSTANT == 0 ?                 /* if memory region is aligned */ \
    MEM_REGION :                                       /* does nothing else */ \
    (MEM_REGION / ALIGN_CONSTANT + 1) * ALIGN_CONSTANT /* aligns memory region to next ALIGN_CONSTANT */ \
    )

/* Size in bytes of reserved memory regions */
#define MEM_BIN_IMAGES 0x01000000
#define MEM_BIN_PARAMS 0x01000000
#define MEM_CH_IMAGES ALIGN(  \
    N_IMAGES *                \
    IMAGE_SIZE *              \
    sizeof(unsigned char)     \
    )
#define MEM_FP_PARAMS ALIGN(  \
    TOTAL_PARAMS *            \
    sizeof(float)             \
    )
#define MEM_FP_IMAGE ALIGN(   \
    IMAGE_SIZE *              \
    sizeof(float)             \
    )
#define MEM_MAT_A ALIGN(      \
    CONV_OUTPUT_WIDTH *       \
    CONV_OUTPUT_HEIGHT *      \
    CONV_KERNEL_SIZE *        \
    CONV_KERNEL_SIZE *        \
    IMAGE_CHANNELS *          \
    sizeof(float)             \
    )
#define MEM_MAT_C_V ALIGN(    \
    CONV_OUTPUT_WIDTH *       \
    CONV_OUTPUT_HEIGHT *      \
    CONV_OFM_NUMBER *         \
    sizeof(float)             \
    )
#define MEM_MAT_C_BIAS MEM_MAT_C_V
#define MEM_MAT_C_RELU MEM_MAT_C_V
#define MEM_MAT_C_POOL ALIGN( \
    POOL_OUTPUT_WIDTH *       \
    POOL_OUTPUT_HEIGHT *      \
    CONV_OFM_NUMBER *         \
    sizeof(float)             \
    )
#define MEM_MAT_CONN ALIGN(   \
    N_CLASSES *               \
    sizeof(float)             \
    )
#define MEM_MAT_CONN2 ALIGN(   \
    N_CLASSES *               \
    sizeof(int64_t)             \
    )
#define MEM_MAT_CONN_B MEM_MAT_CONN
#define MEM_MAT_SOFT_M MEM_MAT_CONN

#define MEM_TOTAL_RESERVED    \
    MEM_BIN_IMAGES +          \
    MEM_BIN_PARAMS +          \
    MEM_FP_IMAGE +            \
    MEM_MAT_A +               \
    MEM_MAT_C_V +             \
    MEM_MAT_C_BIAS +          \
    MEM_MAT_C_RELU +          \
    MEM_MAT_C_POOL +          \
    MEM_MAT_CONN +            \
    MEM_MAT_CONN_B +          \
    MEM_MAT_SOFT_M

#ifdef EMBEDDED
#define MEM_BASE_ADDR 0x10000000
#define MEM_HW_BASE_ADDR 0x13000000
#define CONVPOOL_BASE_ADDR 0x15000000
#define CONVPOOL2_BASE_ADDR 0x16000000
#define IMAGES_BASE_ADDR 0x17000000
#define TMP_BASE_ADDR 0x1D000000
#else
static unsigned char mem_array[MEM_TOTAL_RESERVED];
#define MEM_BASE_ADDR mem_array
#endif // EMBEDDED

#ifdef __IMAGES_CLASSES__
static const char image_class[10][9] = {"Airplane", "Bird", "Car", "Cat", "Deer", "Dog",
                                  "Horse", "Monkey", "Ship", "Truck"};
#endif // __IMAGES_CLASSES__

/* ============================== END OF MEMORY ASSIGNMENT ============================= */

/**
 * Assigns memory regions to required data.
 */
void init_memory(addresses * addr);

#endif // __SIMPLE_CNN__
