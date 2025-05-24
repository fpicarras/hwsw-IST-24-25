
#ifndef __TB_AXIL_CONV3D_H__
#define __TB_AXIL_CONV3D_H__

#include <stdint.h>

void init_inputs(int8_t *image_in, int16_t * kernel, int16_t * bias);

void sw_convolution_3D(const int8_t *matrix_in, const int16_t * kernel, const int16_t * bias, int8_t *matrix_out);

int check_output(const int8_t *sw_matrix_out, int8_t *hw_matrix_out);

#endif // __TB_AXIL_CONV3D_H__