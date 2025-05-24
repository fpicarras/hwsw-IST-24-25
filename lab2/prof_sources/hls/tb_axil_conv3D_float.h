
#ifndef __TB_AXIL_CONV3D_FLOAT_H__
#define __TB_AXIL_CONV3D_FLOAT_H__

#include <stdint.h>

void init_inputs_f(int8_t *image_in, int16_t * kernel, int16_t * bias);

void sw_convolution_3D_f();

int check_output_f(const int8_t *hw_matrix_out);

#endif // __TB_AXIL_CONV3D_FLOAT_H__
