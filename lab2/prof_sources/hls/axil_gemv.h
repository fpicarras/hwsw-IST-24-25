
#ifndef __AXIL_GEMV_H__
#define __AXIL_GEMV_H__

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include "ap_axi_sdata.h"

#define N_LINES 1849
#define N_COLUMNS 10
#define N_LIN_LOG2 11
#define NUM_BLOCKS 16
#define MATRIX_SIZE N_LINES * N_COLUMNS

// Defined signal representations in terms of inputs
// then varied input precision to evaluate precision loss
#define VIIN 6
#define VFIN 26
#define VWIN VFIN+VIIN
#define MIIN 1
#define MFIN 15
#define MWIN MFIN+MIIN
#define OWIN 64
#define OIIN VIIN+MIIN+N_LIN_LOG2
#define OFIN OWIN-OIIN

typedef ap_fixed<VWIN,VIIN> datavi_t; // Vector Input
typedef ap_fixed<MWIN,MIIN> datami_t; // Matrix Input
typedef ap_fixed<VWIN+MWIN,VIIN+MIIN> mul_t;
typedef ap_fixed<VWIN+MWIN+N_LIN_LOG2,VIIN+MIIN+N_LIN_LOG2> acc_t;

typedef ap_fixed<OWIN,OIIN> datao_t;

typedef hls::axis<datavi_t, 0, 0, 0> strmvin_t;
typedef hls::axis<datami_t, 0, 0, 0> strmmin_t;
typedef hls::axis<datao_t, 0, 0, 0> strmout_t;

void axil_gemv(
      hls::stream<strmvin_t> &strmv_in,
      hls::stream<strmmin_t> &strmm_in,
      hls::stream<strmout_t> &strm_out);

#endif // __AXIL_GEMV_H__
