#include "axil_gemv.h"

// The top-level function
void axil_gemv(
      hls::stream<strmvin_t> &strmv_in,
      hls::stream<strmmin_t> &strmm_in,
      hls::stream<strmout_t> &strm_out) {
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS interface axis port=strmv_in
#pragma HLS interface axis port=strmm_in
#pragma HLS INTERFACE axis port=strm_out

  strmvin_t tmpvi;
  strmmin_t tmpmi;
  strmout_t tmpo;

  static int k = 0;
  static acc_t acc[N_COLUMNS];
  static datami_t matrix[MATRIX_SIZE];

  if(k == 0) {
    for (int i = 0; i < N_COLUMNS; i++) {
      tmpmi = strmm_in.read();
      acc[i] = tmpmi.data; // bias
    }
  }

  loop_m:
  for (int i = 0; i < MATRIX_SIZE; i++) {
    tmpmi = strmm_in.read();
    matrix[i] = tmpmi.data;
  }

  loop_i:
  for (int i = 0; i < N_LINES; i++) {
    tmpvi = strmv_in.read();
    loop_j:
    for(int j = 0; j < N_COLUMNS; j++) {
      #pragma HLS unroll
      mul_t mult = tmpvi.data * matrix[i*N_COLUMNS + j];
      acc[j] += mult;
    }
  }

  if(k == (NUM_BLOCKS - 1)) {
      k = 0;
      tmpo.keep = 0xF;
      tmpo.strb = 0xF;
      for (int i = 0; i < N_COLUMNS; i++) {
        tmpo.data = (datao_t)acc[i];
        tmpo.last = (i == (N_COLUMNS - 1));
        strm_out.write(tmpo);
      }
  } else {
    k ++;
  }
}
