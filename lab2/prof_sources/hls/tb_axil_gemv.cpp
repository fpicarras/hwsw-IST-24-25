#include <stdio.h>
#include <stdlib.h>

#include "axil_gemv.h"

static double mA[MATRIX_SIZE];
static double bA[N_COLUMNS];
static double vB[N_LINES];
static double hw_vC[N_COLUMNS];
static double sw_vC[N_COLUMNS];

void init_bA_vC() {
  for(int i = 0; i < N_COLUMNS; i++) {
    bA[i] = (((double)rand()/RAND_MAX) - 0.5)/0.5; // Q1.15 (-1, 1)
    sw_vC[i] = bA[i];
  }
}

void init_mA_vB() {
  for (int i = 0; i < MATRIX_SIZE; i++) {
    mA[i] = (((double)rand()/RAND_MAX) - 0.5)/0.5; // Q1.15 (-1, 1)
  }
  for (int i = 0; i < N_LINES; i++) {
    vB[i] = 32*(((double)rand()/RAND_MAX) - 0.5)/0.5; // Q6.26 (-32, 32) 
  }
}

void print_mat(double *x, int rows, int cols) {
  int i;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%3.4f ", x[i*cols + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void sw_gemv() {
  for (int i = 0; i < N_LINES; i++) {
    for (int k = 0; k < N_COLUMNS; k++) {
      sw_vC[k] += mA[i*N_COLUMNS+k]*vB[i];
    }
  }
}

int check_output() {
  int err_cnt = 0;
  double err_max = 0.0;
  for (int i = 0; i < N_COLUMNS; i++) {
    double err = fabs(hw_vC[i] - sw_vC[i]);
    double maximum_tolerable_error = (double)1.0e-01;
    if (err > maximum_tolerable_error) err_cnt++;
    if (err > err_max) err_max = err;
  }
  printf("max_err = %.2e\n", err_max);
  return err_cnt;
}

int main() {
  hls::stream<strmvin_t> strv_in;
  hls::stream<strmmin_t> strm_in;
  hls::stream<strmout_t> str_out;
  strmvin_t tmpvi;
  strmmin_t tmpmi;
  strmout_t tmpo;

  init_bA_vC();
  // Send bias
  for (int i = 0; i < N_COLUMNS; i++) {
    tmpmi.data = (datami_t)bA[i];
    tmpmi.last = (ap_int<1>)0;
    strm_in.write(tmpmi);
  }

  for(int k = 0; k < NUM_BLOCKS; k++) {
    init_mA_vB();
    // Write vector B to str_in
    for (int i = 0; i < N_LINES; i++) {
      tmpvi.data = (datavi_t)vB[i];
      tmpvi.last = (ap_int<1>)(i==(N_LINES-1));
      strv_in.write(tmpvi);
    }

    // Write (append) matrix A to str_in
    for (int i = 0; i < MATRIX_SIZE; i++) {
      tmpmi.data = (datami_t)mA[i];
      tmpmi.last = (ap_int<1>)(i==(MATRIX_SIZE-1));
      strm_in.write(tmpmi);
    }

    sw_gemv();
    // Simulates one execution of gemv IP
    axil_gemv(strv_in, strm_in, str_out);
  }

  // Read vector C from str_out
  for (int i = 0; i < N_COLUMNS; i++) {
    tmpo = str_out.read();
    hw_vC[i] = (double)tmpo.data;
  }

  // Compare results with software implementation
  return check_output();
}
