#include <stdio.h>
#include "xparameters.h"
#include "xaxil_mat_prod1.h"
#include "xiltimer.h"

#define MAT_SIZE 25

#define N1 MAT_SIZE
#define N2 MAT_SIZE
#define N3 MAT_SIZE

#define MATA(I,J) (matA[(I)*N2+(J)])
#define MATB(I,J) (matB[(I)*N3+(J)])
#define MATCS(I,J) (matCS[(I)*N3+(J)])
#define MATCH(I,J) (matCH[(I)*N3+(J)])

#define MATA_START_ADDRESS 0x100000
#define MATB_START_ADDRESS (MATA_START_ADDRESS+4*N1*N2)
#define MATCS_START_ADDRESS (MATA_START_ADDRESS+4*N1*N2+4*N2*N3)
#define MATCH_START_ADDRESS (MATCS_START_ADDRESS+4*N1*N3)

volatile int *matA = (int *)(MATA_START_ADDRESS);   // matA N1xN2
volatile int *matB = (int *)(MATB_START_ADDRESS);   // matB N2xN3
volatile int *matCS = (int *)(MATCS_START_ADDRESS);   // matC N1xN3
volatile int *matCH = (int *)(MATCH_START_ADDRESS);   // matC N1xN3

void print_mat(int *x, int colsize, int rowsize)
{
  int i, j;

  for (i=0; i<colsize; i++) {
    for (j=0; j<rowsize; j++) {
      printf("%d ", x[i*rowsize+j]);
    }
    printf("\n");
  }
  // printf("\n");
}

void SW_mat_prod()
{
  int i, j, k;

  for (i=0; i<N1; i++) {
    for (j=0; j<N3; j++) {
      MATCS(i,j) = 0;
      for (k=0; k<N2; k++) {
    	  MATCS(i,j) += MATA(i,k)*MATB(k,j);
      }
    }
  }
}

#define IP_BASEADDR 0x40000000
void HW_mat_prod()
{
	int i;
	// Explicitly define the addresses of the IP memory-mapped I/O registers
	volatile int *a = (int *)(IP_BASEADDR + 4096);
	volatile int *b = (int *)(IP_BASEADDR + 8192);
	volatile int *c = (int *)(IP_BASEADDR + 12288);
	volatile int *rowsA = (int *)(IP_BASEADDR + 0x10);
	volatile int *colsA = (int *)(IP_BASEADDR + 0x18);
	volatile int *colsB = (int *)(IP_BASEADDR + 0x20);
	volatile int *do_matp_mem = (int *)(IP_BASEADDR + 0x0);

	*rowsA = N1;
	*colsA = N2;
	*colsB = N3;
	for (i=0; i<(N1*N2); i++) {
		a[i] = matA[i];
	}
	for (i=0; i<(N2*N3); i++) {
		b[i] = matB[i];
	}
	*do_matp_mem = 1;
	while ((*do_matp_mem & 2) == 0);

	for (i=0; i<(N1*N3); i++) {
		matCH[i] = c[i];
	}
}

void  compare_mat(){
    for(int i = 0; i < N1; i++){
        for(int j = 0; j < N3; j++){
            if(MATCS(i,j) != MATCH(i,j)){
                printf("Different value at element (%d, %d), SW: %d, HW: %d\n", i, j,
                                        MATCS(i, j), MATCH(i, j));
            }
        }
    }
}

int main()
{
  XTime tStart, tEnd;

  print_mat((int *)matA,N1,N2);
  //print_mat((int *)matB,N2,N3);

  XTime_GetTime(&tStart); // start measuring time
  SW_mat_prod();
  XTime_GetTime(&tEnd);

  // print_mat((int *)matCS,N1,N3);
  printf("Execution took %llu clock cycles.\n", 2*(tEnd - tStart));
  printf("SW Execution took %.2f us.\n\n",
         1.0 * (tEnd - tStart) * 1000000/ (COUNTS_PER_SECOND));

  XTime_GetTime(&tStart); // start measuring time
  HW_mat_prod();
  XTime_GetTime(&tEnd);

  // print_mat((int *)matCH,N1,N3);
  printf("Execution took %llu clock cycles.\n", 2*(tEnd - tStart));
  printf("(%d) HW Execution took %.2f us.\n\n", N1,
         1.0 * (tEnd - tStart) * 1000000/ (COUNTS_PER_SECOND));
         
  compare_mat();
  printf("Finish!\n");
  return 0;
}
