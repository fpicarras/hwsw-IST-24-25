HLS folder:
    axil_macc.cpp - C-specification of the multiply-accumulate HW IP
    tb_axil_macc.cpp - C-testbench of the multiply-accumulate HW IPA.
                       Implements a vector product, with the vector elements randomly generated.
  
SW-only folder:
    dotprod_v0.c - sw program with simple dot product example
                   (to execute on the ARM processor using Vitis)

SW_HW folder: 
    dotprod_HWSW.c  - sw program for dot product example with the multiply-accumulate operations 
	                  executed in the HLS-synthesized HW IP and using direct pointers to access 
					  the I/O registers of the HW IP 
					  (to execute on the ARM processor using Vitis)
