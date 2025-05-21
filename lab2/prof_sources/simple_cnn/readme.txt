CNN_architecture.png 
  Block diagram of CNN architecture (adapted from
  http://neuralnetworksanddeeplearning.com/chap6.html)

wb.bin
  Binary file with 16+(16*3*3*3)+10+(10*16*43*43) floating-point neural net weights

Vitis Application Source and Header files:
  simple_cnn.h
  simple_cnn.c
  image.h
  image.c
  gemm.h
  gemm.c          
  
Note: to use the math function exp() you must add the linker library "m" 
(in Vitis App_Properties->C/C++ Build Settings)
