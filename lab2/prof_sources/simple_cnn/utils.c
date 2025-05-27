
#include "utils.h"

int float2fixed(float f, int scale) {
  return (int)(f * (float)(1 << scale) + 0.5F);
}

float fixed2float(int i, int scale) {
  return (float)i / (float)(1 << scale);
}
