#include "b.h"

__device__ int g[N];

__device__ void bar ()
{
  g[threadIdx.x]++;
}
