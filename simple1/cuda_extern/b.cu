#include "b.h"

__device__ int g[N];

__device__ void bar ()
{
  g[threadIdx.x]++;
}
/**
mergedBwt =
3967106048, workingMemory =
3967105440, mergedOccSizeInWord =
176

*/
