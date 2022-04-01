
#include <stdlib.h>
#include <memory.h>

void* _aligned_malloc(int reqSize, int align_unit_size);

#define _aligned_free(p) free(p)


