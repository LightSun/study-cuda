#include "common.h"

void* _aligned_malloc(int reqSize, int align_unit_size){

    int size = reqSize % align_unit_size != 0 ? ((reqSize / align_unit_size) + 1)
                                    * align_unit_size : reqSize;
    void* ptr = malloc(size);
    memset(ptr, 0, size);
    return ptr;
}
