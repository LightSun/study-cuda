#include "study_avx.h"

//raw
static const int length = 1024*8;
static float a[length];
#define _countof(_arr) length
static float computeAvg() {
  float sum = 0.0;
  for (uint32_t j = 0; j < _countof(a); ++j) {
    sum += a[j];
  }
  return sum / _countof(a);
}
//256
static float avxAverage () {
    //256/32 = 8
  __m256 sumx8 = _mm256_setzero_ps();
  for (uint32_t j = 0; j < _countof(a); j = j + 8) {
    sumx8 = _mm256_add_ps(sumx8, _mm256_loadu_ps(&(a[j])));
  }
  float sum;
  //compile error
//  sum = sumx8.m256_f32[0] + sumx8.m256_f32[1] +
//  sumx8.m256_f32[2] + sumx8.m256_f32[3] +
//  sumx8.m256_f32[4] + sumx8.m256_f32[5] +
//  sumx8.m256_f32[6] + sumx8.m256_f32[7];
  return sum / _countof(a);
}

//512.  512/32 = 16
#ifdef __AVX512F__
static float avx512ComputeAvg() {
  __m512 sumx16 = _mm512_setzero_ps();
  for (uint32_t j = 0; j < _countof(a); j = j + 16) {
    sumx16 = _mm512_add_ps(sumx16, _mm512_loadu_ps(&(a[j])));
  }
  float sum = _mm512_reduce_add_ps (sumx16);
  return sum / _countof(a);
}
#endif
/** https://www.zhihu.com/question/374968330
AVX512是SIMD指令，也就是单指令多数据，
而x86架构上最早的SIMD指令是128bit的SSE，
然后是256bit的AVX/AVX2，到现在512bit的AVX512
*/
void StudyAVX::test1(){
    assert(avx512ComputeAvg() == computeAvg());
}
