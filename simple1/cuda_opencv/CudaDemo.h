#ifndef CUDADEMO_H
#define CUDADEMO_H

/**
thread：一个CUDA的并行程序会被以许多个threads来执行。
block：数个threads会被群组成一个block，同一个block中的threads可以同步，也可以通过shared memory通信。
grid：多个blocks则会再构成grid。
 */
class CudaDemo
{
public:
    static void testTotal();
    //openmp vs cuda
    static void test_performance();

    static void test_findOverlap();
};

#endif // CUDADEMO_H
