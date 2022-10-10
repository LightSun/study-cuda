

## compile
```
nvcc --gpu-architecture=sm_80 --device-c a.cu b.cu
nvcc --gpu-architecture=sm_80 -o a_test a.o b.o -lcudart -lcuda
```
