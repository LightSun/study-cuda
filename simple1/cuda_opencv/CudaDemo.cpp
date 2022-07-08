#include "CudaDemo.h"

// cuda 单元从大到小 grid->blocks->threads
//cuda 通过<<< >>>符号来分配索引线程的方式，我知道的一共有15种索引方式。

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>

using namespace std;

extern "C" void addWithCuda(int *c, const int *a, const int *b, unsigned int size);
extern "C" void test_performance0();

void CudaDemo::testTotal()
{
    printf("----> start CudaDemo::testTotal()\n");
    const int n = 1000;

   int *a = new int[n];
   int *b = new int[n];
   int *c = new int[n];
   int *cc = new int[n];

   for (int i = 0; i < n; i++)
   {
       a[i] = rand() % 100;
       b[i] = rand() % 100;
       c[i] = b[i] - a[i];
   }

   addWithCuda(cc, a, b, n);

   FILE *fp = fopen("out.txt", "w");
   for (int i = 0; i < n; i++)
       fprintf(fp, "%d %d\n", c[i], cc[i]);
   fclose(fp);

   bool flag = true;
   for (int i = 0; i < n; i++)
   {
       if (c[i] != cc[i])
       {
           flag = false;
           break;
       }
   }

   if (flag == false)
       printf("no pass!\n");
   else
       printf("pass!\n");

   //cudaDeviceReset();

   delete[] a;
   delete[] b;
   delete[] c;
   delete[] cc;

   printf("----> end CudaDemo::testTotal()\n");
}

void CudaDemo::test_performance(){
/**
    GPU memory: 7.629395e+00 MB
    GPU time: 21.436000 ms
    CPU time: 2310.440000 ms
    Max error: 1.19208e-07 Average error: 1.14175e-09
*/
    test_performance0();
}

extern "C" void cuda_processOverlap(int* _ls,
                                    int* _columns, int colLen,
                                    int* _starts, int* _ends, int rc,
                                    int s1_start, int s1_end, int grid_block[4]);
void CudaDemo::test_findOverlap(){
    int rc = 10;
    int _starts[rc];
    int _ends[rc];
    for(int i = 0 ; i < rc ; i ++){
        _starts[i] = i;
        _ends[i] = i + 10;//0-10, 1-11, 2-12 .... 9-19
    }
    int colLen = 8;
    int _columns[colLen];
    for(int i = 0 ; i < colLen ; i ++){
        _columns[i] = colLen - i - 1;//7,6,5....0
    }
    int _ls[colLen];
    int s1_start = 3;
    int s1_end = 16;
    //
    int grid_block[4];
    memset(grid_block, 0, 4 * sizeof (int));
//    grid_block[0] = 2;
//    grid_block[1] = 2;
//    grid_block[2] = 2;
//    grid_block[3] = 0;
    cuda_processOverlap(_ls, _columns, colLen, _starts, _ends,
                        rc, s1_start, s1_end, grid_block);
    for(int i = 0 ; i < colLen ; i ++){
        printf("_ls[%d] = %d\n", i, _ls[i]);
        //0: 7-17 ,3-16 ---> 16 - 7 = 9
        //1: 6-16       ---> 16 - 6 = 10
        //2: 5-15       ---> 15-5 = 10
        //3: 4-14       ---> 14-4 = 10
        //4: 3-13       ---> 13-3 = 10
        //5: 2-12       ---> 12-3 = 9
        //6: 1-11       ---> 11-3 = 8
        //7: 0-10       ---> 10-3 = 7
    }
}
