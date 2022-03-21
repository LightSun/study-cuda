#include "CudaDemo.h"

// cuda 单元从大到小 grid->blocks->threads
//cuda 通过<<< >>>符号来分配索引线程的方式，我知道的一共有15种索引方式。

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

extern "C" void addWithCuda(int *c, const int *a, const int *b, unsigned int size);

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
