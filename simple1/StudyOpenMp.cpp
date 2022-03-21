#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <time.h>
#include <sys/time.h>

using namespace std;
#include "StudyOpenMp.h"
#define DEFINE_idx auto idx = omp_get_thread_num();//线程id
#define _ROWS (omp_get_num_threads())

StudyOpenMp::StudyOpenMp()
{

}

//from: https://zhuanlan.zhihu.com/p/61857547
void StudyOpenMp::test1(){
//需要并行运算的才能这样. blow 32个线程
//reduction: 表示sum 会并发进行+运算。需要保护
    int sum = 0;
        #pragma omp parallel for num_threads(32) reduction(+:sum)
        for(int i=0; i<100; i++)
        {
            sum +=  i;
        }

        cout << sum << endl;
}

//原子操作与同步
void StudyOpenMp::test2(){
    int sum = 0;
    #pragma omp parallel  num_threads(3)
        {
    #pragma omp atomic
            sum += 10;

    #pragma omp barrier  // TODO : disable this to see
            cout << sum << endl;
        }
}

void StudyOpenMp::test3(){
    auto diff_in_seconds_jd = [](timeval *end, timeval *start)
        {
            double sec;
            sec=(end->tv_sec - start->tv_sec);
            sec+=(end->tv_usec - start->tv_usec)/1e6;
            return sec;
        };

        struct timeval start, end;

        vector<int> v_i{};

        int len = int(1e7);
        v_i.resize(len);
        vector<int> v_ans {};
        v_ans.resize(len);


        for(int i=0;i<len;i++)
        {
            v_i[i] = i;
        }

        int cnt_ans = 0;

        v_i[len - 1] = 2;

        gettimeofday (&start, NULL);  // time start

    #pragma omp parallel for reduction(+:cnt_ans) default(shared)  num_threads(10)
            for(int i=0; i<len; i++)
        {
            auto &e = v_i[i];
            int t = 0;

            if ( i < len / 2)
            {
                t = pow(e, 2);
            }
            else
            {
                t = (int)sqrt(e);
            }
            if ( t % 2 == 1)
            {
                cnt_ans += 1;
            }
        }

        gettimeofday (&end, NULL); // time end
        auto time_used = diff_in_seconds_jd(&end, &start);
        printf("- time used:\t%f seconds\n", time_used);

        cout << "- size of v_ans:\t" << cnt_ans << endl;


        for(auto &e : v_ans)
        {
            //cout << e << endl;
        }
}

#define NUM_THREADS 4
void StudyOpenMp::test4(){
    int num_steps = 28;
    int i;
    double x, pi, sum = 0.0; // 多个变量定义方式
    double step = 1.0/(double) num_steps;
    omp_set_num_threads(NUM_THREADS); // 设置使用的线程数


    const clock_t begin_time = clock(); // 统计一下使用的时间

    #pragma omp parallel for private(x) reduction(+:sum)
    for (i=0;i< num_steps; i++){
        printf("i: %d --- x: %f -- sum: %f---- Thread %d !!!\n",i, x, sum, omp_get_thread_num());
        x = (i+0.5)*step;
        sum = sum + 4.0/(1.0+x*x);
    }
    pi = step * sum;
    printf("x: %f --  pi: %f---- Thread %d !!!\n", x, pi, omp_get_thread_num());

    std::cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
}

double x;
void StudyOpenMp::test5(){
    int num_steps = 28;
    int i;
    double x, pi, sum = 0.0; // 多个变量定义方式
    double step = 1.0/(double) num_steps;
    omp_set_num_threads(NUM_THREADS); // 设置使用的线程数


    const clock_t begin_time = clock(); // 统计一下使用的时间

    #pragma omp parallel for /*private(x)*/ reduction(+:sum)
    for (i=0;i< num_steps; i++){
        printf("i: %d --- x: %f -- sum: %f---- Thread %d !!!\n",i, x, sum, omp_get_thread_num());
        x = (i+0.5)*step;
        sum = sum + 4.0/(1.0+x*x);
    }
    pi = step * sum;
    printf("x: %f --  pi: %f---- Thread %d !!!\n", x, pi, omp_get_thread_num());

    std::cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
}
