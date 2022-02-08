#include "cudaopencv.h"
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"

#define IMG_SRC "/home/heaven7/Pictures/1.jpeg"

using namespace cv;
using namespace cv::cuda;
using namespace std;
#define BYTE unsigned char

extern "C"
void cpuMirrorImg(const cv::Mat& src, cv::Mat& dst);

extern "C"
void gpuMirrorImg(const cv::Mat& src, cv::Mat& dst);

CudaOpencv::CudaOpencv()
{
//cv::cuda::GpuMat mat;
}

extern "C"
double cudaInverseImg(BYTE* pImgOut, BYTE* pImgIn, int nWidth, int nHeight, int nWidthStep, int nChannels);

void CudaOpencv::test_iplimage(){
    IplImage* img = cvLoadImage(IMG_SRC, CV_LOAD_IMAGE_GRAYSCALE);

    cvShowImage("原始图", img);

    BYTE* pImgIn = (BYTE* ) img->imageData;
    BYTE* pImgOut = (BYTE*)img->imageData;
    int nWidth = img->width;
    int nHeight = img->height;
    int nDepth = img->depth;
    int nWidthStep = img->widthStep;
    int nChannels = img->nChannels;

    double time = cudaInverseImg(pImgOut, pImgIn, nWidth, nHeight, nWidthStep, nChannels);

    printf("test_iplimage >> time : %f\n", time);

    IplImage* imgOut = cvCreateImageHeader(cvSize(nWidth, nHeight), nDepth, nChannels);
    cvSetData(imgOut, pImgOut, nWidthStep);
    cvShowImage("反相图", imgOut);

    cvWaitKey(0);
}

void CudaOpencv::test_mat(){
    Mat srcImage = imread(IMG_SRC);

    Mat dstImageCpu = srcImage.clone();
    const int64 startCpu = getTickCount();
    cpuMirrorImg(srcImage, dstImageCpu);
    const double timeSecCpu = (getTickCount() - startCpu) / getTickFrequency();
    cout << "test_mat >> CPU Time : " << timeSecCpu * 1000 << " ms" << endl;

    Mat dstImageGpu = Mat::zeros(srcImage.size(), srcImage.type());
    const int64 startGpu = getTickCount();
    gpuMirrorImg(srcImage, dstImageGpu);
    const double timeSecGpu = (getTickCount() - startGpu) / getTickFrequency();
    cout << "test_mat >> GPU Time : " << timeSecGpu * 1000 << " ms" << endl;

    imshow("source", srcImage);
    imshow("mirror [CPU]", dstImageCpu);
    imshow("mirror [GPU]", dstImageGpu);

    waitKey(0);
}

void swap_rb(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null());

void CudaOpencv::test_gpu_mat(){
    Mat srcImage = imread(IMG_SRC);
    Mat dstImage = Mat::zeros(srcImage.size(), srcImage.type());

    GpuMat srcImageGpu(srcImage);
    GpuMat dstImageGpu;
    dstImageGpu.create(srcImageGpu.size(), srcImageGpu.type());
    swap_rb(srcImageGpu, dstImageGpu);
    dstImageGpu.download(dstImage);

    imshow("source image", srcImage);
    imshow("gpu image", dstImage);
    waitKey(0);
}

#include <cmath>
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"
static void help()
{
    cout << "This program demonstrates line finding with the Hough transform." << endl;
    cout << "Usage:" << endl;
    cout << "./gpu-example-houghlines <image_name>, Default is ../data/pic1.png\n" << endl;
}
void CudaOpencv::test_cu_opencv(){
    Mat src = imread(IMG_SRC, IMREAD_GRAYSCALE);
    if (src.empty())
    {
        help();
        cout << "can not open " << IMG_SRC << endl;
        return;
    }
    Mat mask;
    cv::Canny(src, mask, 100, 200, 3);
    Mat dst_cpu;
    cv::cvtColor(mask, dst_cpu, COLOR_GRAY2BGR);
    Mat dst_gpu = dst_cpu.clone();
    vector<Vec4i> lines_cpu;
    {
        const int64 start = getTickCount();
        cv::HoughLinesP(mask, lines_cpu, 1, CV_PI / 180, 50, 60, 5);
        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << "test_cu_opencv >> CPU Time : " << timeSec * 1000 << " ms" << endl;
        cout << "test_cu_opencv >> CPU Found : " << lines_cpu.size() << endl;
    }
    for (size_t i = 0; i < lines_cpu.size(); ++i)
    {
        Vec4i l = lines_cpu[i];
        line(dst_cpu, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
    }
    GpuMat d_src(mask);
    GpuMat d_lines;
    {
        const int64 start = getTickCount();
        Ptr<cuda::HoughSegmentDetector> hough = cuda::createHoughSegmentDetector(1.0f, (float)(CV_PI / 180.0f), 50, 5);
        hough->detect(d_src, d_lines);
        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << "GPU Time : " << timeSec * 1000 << " ms" << endl;
        cout << "GPU Found : " << d_lines.cols << endl;
    }
    vector<Vec4i> lines_gpu;
    if (!d_lines.empty())
    {
        lines_gpu.resize(d_lines.cols);
        Mat h_lines(1, d_lines.cols, CV_32SC4, &lines_gpu[0]);
        d_lines.download(h_lines);
    }
    for (size_t i = 0; i < lines_gpu.size(); ++i)
    {
        Vec4i l = lines_gpu[i];
        line(dst_gpu, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
    }
    imshow("source", src);
    imshow("detected lines [CPU]", dst_cpu);
    imshow("detected lines [GPU]", dst_gpu);
    waitKey();
}
