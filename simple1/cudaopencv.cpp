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

    printf("time : %f", time);

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
    cout << "CPU Time : " << timeSecCpu * 1000 << " ms" << endl;

    Mat dstImageGpu = Mat::zeros(srcImage.size(), srcImage.type());
    const int64 startGpu = getTickCount();
    gpuMirrorImg(srcImage, dstImageGpu);
    const double timeSecGpu = (getTickCount() - startGpu) / getTickFrequency();
    cout << "GPU Time : " << timeSecGpu * 1000 << " ms" << endl;

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
