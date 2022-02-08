#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

using namespace cv;
using namespace cv::cuda;

void swap_rb_caller(const PtrStepSz<uchar3>& src, PtrStep<uchar3> dst, cudaStream_t stream);
void swap_rb(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null())
{
    CV_Assert(src.type() == CV_8UC3);
    dst.create(src.size(), src.type()); // create if not allocated yet
    cudaStream_t s = StreamAccessor::getStream(stream);
    swap_rb_caller(src, dst, s);
}
