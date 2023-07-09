#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include findMaxGpu.h
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

//Define cuda kernel
template <typename T>//maybe change to int*
__global__ void FindMaxCudaKernel(const int size, const float32* hInitial, const float32* hEnd, const float32* gValues, float32* out){
//do calculations in here. 

}

void FindMaxFunctor<GPUDevice, T>::operator()(const GPUDevice& d, int size, const float32* hInitial,const float32* hInitial, const float32* hEnd, const float32* gValues, int32* out){
    //launch cuda kernel. 
    //look at core/util/gpu_kernel_helper.h for computing these block and thread per block counts. 
    int block_count = 1024;
    int thread_per_block = 20;
    FindMaxCudaKernel<T><<<block_count, thread_per_block, 0, d.stream()>>>(size, hInitial, hEnd, gValues, out);
}
template struct FindMaxFunctor<GPUDevice, float>;


#endif // GOOGLE_CUDA
