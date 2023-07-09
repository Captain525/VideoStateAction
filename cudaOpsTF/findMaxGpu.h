#ifndef FINDMAXGPU_H
#define FINDMAXGPU_H

#include <unsupported/Eigen/CXX11/Tensor>
template <typename Device, typename T>
struct ExampleFunctor{
    void operator()( const Device& d, int size, const T* in, const T* out);
};

#if GOOGLE_CUDA
//partially specialize functor for GpuDevice. 
template<typename T>
struct ExampleFunctor<Eigen::GpuDevice, T>{
    void operator()(const Eigen::GpuDevice& d, int size, const T* in, T* out);
    };
#endif
#endif FINDMAXGPU_H