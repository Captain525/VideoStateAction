#include findMaxGpu.h
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
using CPUDevice = Eigen:::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
REGISTER_OP("FindMax").Input("hInitial: float32", "hEnd: float32", "gValues: float32", "videoLen: int").Output("finalIndices: int32")
//Cpu specialization of computation 
template <typename T>
struct FindMaxOpFunctor<CPUDevice,T>{
    void operator()(const CPUDevice& d, int size, const T* hInitial, const T* hEnd, const T* gValues, T* out){
        //do op here. 
    }
};

//op kernel definition
template <typename Device, typename T>
class FindMaxOp : public OpKernel{
    public: 
        explicit FindMaxOp(OpKernelConstruction * context): OpKernel(context){}
        void Compute(OpKernelContext* context) override{
            const Tensor& hInitial = context->input(0);
            const Tensor& hEnd = context->input(1);
            const Tensor& gValues = context->input(2);
            //create output tensor
            Tensor* output_tensor = NULL;
            //macros to check and handle exceptions and stuff. 
            OP_REQUIRES_OK(context, context->allocate_output(0, (3,), &output_tensor));

            //Do computation
            OP_REQUIRES(context, (hInitial.NumElements()== hEnd.NumElements())&&(hInitial.NumElements() == gValues.NumElements()), errors::InvalidArgument("Tensors have differing number of elements"))
            FindMaxOpFunctor<Device, T>()(
                context->eigen_device<Device>(), 
                static_cast<int>(hInitial.NumElements()),hInitial.flat<T>().data(), hEnd.flat<T>().data(), gValues.flat<T>().data(),output_tensor->flat<int32>().data());
                
            
        }
};


// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("FindMaxOp").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      FindMaxOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template class FindMaxOpFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("FindMaxOp").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      FindMaxOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA