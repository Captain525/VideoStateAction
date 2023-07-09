#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

REGISTER_OP("FindMax").Input("hInitial: float32", "hEnd: float32", "gValues: float32", "videoLen: int").Output("finalIndices: int32")

class FindMaxOp : public OpKernel{
    public: 
        explicit FindMaxOp(OpKernelConstruction * context): OpKernel(context){}
        void Compute(OpKernelContext* context) override{
            const Tensor& hInitial = context->input(0);
            const Tensor& hEnd = context->input(1);
            const Tensor& gValues = context->input(2);
            auto input 
        }
}