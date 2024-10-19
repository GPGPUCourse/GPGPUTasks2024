#ifndef SUM_KERNEL_RUNNER_H
#define SUM_KERNEL_RUNNER_H

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"
#include "utils.h"

class SumKernelRunner {
public:
    SumKernelRunner(gpu::Device device, gpu::Context context, std::string name, unsigned work_size, unsigned arr_size, unsigned wg_size)
        : device_(std::move(device))
        , context_(std::move(context))
        , arr_size_(arr_size)
        , array_(arr_size)
        , kernel_(sum_kernel, sum_kernel_length, std::move(name))
        , wg_size_(wg_size)
        , global_work_size_((work_size + wg_size - 1) / wg_size * wg_size)
    {
        FastRandom r(42);
        for (int i = 0; i < arr_size_; ++i) {
            array_[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / arr_size_);
            expected_ += array_[i];
        }

        array_gpu_.resizeN(arr_size_);
        array_gpu_.writeN(array_.data(), arr_size_);
        result_gpu_.resizeN(1);
        unsigned null{0};
        result_gpu_.writeN(&null, 1);
    };

    void operator()() {
        unsigned null{0};
        result_gpu_.writeN(&null, 1);
        kernel_.exec(gpu::WorkSize(wg_size_, global_work_size_),
                     array_gpu_, result_gpu_, arr_size_);

        unsigned res;
        result_gpu_.readN(&res, 1);
        EXPECT_THE_SAME(res, expected_, "Invalid sum for kernel");
    }

private:
    gpu::Device device_;
    gpu::Context context_;
    unsigned arr_size_;
    unsigned wg_size_;
    unsigned global_work_size_;
    std::vector<unsigned int> array_;
    uint32_t expected_{};
    gpu::gpu_mem_32u array_gpu_;
    gpu::gpu_mem_32u result_gpu_;
    ocl::Kernel kernel_;
};

#endif // SUM_KERNEL_RUNNER_H
