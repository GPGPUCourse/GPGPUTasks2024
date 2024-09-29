#include "libgpu/context.h"


#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libgpu/context.h>
#include "libgpu/shared_device_buffer.h"

#include "cl/sum_cl.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // TODO: implement on OpenCL
        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);
        gpu::gpu_mem_32u result_gpu;
        result_gpu.resizeN(1);

        unsigned int workGroupSize = 128;
        constexpr unsigned int TASK_SIZE = 96;

        {
            ocl::Kernel global_atomic_sum_kernel(sum_kernel, sum_kernel_length, "global_atomic_sum");
            global_atomic_sum_kernel.compile(false);
            unsigned int globalWorkSizeX = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            timer t;
            unsigned int gpu_result = 0;
            for (int i = 0; i < benchmarkingIters; ++i) {
                result_gpu.writeN(&gpu_result, 1);
                global_atomic_sum_kernel.exec(gpu::WorkSize(workGroupSize, globalWorkSizeX), as_gpu, result_gpu, n);
                t.nextLap();
            }
            std::cout << "GPU (global atomic):     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (global atomic):     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
            result_gpu.readN(&gpu_result, 1);
            EXPECT_THE_SAME(reference_sum, gpu_result, "CPU and GPU results should be consistent!");
        }

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "global_looped_sum");
            kernel.compile(false);
            unsigned int globalWorkSizeX = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            globalWorkSizeX = (globalWorkSizeX + TASK_SIZE - 1) / TASK_SIZE;
            timer t;
            unsigned int gpu_result = 0;
            for (int i = 0; i < benchmarkingIters; ++i) {
                result_gpu.writeN(&gpu_result, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, globalWorkSizeX), as_gpu, result_gpu, n);
                t.nextLap();
            }
            std::cout << "GPU (global looped):     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (global looped):     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
            result_gpu.readN(&gpu_result, 1);
            EXPECT_THE_SAME(reference_sum, gpu_result, "CPU and GPU results should be consistent!");
        }

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "global_looped_coalesced_sum");
            kernel.compile(false);
            unsigned int globalWorkSizeX = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            globalWorkSizeX = (globalWorkSizeX + TASK_SIZE - 1) / TASK_SIZE;
            timer t;
            unsigned int gpu_result = 0;
            for (int i = 0; i < benchmarkingIters; ++i) {
                result_gpu.writeN(&gpu_result, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, globalWorkSizeX), as_gpu, result_gpu, n);
                t.nextLap();
            }
            std::cout << "GPU (global coalesced looped):     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (global coalesced looped):     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
            result_gpu.readN(&gpu_result, 1);
            EXPECT_THE_SAME(reference_sum, gpu_result, "CPU and GPU results should be consistent!");
        }

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "local_mem_sum");
            kernel.compile(false);
            unsigned int globalWorkSizeX = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            timer t;
            unsigned int gpu_result = 0;
            for (int i = 0; i < benchmarkingIters; ++i) {
                result_gpu.writeN(&gpu_result, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, globalWorkSizeX), as_gpu, result_gpu, n);
                t.nextLap();
            }
            std::cout << "GPU (local mem):     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (local mem):     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
            result_gpu.readN(&gpu_result, 1);
            EXPECT_THE_SAME(reference_sum, gpu_result, "CPU and GPU results should be consistent!");
        }

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "tree_local_mem_sum");
            kernel.compile(false);
            unsigned int globalWorkSizeX = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            timer t;
            unsigned int gpu_result = 0;
            for (int i = 0; i < benchmarkingIters; ++i) {
                result_gpu.writeN(&gpu_result, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, globalWorkSizeX), as_gpu, result_gpu, n);
                t.nextLap();
            }
            std::cout << "GPU (tree local mem):     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (tree local mem):     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
            result_gpu.readN(&gpu_result, 1);
            EXPECT_THE_SAME(reference_sum, gpu_result, "CPU and GPU results should be consistent!");
        }
    }
}
