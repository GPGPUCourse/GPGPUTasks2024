#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

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

void run(const std::vector<unsigned int>& as, unsigned int referenceSum, int benchmarkingIters, gpu::Device device, ocl::Kernel kernel, std::string kernelName, int workGroupSize) {
    unsigned int n = as.size();
    unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);
    as_gpu.writeN(as.data(), n);

    kernel.compile(false);
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;

            gpu::gpu_mem_32u sum_gpu;
            sum_gpu.resizeN(1);
            sum_gpu.writeN(&sum, 1);
            kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, sum_gpu, n);
            sum_gpu.readN(&sum, 1);

            EXPECT_THE_SAME(referenceSum, sum, "GPU " + kernelName + " result should be consistent!");

            t.nextLap();
        }
        std::cout << "GPU " + kernelName + ":     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU " + kernelName + ":     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
}

int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100 * 1000 * 1000;
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
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        ocl::Kernel globalAtomic(sum_kernel, sum_kernel_length, "sum_gpu_1");
        run(as, reference_sum, benchmarkingIters, device, globalAtomic, "globalAtomic", 256);

        ocl::Kernel loopSum(sum_kernel, sum_kernel_length, "sum_gpu_2");
        run(as, reference_sum, benchmarkingIters, device, loopSum, "loopSum", 32);

        ocl::Kernel loopSumCoalesced(sum_kernel, sum_kernel_length, "sum_gpu_3");
        run(as, reference_sum, benchmarkingIters, device, loopSumCoalesced, "loopSumCoalesced", 32);

        ocl::Kernel localMemorySum(sum_kernel, sum_kernel_length, "sum_gpu_4");
        run(as, reference_sum, benchmarkingIters, device, localMemorySum, "localMemorySum", 256);

        ocl::Kernel treeSum(sum_kernel, sum_kernel_length, "sum_gpu_5");
        run(as, reference_sum, benchmarkingIters, device, treeSum, "treeSum", 256);
    }   
}
