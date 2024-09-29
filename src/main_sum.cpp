#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"

#include "cl/sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

void benchKernelName(const std::string &kernel_name,
                     int benchmarking_iters,
                     unsigned int n,
                     const gpu::gpu_mem_32u &gpu_as,
                     unsigned int wg_size,
                     unsigned int gw_size,
                     unsigned int reference_sum);

int main(int argc, char **argv) {
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
        std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
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
        std::cout << "CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u gpu_as;
        gpu_as.resizeN(n);
        gpu_as.writeN(as.data(), n);

        unsigned int workGroupSize = 128;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

        benchKernelName("global_atomic_add", benchmarkingIters, n, gpu_as, workGroupSize, global_work_size,
                        reference_sum);
        benchKernelName("cycled", benchmarkingIters, n, gpu_as, workGroupSize, global_work_size / 64, reference_sum);
        benchKernelName("cycled_coalesced", benchmarkingIters, n, gpu_as, workGroupSize, global_work_size / 64,
                        reference_sum);
        benchKernelName("local_mem_with_main_thread", benchmarkingIters, n, gpu_as, workGroupSize, global_work_size,
                        reference_sum);
        benchKernelName("tree", benchmarkingIters, n, gpu_as, workGroupSize, global_work_size, reference_sum);
    }
}

void benchKernelName(const std::string &kernel_name,
                     int benchmarking_iters,
                     unsigned int n,
                     const gpu::gpu_mem_32u &gpu_as,
                     unsigned int wg_size,
                     unsigned int gw_size,
                     unsigned int reference_sum) {
    ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernel_name);
    timer t;

    gpu::gpu_mem_32u gpu_sum;
    gpu_sum.resizeN(1);

    for (int iter = 0; iter < benchmarking_iters; ++iter) {
        unsigned int sum = 0;
        gpu_sum.writeN(&sum, 1);
        kernel.exec(gpu::WorkSize(wg_size, gw_size),
                    gpu_as,
                    gpu_sum,
                    n);
        gpu_sum.readN(&sum, 1);
        EXPECT_THE_SAME(reference_sum, sum, "GPU global_atomic_add result should be consistent!");
        t.nextLap();
    }
    std::cout << "GPU " + kernel_name + ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "GPU " + kernel_name + ": " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
}
