#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <functional>

#include "cl/sum_cl.h"

#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"

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

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    const unsigned int ZERO = 0;
    const unsigned int workGroupSize = 128;
    const unsigned int values_per_workitem = 64;

    std::cout << "GPU: " << std::endl;

    gpu::gpu_mem_32u as_gpu, tmp1_gpu, tmp2_gpu;
    gpu::gpu_mem_32u result_gpu;
    as_gpu.resizeN(n);
    tmp1_gpu.resizeN(n);
    tmp2_gpu.resizeN(n);
    result_gpu.resizeN(1);
    as_gpu.writeN(as.data(), n);

    std::vector<std::pair<const char *, std::function<void(ocl::Kernel &kernel)>>> kernels = {
            {"sum01_global_atomic", [&](ocl::Kernel &kernel) {
                kernel.exec(gpu::WorkSize(workGroupSize, n), as_gpu, n, result_gpu);
            }},
            {"sum02_multiple_per_warp_non_coalesced", [&](ocl::Kernel &kernel) {
                kernel.exec(gpu::WorkSize(workGroupSize, (n + values_per_workitem - 1) / values_per_workitem), as_gpu, n, result_gpu);
            }},
            {"sum03_multiple_per_warp_coalesced", [&](ocl::Kernel &kernel) {
                const int k = 128;
                // why????
                // kernel.exec(gpu::WorkSize(workGroupSize, ((n + k - 1) / k + 127) / 128 * 128), as_gpu, n, result_gpu);
                kernel.exec(gpu::WorkSize(workGroupSize, (n + k - 1) / k), as_gpu, n, result_gpu);
            }},
            {"sum04_local_memory_tree", [&](ocl::Kernel &kernel) {
                kernel.exec(gpu::WorkSize(workGroupSize, (n + 1) / 2), as_gpu, n, result_gpu);
            }},
            {"sum05_global_tree", [&](ocl::Kernel &kernel) {
                int size = n;
                const int align = 128;
                gpu::gpu_mem_32u *a = &tmp1_gpu;
                gpu::gpu_mem_32u *b = &tmp2_gpu;
                as_gpu.copyToN(*a, size);
                while (size > 1024) {
                    int half_size = ((size + 1) / 2 + align - 1) / align * align;
                    kernel.exec(
                            gpu::WorkSize(workGroupSize, half_size),
                            *a, *b,
                            size, half_size
                    );
                    size = half_size;
                    std::swap(a, b);
                }
                unsigned int values[1024];
                a->readN(values, size);
                unsigned int sum = 0;
                for (int i = 0; i < size; ++i) {
                    sum += values[i];
                }
                result_gpu.writeN(&sum, 1);
            }},
            {"sum06_wider_global_tree", [&](ocl::Kernel &kernel) {
                int size = n;
                gpu::gpu_mem_32u *a = &tmp1_gpu;
                gpu::gpu_mem_32u *b = &tmp2_gpu;
                as_gpu.copyToN(*a, size);
                while (size > 1024) {
                    int part_size = (size + values_per_workitem - 1) / values_per_workitem;
                    kernel.exec(
                            gpu::WorkSize(workGroupSize, part_size),
                            *a, *b,
                            size, part_size
                    );
                    size = part_size;
                    std::swap(a, b);
                }
                unsigned int values[1024];
                a->readN(values, size);
                unsigned int sum = 0;
                for (int i = 0; i < size; ++i) {
                    sum += values[i];
                }
                result_gpu.writeN(&sum, 1);
            }}
    };

    for (const auto &kernel_info : kernels) {
        const char *kernel_name = kernel_info.first;
        auto exec = kernel_info.second;

        ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernel_name);
        kernel.compile();

        {
            timer t;

            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                result_gpu.writeN(&ZERO, 1);

                exec(kernel);

                unsigned int result;
                result_gpu.readN(&result, 1);
                EXPECT_THE_SAME(reference_sum, result, "Result should be equal to CPU result!");
                t.nextLap();
            }

            std::cout << " - " << kernel_name << ":" << std::endl;
            std::cout << "            time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "      throughput: " << (n / 1e9) / t.lapAvg() << " B/s" << std::endl;
        }
    }
}
