#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>

const int benchmarkingIters = 10;
const int benchmarkingItersCPU = 1;
const unsigned int n = 32 * 1024 * 1024;

const int bits_count = 4;

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

std::vector<unsigned int> computeCPU(const std::vector<unsigned int> &as)
{
    std::vector<unsigned int> cpu_sorted;

    timer t;
    for (int iter = 0; iter < benchmarkingItersCPU; ++iter) {
        cpu_sorted = as;
        t.restart();
        std::sort(cpu_sorted.begin(), cpu_sorted.end());
        t.nextLap();
    }
    std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

    return cpu_sorted;
}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    const std::vector<unsigned int> cpu_reference = computeCPU(as);
    
    unsigned int workGroupSize = 128;
    unsigned int globalWorkSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
    unsigned int countersSize = globalWorkSize / workGroupSize * bits_count;
    unsigned int countersWorkSize = (countersSize + workGroupSize - 1) / workGroupSize * workGroupSize;

    gpu::gpu_mem_32u as_gpu;
    gpu::gpu_mem_32u bs_gpu;
    gpu::gpu_mem_32u counters_gpu;
    gpu::gpu_mem_32u counters_gpu_tmp;

    as_gpu.resizeN(n);
    bs_gpu.resizeN(n);
    counters_gpu.resizeN(countersSize);
    counters_gpu_tmp.resizeN(countersSize);
    
    {
        ocl::Kernel fill_with_zeros(radix_kernel, radix_kernel_length, "fill_with_zeros");
        fill_with_zeros.compile();
        ocl::Kernel count(radix_kernel, radix_kernel_length, "count");
        count.compile();
        ocl::Kernel prefix_sum(radix_kernel, radix_kernel_length, "prefix_sum");
        prefix_sum.compile();
        ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");
        radix_sort.compile();
        
        
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();
            
            for (unsigned int shift = 0; shift < 32; shift += bits_count) {
                fill_with_zeros.exec(gpu::WorkSize(workGroupSize, countersWorkSize), counters_gpu, countersSize);
                count.exec(gpu::WorkSize(workGroupSize, globalWorkSize), as_gpu, counters_gpu, n, shift, bits_count);
                for (unsigned int i = 1; i < countersSize; i *= 2) {
                    prefix_sum.exec(gpu::WorkSize(workGroupSize, countersWorkSize), counters_gpu, counters_gpu_tmp, i, countersSize);
                    std::swap(counters_gpu, counters_gpu_tmp);
                }
                radix_sort.exec(gpu::WorkSize(workGroupSize, globalWorkSize), as_gpu, bs_gpu, counters_gpu, n, shift, bits_count);
                std::swap(as_gpu, bs_gpu);
            }
            
            t.nextLap();
        }
        t.stop();

        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_reference[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}
