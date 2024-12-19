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

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)
#define WGS 64
#define BTS 4

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

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);
    gpu::gpu_mem_32u bs_gpu;
    bs_gpu.resizeN(n);

    gpu::gpu_mem_32u counters;
    uint counters_size = (1 << BTS) * (n / WGS);
    counters.resizeN(counters_size);

    gpu::gpu_mem_32u pref;
    pref.resizeN(counters_size);

    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        ocl::Kernel count(radix_kernel, radix_kernel_length, "counters");
        count.compile();

        ocl::Kernel prefix(radix_kernel, radix_kernel_length, "prefix");
        prefix.compile();

        ocl::Kernel prefix_bin(radix_kernel, radix_kernel_length, "prefix_bin");
        prefix_bin.compile();

        ocl::Kernel copy(radix_kernel, radix_kernel_length, "copy_arrays");
        copy.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), as.size());
            t.restart();

            for (int shift = 0; shift < 32; shift += BTS) {
                // counter
                count.exec(gpu::WorkSize(WGS, n), as_gpu, counters, shift);
                copy.exec(gpu::WorkSize(WGS, n), counters, pref, counters_size);
                // pref sums
                for (unsigned int stride = 2; stride <= counters_size; stride *= 2) {
                    prefix_bin.exec(gpu::WorkSize(8, counters_size / stride), pref, stride, counters_size);
                }
                for (unsigned int stride = counters_size / 2; stride >= 2; stride /= 2) {
                    prefix.exec(gpu::WorkSize(4, (counters_size + stride - 1) / stride), pref, stride, counters_size);
                }
                // sort
                radix.exec(gpu::WorkSize(1, (n + WGS - 1) / WGS), as_gpu, bs_gpu, counters, pref, shift);

                std::swap(as_gpu, bs_gpu);
            }

            t.nextLap();
        }

        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    as_gpu.readN(as.data(), n);

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_reference[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}
