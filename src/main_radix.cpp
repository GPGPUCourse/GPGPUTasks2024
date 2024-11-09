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

    {
        ocl::Kernel make_counters_kernel(radix_kernel, radix_kernel_length, "make_counters");
        ocl::Kernel prefix_sum_kernel(radix_kernel, radix_kernel_length, "work_efficient_prefix_sum");
        ocl::Kernel matrix_transpose_kernel(radix_kernel, radix_kernel_length, "matrix_transpose_local_good_banks");
        ocl::Kernel set_zeros_kernel(radix_kernel, radix_kernel_length, "set_zeros");
        ocl::Kernel radix_sort_kernel(radix_kernel, radix_kernel_length, "radix_sort");
        make_counters_kernel.compile();
        prefix_sum_kernel.compile();
        matrix_transpose_kernel.compile();
        set_zeros_kernel.compile();
        radix_sort_kernel.compile();

        constexpr unsigned int nbits = 4;
        constexpr unsigned int mainWorkGroupSize = 128;
        constexpr unsigned int mTWorkGroupSize = 16;
        constexpr unsigned int countersWidth = 1 << nbits;
        constexpr unsigned int countersHeight = (n + mainWorkGroupSize - 1) / mainWorkGroupSize;
        constexpr unsigned int countersSize = countersWidth * countersHeight; // 2^nbits * nWGs

        constexpr unsigned int workSizeX = (countersWidth + mTWorkGroupSize - 1) / mTWorkGroupSize * mTWorkGroupSize;
        constexpr unsigned int workSizeY = (countersHeight + mTWorkGroupSize - 1) / mTWorkGroupSize * mTWorkGroupSize;
        gpu::WorkSize mTWorkSize(mTWorkGroupSize, mTWorkGroupSize, workSizeX, workSizeY);
        gpu::WorkSize mainWorkSize(128, n);

        gpu::gpu_mem_32u counters;
        counters.resizeN(countersSize);
        gpu::gpu_mem_32u countersTransposed;
        countersTransposed.resizeN(countersSize);
        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);
        gpu::gpu_mem_32u bs_gpu;
        bs_gpu.resizeN(n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();
            // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            for (int pos = 0; pos < sizeof(unsigned int) * 8; pos += nbits) {
                // 0. set zeros
                set_zeros_kernel.exec(gpu::WorkSize(128, countersSize), counters);
                // 1. calculate counters matrix
                make_counters_kernel.exec(mainWorkSize, as_gpu, counters, pos, nbits);
                // 2. transpose counters matrix
                matrix_transpose_kernel.exec(mTWorkSize, counters, countersTransposed, countersWidth, countersHeight);
                // 3. counters prefix sum
                // очень удачно, что число счетчиков это степень двойки, а пришлось бы все переписывать...
                int step = 1;
                for (; step < countersSize; step *= 2) {
                    prefix_sum_kernel.exec(gpu::WorkSize(std::min(256u, countersSize / step / 2), countersSize / step / 2),
                        countersTransposed, step, countersSize, 0);
                }
                for (step /= 2; step > 0; step /= 2) {
                    prefix_sum_kernel.exec(gpu::WorkSize(std::min(256u, countersSize / step / 2), countersSize / step / 2),
                        countersTransposed, step, countersSize, 1);
                }
                // 4. radix sort iter (idx based on counters prefix sum and own wg idx)
                radix_sort_kernel.exec(mainWorkSize, as_gpu, bs_gpu, countersTransposed, pos, nbits);
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
