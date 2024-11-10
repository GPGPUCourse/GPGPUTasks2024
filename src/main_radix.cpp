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

std::vector<unsigned int> computeCPU(const std::vector<unsigned int> &as) {
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
        ocl::Kernel radix_numbers_kernel(radix_kernel, radix_kernel_length, "radix_numbers");
        ocl::Kernel prefix_sum_up_kernel(radix_kernel, radix_kernel_length, "prefix_sum_up");
        ocl::Kernel prefix_sum_down_kernel(radix_kernel, radix_kernel_length, "prefix_sum_down");
        ocl::Kernel matrix_transpose_kernel(radix_kernel, radix_kernel_length, "matrix_transpose_local_good_banks");
        ocl::Kernel clear_kernel(radix_kernel, radix_kernel_length, "clear");
        ocl::Kernel radix_sort_kernel(radix_kernel, radix_kernel_length, "radix_sort");

        radix_numbers_kernel.compile();
        prefix_sum_up_kernel.compile();
        prefix_sum_down_kernel.compile();
        matrix_transpose_kernel.compile();
        clear_kernel.compile();
        radix_sort_kernel.compile();

        constexpr unsigned int nbits = 4;
        constexpr unsigned int work_group_size = 128;
        constexpr unsigned int transpose_work_group_size = 16;
        constexpr unsigned int c_width = 1 << nbits;
        constexpr unsigned int c_height = (n + work_group_size - 1) / work_group_size;
        constexpr unsigned int c_size = c_width * c_height;

        gpu::WorkSize matrix_transpose_work_size(
                transpose_work_group_size, transpose_work_group_size,
                (c_width + transpose_work_group_size - 1) / transpose_work_group_size * transpose_work_group_size,
                (c_height + transpose_work_group_size - 1) / transpose_work_group_size * transpose_work_group_size);
        gpu::WorkSize work_size(128, n);

        gpu::gpu_mem_32u as_gpu;
        gpu::gpu_mem_32u bs_gpu;
        gpu::gpu_mem_32u counters;
        gpu::gpu_mem_32u counters_transposed;

        counters.resizeN(c_size);
        counters_transposed.resizeN(c_size);
        as_gpu.resizeN(n);
        bs_gpu.resizeN(n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();
            for (int i = 0; i < 32; i += nbits) {
                clear_kernel.exec(gpu::WorkSize(128, c_size), counters);
                radix_numbers_kernel.exec(work_size, as_gpu, counters, i, nbits);
                matrix_transpose_kernel.exec(matrix_transpose_work_size, counters, counters_transposed, c_width,
                                             c_height);
                int j = 1;
                for (; j < c_size; j *= 2) {
                    prefix_sum_down_kernel.exec(gpu::WorkSize(std::min(256, int(c_size / 2 / j)), c_size / 2 / j),
                                                counters_transposed, j, c_size);
                }
                for (j /= 2; j > 0; j /= 2) {
                    prefix_sum_up_kernel.exec(gpu::WorkSize(std::min(256, int(c_size / 2 / j)), c_size / 2 / j),
                                              counters_transposed, j, c_size);
                }
                radix_sort_kernel.exec(work_size, as_gpu, bs_gpu, counters_transposed, i, nbits);
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
