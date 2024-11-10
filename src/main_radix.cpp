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

    unsigned int bits = 4;
    unsigned int num_values = (1 << bits);
    unsigned int work_group_size = 128;
    unsigned int tile_size = 16;

    gpu::gpu_mem_32u as_gpu, count_gpu, partial_sums_gpu, prefix_sums_gpu, out_gpu;
    as_gpu.resizeN(n);
    count_gpu.resizeN(n / work_group_size * num_values);
    partial_sums_gpu.resizeN(n / work_group_size * num_values);
    prefix_sums_gpu.resizeN(n / work_group_size * num_values + 1);
    out_gpu.resizeN(n);

    ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");
    ocl::Kernel count(radix_kernel, radix_kernel_length, "count");
    ocl::Kernel calculate_partial_sums(radix_kernel, radix_kernel_length, "calculate_partial_sums");
    ocl::Kernel calculate_prefix_sums(radix_kernel, radix_kernel_length, "calculate_prefix_sums");
    ocl::Kernel transpose(radix_kernel, radix_kernel_length, "transpose");
    ocl::Kernel zero_memory(radix_kernel, radix_kernel_length, "zero_memory");

    radix_sort.compile();
    count.compile();
    calculate_partial_sums.compile();
    calculate_prefix_sums.compile();
    transpose.compile();
    zero_memory.compile();

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();
            for (unsigned int bit_group = 0; bit_group * bits < 32; bit_group++) {
                count.exec(gpu::WorkSize(work_group_size, n), as_gpu, count_gpu, bit_group);

                transpose.exec(
                    gpu::WorkSize(
                        tile_size, tile_size,
                        (num_values + tile_size - 1) / tile_size *
                        tile_size,
                        (n / work_group_size + tile_size - 1) / tile_size * tile_size
                    ),
                    count_gpu, partial_sums_gpu, num_values, n / work_group_size
                );

                unsigned int out_n = n / work_group_size * num_values;

                for (unsigned int size = 1; size <= out_n; size <<= 1) {
                    if (size > 1) {
                        calculate_partial_sums.exec(gpu::WorkSize(std::min(work_group_size, out_n / size), out_n / size), partial_sums_gpu, size);
                    }
                    calculate_prefix_sums.exec(gpu::WorkSize(work_group_size, out_n), prefix_sums_gpu, partial_sums_gpu, size);
                }

                radix_sort.exec(gpu::WorkSize(work_group_size, n), as_gpu, prefix_sums_gpu, out_gpu, bit_group, n / work_group_size);

                zero_memory.exec(gpu::WorkSize(work_group_size, out_n), prefix_sums_gpu);
                std::swap(as_gpu, out_gpu);
            }

            t.nextLap();
        }
        t.stop();

        as_gpu.readN(as.data(), n);

        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

        // Проверяем корректность результатов
        for (int i = 0; i < n; ++i) {
            EXPECT_THE_SAME(as[i], cpu_reference[i], "GPU results should be equal to CPU results!");
        }
    }

   return 0;
}

