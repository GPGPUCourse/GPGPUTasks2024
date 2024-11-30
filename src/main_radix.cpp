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

    ocl::Kernel write_zeros(radix_kernel, radix_kernel_length, "write_zeros");
    ocl::Kernel count(radix_kernel, radix_kernel_length, "count");
    ocl::Kernel transpose_counters(radix_kernel, radix_kernel_length, "transpose_counters");
    ocl::Kernel prefix_sum_up(radix_kernel, radix_kernel_length, "prefix_sum_up");
    ocl::Kernel prefix_sum_down(radix_kernel, radix_kernel_length, "prefix_sum_down");
    ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");

    write_zeros.compile();
    count.compile();
    transpose_counters.compile();
    prefix_sum_up.compile();
    prefix_sum_down.compile();
    radix_sort.compile();

    constexpr unsigned int nbits = 4;
    constexpr unsigned int work_size = 128;
    constexpr unsigned int transpose_work_group_size = 16;
    constexpr unsigned int c_width = 1 << nbits;
    constexpr unsigned int c_height = (n + work_size - 1) / work_size;
    constexpr unsigned int c_size = c_width * c_height;

    gpu::gpu_mem_32u as_gpu, bs_gpu, counters, counters_tr;
    
    counters.resizeN(c_size);
    counters_tr.resizeN(c_size);
    as_gpu.resizeN(n);
    bs_gpu.resizeN(n);

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            as_gpu.writeN(as.data(), n);
            t.restart();

            for (int bit_shift = 0; bit_shift < 32; bit_shift += nbits) {
                write_zeros.exec(gpu::WorkSize(work_size, c_size), counters, n);
                count.exec(gpu::WorkSize(work_size, n), as_gpu, counters, bit_shift, nbits);
                transpose_counters.exec(gpu::WorkSize(16, 16, c_width, c_height), 
                                            counters, counters_tr, c_width, c_height);
                
                for (unsigned int j = 1; j < c_size; j *= 2) {
                    prefix_sum_up.exec(gpu::WorkSize(work_size, c_size / j / 2), counters_tr, j, c_size);
                }
                for (unsigned int j = c_size / 2; j > 0; j /= 2) {
                    prefix_sum_down.exec(gpu::WorkSize(work_size, c_size / j / 2), counters_tr, j, c_size);
                }

                radix_sort.exec(gpu::WorkSize(work_size, n), as_gpu, bs_gpu, counters_tr, 
                                    bit_shift, nbits, n);
                as_gpu.swap(bs_gpu);
            }
            t.nextLap();
        }
        t.stop();
        as_gpu.readN(as.data(), n);

        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_reference[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}
