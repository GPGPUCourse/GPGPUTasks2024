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

    ocl::Kernel clear(radix_kernel, radix_kernel_length, "clear");
    clear.compile();

    ocl::Kernel count(radix_kernel, radix_kernel_length, "count");
    count.compile();

    ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");
    radix_sort.compile();

    ocl::Kernel matrix_transpose(radix_kernel, radix_kernel_length, "matrix_transpose");
    matrix_transpose.compile();

    ocl::Kernel prefix_sum_up_sweep(radix_kernel, radix_kernel_length, "up_sweep");
    prefix_sum_up_sweep.compile();
    ocl::Kernel prefix_sum_down_sweep(radix_kernel, radix_kernel_length, "down_sweep");
    prefix_sum_down_sweep.compile();


    unsigned int workgroup_size = 8;
    unsigned int workgroups = (n + workgroup_size - 1) / workgroup_size;
    unsigned int bits_for_sort = 4;
    unsigned int counters_number = 1 << bits_for_sort;
    unsigned int counters_n = workgroups * counters_number;

    gpu::gpu_mem_32u as_gpu, bs_gpu, counters_gpu, counters_t_gpu;
    as_gpu.resizeN(n);
    bs_gpu.resizeN(n);
    counters_gpu.resizeN(counters_n);
    counters_t_gpu.resizeN(counters_n);

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            t.restart();
            for (unsigned int shift = 0; shift < 32; shift += bits_for_sort) {
                clear.exec(gpu::WorkSize(workgroup_size, counters_n), counters_gpu, counters_n);
                count.exec(gpu::WorkSize(workgroup_size, n), as_gpu, counters_gpu, n, shift);
                matrix_transpose.exec(gpu::WorkSize(16, 16, counters_number, workgroups),
                                      counters_gpu, counters_t_gpu,counters_number, workgroups);
                unsigned int sum_len = 2;
                for (; sum_len <= n; sum_len *= 2)
                    if (counters_n / sum_len > 0)
                        prefix_sum_up_sweep.exec(gpu::WorkSize(workgroup_size, counters_n / sum_len), counters_t_gpu, sum_len,
                                                 counters_n);
                for (; sum_len > 1; sum_len /= 2)
                    if (counters_n / sum_len > 0)
                        prefix_sum_down_sweep.exec(gpu::WorkSize(workgroup_size, counters_n / sum_len), counters_t_gpu, sum_len,
                                                   counters_n);
                radix_sort.exec(gpu::WorkSize(workgroup_size, n), as_gpu, bs_gpu, counters_t_gpu, n, shift);
                std::swap(as_gpu, bs_gpu);
            }
            t.nextLap();
        }
        t.stop();

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
