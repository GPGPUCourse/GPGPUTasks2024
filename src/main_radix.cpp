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
        ocl::Kernel write_zeros(radix_kernel, radix_kernel_length, "write_zeros");
        write_zeros.compile();

        ocl::Kernel radix_count(radix_kernel, radix_kernel_length, "radix_count");
        radix_count.compile();

        ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");
        radix_sort.compile();

        ocl::Kernel matrix_transpose(radix_kernel, radix_kernel_length, "matrix_transpose");
        matrix_transpose.compile();

        ocl::Kernel prefix_sum_up(radix_kernel, radix_kernel_length, "prefix_sum_up");
        prefix_sum_up.compile();

        ocl::Kernel prefix_sum_down(radix_kernel, radix_kernel_length, "prefix_sum_down");
        prefix_sum_down.compile();

        const unsigned int workSize = n;
        const unsigned int workGroupSize = 128;
        const unsigned int nbits = 4;
        const unsigned int countersX = (1 << nbits);
        const unsigned int countersY = (n + workGroupSize - 1) / workGroupSize;
        const unsigned int countersSize = countersX * countersY;

        gpu::gpu_mem_32u as_gpu, bs_gpu, counters_gpu, counters_transposed_gpu;
        as_gpu.resizeN(n);
        bs_gpu.resizeN(n);
        counters_gpu.resizeN(countersSize);
        counters_transposed_gpu.resizeN(countersSize);
        
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            t.restart();

            for (unsigned int bit_shift = 0; bit_shift < 32; bit_shift += nbits) {
                write_zeros.exec(gpu::WorkSize(workGroupSize, countersSize), counters_gpu);

                radix_count.exec(gpu::WorkSize(workGroupSize, n), as_gpu, counters_gpu, bit_shift);


                matrix_transpose.exec(gpu::WorkSize(16, 16, countersX, countersY), counters_gpu,
                                      counters_transposed_gpu, countersX, countersY);

                {
                    unsigned int step = 1;
                    for (; step < countersSize; step *= 2) {
                        const unsigned int workSize = countersSize / (step * 2);
                        prefix_sum_up.exec(gpu::WorkSize(256, workSize), counters_transposed_gpu, countersSize, step,
                                           workSize);
                    }

                    for (step /= 2; step > 1; step /= 2) {
                        const unsigned int workSize = countersSize / step;
                        prefix_sum_down.exec(gpu::WorkSize(256, workSize), counters_transposed_gpu, countersSize, step,
                                             workSize);
                    }
                }

                radix_sort.exec(gpu::WorkSize(workGroupSize, n), as_gpu, bs_gpu, counters_transposed_gpu, bit_shift);

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
