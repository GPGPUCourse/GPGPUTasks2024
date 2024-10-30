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

constexpr unsigned int LOG_SPAN = 7;
constexpr unsigned int RADIX_COUNT = 4;
constexpr unsigned int GROUP_SIZE = 128;
constexpr unsigned int TILE_SIZE = 16;
constexpr int GROUP_SIZE_X = 16;
constexpr int GROUP_SIZE_Y = 16;

constexpr int LINES_PER_GROUP0 = TILE_SIZE / GROUP_SIZE_X;
constexpr int LINES_PER_GROUP1 = TILE_SIZE / GROUP_SIZE_Y;
constexpr int SPAN_SIZE = 1 << LOG_SPAN;

#define DEF(name) ("-D" #name "=" + std::to_string(name))
#define CHAIN_DEF(name) + " " + DEF(name) // NOLINT(*-macro-parentheses)

std::string defs() {
    return DEF(LOG_SPAN) CHAIN_DEF(RADIX_COUNT) CHAIN_DEF(GROUP_SIZE) CHAIN_DEF(TILE_SIZE) CHAIN_DEF(LINES_PER_GROUP0) CHAIN_DEF(LINES_PER_GROUP1);
}

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

constexpr int ilog2(int x, int acc = 0) {
    return x <= 1 ? acc : ilog2(x >> 1, acc + 1);
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
        timer t;
        ocl::Kernel radix_count(radix_kernel, radix_kernel_length, "radix_count", defs());
        ocl::Kernel matrix_transpose(radix_kernel, radix_kernel_length, "matrix_transpose", defs());
        ocl::Kernel prefix_sum(radix_kernel, radix_kernel_length, "prefix_sum", defs());
        ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort", defs());

        radix_count.compile();
        matrix_transpose.compile();
        prefix_sum.compile();
        radix_sort.compile();

        gpu::gpu_mem_32u as_gpu, bs_gpu, counters_gpu, prefix_counters_gpu;
        as_gpu.resizeN(n);
        bs_gpu.resizeN(n);

        const unsigned int span_count = (n + SPAN_SIZE - 1) >> LOG_SPAN;
        const unsigned int counters_size = span_count << RADIX_COUNT;
        counters_gpu.resizeN(counters_size);
        prefix_counters_gpu.resizeN(counters_size);

        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();
            for (int shift = 0; shift < std::numeric_limits<unsigned int>::digits; shift += RADIX_COUNT) {
                radix_count.exec(gpu::WorkSize(GROUP_SIZE, span_count * GROUP_SIZE), as_gpu, counters_gpu, n, shift);
                matrix_transpose.exec(
                        gpu::WorkSize(GROUP_SIZE_X, GROUP_SIZE_Y, 1 << RADIX_COUNT, span_count),
                        counters_gpu, prefix_counters_gpu, span_count, 1 << RADIX_COUNT
                );

                {
                    int lg = ilog2(counters_size - 1) + 1;
                    for (int logStride = 0; (1 << logStride) < counters_size; ++logStride) {
                        prefix_sum.exec(gpu::WorkSize(GROUP_SIZE, counters_size >> (logStride + 1)), prefix_counters_gpu, 0, 1 << logStride, counters_size);
                    }
                    for (int logStride = lg - 2; logStride >= 0; --logStride) {
                        prefix_sum.exec(gpu::WorkSize(GROUP_SIZE, (counters_size - (1 << logStride)) >> (logStride + 1)), prefix_counters_gpu, (1 << logStride), 1 << (logStride), counters_size);
                    }
                }

                radix_sort.exec(gpu::WorkSize(GROUP_SIZE, span_count * GROUP_SIZE), as_gpu, prefix_counters_gpu, bs_gpu, n, shift);
                std::swap(as_gpu, bs_gpu);
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
