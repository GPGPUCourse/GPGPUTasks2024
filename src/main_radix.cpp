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

#define TILE_SIZE 16

const int benchmarkingIters = 10;
const int benchmarkingItersCPU = 1;
const unsigned int n = 32 * 1024 * 1024;

const unsigned int nbits = 4;
const unsigned int ndigits = 1 << nbits;

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
    std::vector<unsigned int> buffer_cpu(n, 0);

    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    const std::vector<unsigned int> cpu_reference = computeCPU(as);

    unsigned int work_group_size = 128;
    unsigned int work_groups = (n + work_group_size - 1) / work_group_size;

    unsigned int global_size = work_groups * work_group_size;
    unsigned int buf_size = work_groups * ndigits;

    gpu::gpu_mem_32u array_gpu;
    array_gpu.resizeN(n);
    array_gpu.writeN(as.data(), n);

    gpu::gpu_mem_32u res_gpu;
    res_gpu.resizeN(n);

    gpu::gpu_mem_32u counters_gpu;
    counters_gpu.resizeN(buf_size);

    gpu::gpu_mem_32u buffer_gpu;
    buffer_gpu.resizeN(buf_size);

    ocl::Kernel count(radix_kernel, radix_kernel_length, "count");
    count.compile(true);

    ocl::Kernel transpose(radix_kernel, radix_kernel_length, "matrix_transpose_local_good_banks");
    transpose.compile(true);

    ocl::Kernel prefix_sum_up(radix_kernel, radix_kernel_length, "prefix_sum_up");
    prefix_sum_up.compile(true);

    ocl::Kernel prefix_sum_down(radix_kernel, radix_kernel_length, "prefix_sum_down");
    prefix_sum_down.compile(true);

    ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");
    radix_sort.compile(true);

    {
        timer t;
        for (unsigned int iter = 0; iter < benchmarkingIters; ++iter) {
            t.restart();
            for (unsigned int shift = 0; shift < std::numeric_limits<unsigned int>::digits; shift += nbits) {
                count.exec(gpu::WorkSize(work_group_size, global_size), array_gpu, counters_gpu, shift, n);

                transpose.exec(gpu::WorkSize(TILE_SIZE, TILE_SIZE, ndigits, work_groups), counters_gpu, buffer_gpu, work_groups, ndigits);

                for (unsigned int offset = 1; offset <= log2(buf_size); ++offset) {
                    prefix_sum_up.exec(gpu::WorkSize(work_group_size, buf_size >> offset), buffer_gpu, 1 << offset, buf_size);
                }

                buffer_gpu.readN(buffer_cpu.data(), buf_size);
                buffer_cpu[buf_size - 1] = 0;
                buffer_gpu.writeN(buffer_cpu.data(), buf_size);
                
                for (unsigned int offset = log2(buf_size); offset > 0; --offset) {
                    prefix_sum_down.exec(gpu::WorkSize(work_group_size, buf_size >> offset), buffer_gpu, 1 << offset, buf_size);
                }

                radix_sort.exec(gpu::WorkSize(work_group_size, global_size), array_gpu, buffer_gpu, shift / nbits, work_groups, res_gpu);
                std::swap(array_gpu, res_gpu);
            }
            t.nextLap();
        }
        t.stop();

        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    array_gpu.readN(as.data(), n);

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_reference[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}
