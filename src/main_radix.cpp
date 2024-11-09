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
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    const std::vector<unsigned int> cpu_reference = computeCPU(as);

    unsigned int local_size = 128;
    unsigned int work_groups = (n + local_size - 1) / local_size;
    unsigned int global_size = work_groups * local_size;
    unsigned int buf_size = work_groups * ndigits;

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);
    as_gpu.writeN(as.data(), n);

    gpu::gpu_mem_32u bs_gpu;
    bs_gpu.resizeN(n);

    gpu::gpu_mem_32u cs_gpu;
    cs_gpu.resizeN(buf_size);

    gpu::gpu_mem_32u buf;
    buf.resizeN(buf_size);

    ocl::Kernel count_by_workgroup(radix_kernel, radix_kernel_length, "count_by_workgroup");
    ocl::Kernel transpose_counters(radix_kernel, radix_kernel_length, "transpose_counters");
    ocl::Kernel prefix_sum(radix_kernel, radix_kernel_length, "prefix_sum");
    ocl::Kernel prefix_sum_down(radix_kernel, radix_kernel_length, "prefix_sum_down");
    ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");
    ocl::Kernel set_zero(radix_kernel, radix_kernel_length, "set_zero");
    set_zero.compile(true);
    transpose_counters.compile(true);
    count_by_workgroup.compile(true);
    prefix_sum.compile(true);
    prefix_sum_down.compile(true);
    radix_sort.compile(true);

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            t.restart();
            for (int bit_shift = 0; bit_shift < 32; bit_shift += nbits) {
                count_by_workgroup.exec(gpu::WorkSize(local_size, global_size), as_gpu, cs_gpu, bit_shift, n);
                transpose_counters.exec(gpu::WorkSize(16, 16, ndigits, work_groups), cs_gpu, buf, work_groups, ndigits);

                for (int offset = 1; offset <= log2(buf_size); offset++) {
                    prefix_sum.exec(gpu::WorkSize(local_size, (buf_size) >> (offset)), buf, 1 << (offset), buf_size);
                }
                set_zero.exec(gpu::WorkSize(1, 1), buf, buf_size);
                for (int offset = log2(buf_size); offset > 0; offset--) {
                    prefix_sum_down.exec(gpu::WorkSize(local_size, (buf_size) >> (offset)), buf, 1 << (offset), buf_size);
                }

                radix_sort.exec(gpu::WorkSize(local_size, global_size), as_gpu, bs_gpu, buf, bit_shift / nbits, work_groups);
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
