#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/merge_cl.h"

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

std::vector<int> computeCPU(const std::vector<int> &as)
{
    std::vector<int> cpu_sorted;

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

    std::vector<int> as(n);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.next();
    }
    std::vector<int> initial_as(as.begin(), as.end());
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    const std::vector<int> cpu_sorted = computeCPU(as);

    gpu::gpu_mem_32i as_gpu;
    gpu::gpu_mem_32i bs_gpu;

    as_gpu.resizeN(n);
    bs_gpu.resizeN(n);

    const int WORKGROUP_SIZE = 128;
    auto work_size = gpu::WorkSize(WORKGROUP_SIZE, n);

    {
        ocl::Kernel merge_global(merge_kernel, merge_kernel_length, "merge_global");
        // ocl::Kernel merge_sort_small(merge_kernel, merge_kernel_length, "merge_sort_small");
        merge_global.compile();
        // merge_sort_small.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();

            // merge_sort_small.exec(work_size, as_gpu);
            // int block_size = WORKGROUP_SIZE;
            int block_size = 1;
            while (block_size < n) {
                merge_global.exec(work_size, as_gpu, bs_gpu, block_size);
                std::swap(as_gpu, bs_gpu);
                block_size *= 2;
            }

            t.nextLap();
        }
        std::cout << "GPU global: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU global: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
        as_gpu.readN(as.data(), n);

        for (int i = 0; i < n; ++i) {
            EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
        }
    }

    // Если убрать это, я получаю значительно лучшие результаты, потому что as уже отсортирован
    as = initial_as;

    {
        gpu::gpu_mem_32u ind_gpu;
        ind_gpu.resizeN(n / WORKGROUP_SIZE);

        ocl::Kernel merge_sort_small(merge_kernel, merge_kernel_length, "merge_sort_small");
        ocl::Kernel calculate_indices(merge_kernel, merge_kernel_length, "calculate_indices");
        ocl::Kernel merge_local(merge_kernel, merge_kernel_length, "merge_local");
        merge_sort_small.compile();
        calculate_indices.compile();
        merge_local.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();

            merge_sort_small.exec(work_size, as_gpu);

            int block_size = WORKGROUP_SIZE;
            while (block_size < n) {
                calculate_indices.exec(gpu::WorkSize(WORKGROUP_SIZE, n / WORKGROUP_SIZE), as_gpu, ind_gpu, block_size);
                merge_local.exec(work_size, as_gpu, ind_gpu, bs_gpu, block_size, n / WORKGROUP_SIZE);
                std::swap(as_gpu, bs_gpu);
                block_size *= 2;
            }

            t.nextLap();
        }
        std::cout << "GPU local: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU local: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
        as_gpu.readN(as.data(), n);

        for (int i = 0; i < n; ++i) {
            EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
        }
    }

    return 0;
}
