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

void prefixSum(gpu::gpu_mem_32u &as_gpu, unsigned int n, unsigned int groupSize, ocl::Kernel &prefix_up, ocl::Kernel &prefix_down){
    int64_t global_step = 1;
    while (global_step < n) {
        const unsigned int workSize =
                (n + 2 * groupSize * global_step - 1) / (2 * groupSize * global_step) * groupSize;
        prefix_up.exec(
                gpu::WorkSize(groupSize, workSize),
                as_gpu, (int64_t) n, (int64_t) global_step
        );
        global_step *= 2 * groupSize;
    }

    global_step /= (2 * groupSize * 2 * groupSize);

    while (global_step > 0) {
        const unsigned int workSize =
                (n + 2 * groupSize * global_step - 1) / (2 * groupSize * global_step) * groupSize * 2;
        prefix_down.exec(
                gpu::WorkSize(2 * groupSize, workSize),
                as_gpu, (int64_t) n, (int64_t) global_step
        );
        global_step /= (2 * groupSize);
    }
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
    std::cout << std::endl;
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    const std::vector<unsigned int> cpu_reference = computeCPU(as);

    {
        const unsigned int groupSize = std::min(128u, n);
        const unsigned int bits = 4;

        std::string prefix_sum_defines = "-DGROUP_SIZE=" + std::to_string(groupSize) + " -DUSE_PREFIX_SUM";
        ocl::Kernel prefix_up(radix_kernel, radix_kernel_length, "prefix_up", prefix_sum_defines);
        prefix_up.compile();
        ocl::Kernel prefix_down(radix_kernel, radix_kernel_length, "prefix_down", prefix_sum_defines);
        prefix_down.compile();

        const unsigned int tileSize = std::min(16u, std::min(n, (1u << bits)));
        std::string transpose_defines = "-DTILE_SIZE=" + std::to_string(tileSize) + " -DUSE_TRANSPOSE";
        ocl::Kernel transpose_kernel(radix_kernel, radix_kernel_length, "transpose", transpose_defines);
        transpose_kernel.compile();


        std::string radix_defines = "-DRADIX_BITS=" + std::to_string(bits);
        ocl::Kernel local_radix_kernel(radix_kernel, radix_kernel_length, "local_radix", radix_defines);
        local_radix_kernel.compile();

        ocl::Kernel index_radix_kernel(radix_kernel, radix_kernel_length, "index_radix", radix_defines);
        index_radix_kernel.compile();

        const unsigned int groups = ((int) n + groupSize - 1) / groupSize;
        gpu::gpu_mem_32u buckets, buckets2;
        buckets.resizeN(groups * (1u << bits)); buckets2.resizeN(groups * (1u << bits));

        gpu::gpu_mem_32u as_gpu, as_gpu2;
        as_gpu.resizeN(n), as_gpu2.resizeN(n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();

            for (unsigned int step = 0; step < 32; step += bits) {

                local_radix_kernel.exec(gpu::WorkSize(groupSize, groups * groupSize), as_gpu, n, step, buckets);

                transpose_kernel.exec(
                        gpu::WorkSize(tileSize, tileSize, ((int) (1 << bits) + tileSize - 1) / tileSize * tileSize, (groups + tileSize - 1) / tileSize * tileSize),
                        buckets, buckets2, groups, (1u << bits)
                );

                prefixSum(buckets2, groups * (1u << bits), groupSize, prefix_up, prefix_down);

                transpose_kernel.exec(
                        gpu::WorkSize(tileSize, tileSize, (groups + tileSize - 1) / tileSize * tileSize, ((int) (1 << bits) + tileSize - 1) / tileSize * tileSize),
                        buckets2, buckets, (1u << bits), groups
                );

                index_radix_kernel.exec(gpu::WorkSize(groupSize, groups * groupSize), buckets, n, step, as_gpu, as_gpu2);
                std::swap(as_gpu, as_gpu2);

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
