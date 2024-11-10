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

    std::cout << "compiling kernels..." << std::endl;

    ocl::Kernel count_by_wg(radix_kernel, radix_kernel_length, "count_by_wg");
    count_by_wg.compile();
    ocl::Kernel matrix_transpose(radix_kernel, radix_kernel_length, "matrix_transpose");
    matrix_transpose.compile();
    ocl::Kernel prefix_stage1(radix_kernel, radix_kernel_length, "prefix_stage1");
    prefix_stage1.compile();
    ocl::Kernel prefix_stage2(radix_kernel, radix_kernel_length, "prefix_stage2");
    prefix_stage2.compile();
    ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");
    radix_sort.compile();

    std::cout << "all kernels are compiled..." << std::endl;

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    gpu::gpu_mem_32u bs_gpu;
    bs_gpu.resizeN(n);

    unsigned int workgroup_size = 128;

    unsigned int nbits = 4;

    std::cout <<  "creating buffers..."  << std::endl;

    unsigned int total_counters_count = (1 << nbits) * (n / workgroup_size);
    gpu::gpu_mem_32u counters_gpu;
    counters_gpu.resizeN(total_counters_count);
    gpu::gpu_mem_32u prefix_sums_gpu;
    prefix_sums_gpu.resizeN(total_counters_count);
    std::vector<unsigned int> tmp(total_counters_count, 0);

    std::cout << "starting sort..."  << std::endl;
    {
        timer t;
        timer t_count;
        timer t_prefix;
        timer t_radix;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();

            for (int bit_shift = 0; bit_shift < 32; bit_shift += nbits) {

                t_count.restart();
                count_by_wg.exec(
                        gpu::WorkSize(workgroup_size, n),
                        as_gpu,
                        counters_gpu,
                        bit_shift
                );
                t_count.nextLap();

                std::cout << "values counted" << std::endl;


                t_prefix.restart();
                int step = 1;
                while ((1 << step) <= total_counters_count) {
                    prefix_stage1.exec(gpu::WorkSize(workgroup_size, total_counters_count / ((1 << step))), counters_gpu,
                                       step, total_counters_count);
                    ++step;
                }
                step -= 2;

                while (step >= 1) {
                    prefix_stage2.exec(gpu::WorkSize(workgroup_size, total_counters_count / ((1 << step)) - 1), counters_gpu,
                                       step, total_counters_count);
                    --step;
                }
                t_prefix.nextLap();

                std::cout << "prefix sums calculated"  << std::endl;

                t_radix.restart();
                radix_sort.exec(gpu::WorkSize(workgroup_size, n), as_gpu, bs_gpu, counters_gpu, bit_shift, n);
                t_radix.nextLap();

                std::cout <<  "sorted"  << std::endl;

                std::swap(as_gpu, bs_gpu);

            }
            t.nextLap();
        }
        t.stop();

//        std::cout << "GPU count values: " << t_count.lapAvg() * (32 / nbits) << "+-" << t_count.lapStd() * (32 / nbits) << " s" << std::endl;
//        std::cout << "GPU calculate prefix sums: " << t_prefix.lapAvg() * (32 / nbits) << "+-" << t_prefix.lapStd() * (32 / nbits) << " s" << std::endl;
//        std::cout << "GPU radix sort: " << t_radix.lapAvg() * (32 / nbits) << "+-" << t_radix.lapStd() * (32 / nbits) << " s" << std::endl;
//
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
