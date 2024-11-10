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

#define debug false

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

struct KernelConfig {
    std::string kernel_name;
    gpu::WorkSize work_size;
    std::string defines;
    std::string prefix;
};

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

    ocl::Kernel write_zeros_kernel(radix_kernel, radix_kernel_length, "write_zeros");
    write_zeros_kernel.compile();
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

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    gpu::gpu_mem_32u bs_gpu;
    bs_gpu.resizeN(n);

    unsigned int workgroup_size = 2;

    unsigned int nbits = 2;

    unsigned int total_counters_count = (1 << nbits) * (n / workgroup_size);
    gpu::gpu_mem_32u counters_gpu;
    counters_gpu.resizeN(total_counters_count);
    gpu::gpu_mem_32u prefix_sums_gpu;
    prefix_sums_gpu.resizeN(total_counters_count);
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            if (debug) {
                for (auto kek: as) {
                    std::cout << kek << " ";
                }
                std::cout << std::endl;
            }
            as_gpu.writeN(as.data(), n);

            // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            for (int bit_shift = 0; bit_shift < 32; bit_shift += nbits) {
                write_zeros_kernel.exec(gpu::WorkSize(workgroup_size, total_counters_count), counters_gpu);

                std::vector<unsigned int> tmp(total_counters_count, 0);

                if (debug) {
                    counters_gpu.readN(tmp.data(), total_counters_count);
                    std::cout << "counters" << std::endl;
                    for (int i = 0; i < (1 << nbits); ++i) {
                        for (int j = 0; j < n / workgroup_size; ++j) {
                            std::cout << tmp[i * (n / workgroup_size) + j] << " ";
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
                }

                // calculate counters
                count_by_wg.exec(
                        gpu::WorkSize(workgroup_size, n),
                        as_gpu,
                        counters_gpu,
                        bit_shift
                );

                if (debug) {
                    counters_gpu.readN(tmp.data(), total_counters_count);
                    std::cout << "counters" << std::endl;
                    for (int i = 0; i < (1 << nbits); ++i) {
                        for (int j = 0; j < n / workgroup_size; ++j) {
                            std::cout << tmp[i * (n / workgroup_size) + j] << " ";
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
                }


                // count prefix sums
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

                // transpose in favor of coalesced access
//                unsigned int global_work_size_x = ((1 << nbits)  + workgroup_size - 1) / workgroup_size * workgroup_size;
//                unsigned int global_work_size_y = ( n / workgroup_size+ workgroup_size - 1) / workgroup_size * workgroup_size;
//                matrix_transpose.exec(
//                        gpu::WorkSize(1, 1, global_work_size_x, global_work_size_y),
//                        counters_gpu,
//                        prefix_sums_gpu,
//                        n / workgroup_size
//                        );

                if (debug) {
                    std::cout << "prefix sums ready" << std::endl;
                    counters_gpu.readN(tmp.data(), total_counters_count);
                    for (int i = 0; i < (1 << nbits); ++i) {
                        for (int j = 0; j < n / workgroup_size; ++j) {
                            std::cout << tmp[i * (n / workgroup_size) + j] << " ";
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
                }
                // sort
                radix_sort.exec(gpu::WorkSize(workgroup_size, n), as_gpu, bs_gpu, counters_gpu, bit_shift, n);

                if (debug) {
                    std::cout << "sorted arr" << std::endl;
                    bs_gpu.readN(tmp.data(), n);
                    for (int i = 0; i < n; ++i) {
                        std::cout << tmp[i] << " ";
                    }
                    std::cout << std::endl;
                }
                std::swap(as_gpu, bs_gpu);
            }
            printf("iter %d ready\n", iter);
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
