#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <algorithm>
#include <cassert>
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

    gpu::gpu_mem_32u as_gpu;
    gpu::gpu_mem_32u bs_gpu;
    gpu::gpu_mem_32u buf1_gpu;
    gpu::gpu_mem_32u buf2_gpu;

    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    const std::vector<unsigned int> cpu_reference = computeCPU(as);

    const unsigned int work_group_size = 128;
    const unsigned int nbits = 4;
    const unsigned int nwork_groups = (n + work_group_size - 1) / work_group_size;
    const unsigned int buf_size = nwork_groups * (1 << nbits);

    {
        as_gpu.resizeN(n);
        bs_gpu.resizeN(n);
        buf1_gpu.resizeN(buf_size);
        buf2_gpu.resizeN(buf_size);

        ocl::Kernel radix_write_zeros(
            radix_kernel,
            radix_kernel_length,
            "write_zeros"
        );
        radix_write_zeros.compile();

        ocl::Kernel radix_count(
            radix_kernel,
            radix_kernel_length,
            "count"
        );
        radix_count.compile();

        ocl::Kernel radix_transpose(
            radix_kernel,
            radix_kernel_length,
            "transpose"
        );
        radix_transpose.compile();

        ocl::Kernel radix_prefix_sum_pass1(
            radix_kernel,
            radix_kernel_length,
            "prefix_sum_pass1"
        );
        radix_prefix_sum_pass1.compile();

        ocl::Kernel radix_prefix_sum_pass2(
            radix_kernel,
            radix_kernel_length,
            "prefix_sum_pass2"
        );
        radix_prefix_sum_pass2.compile();

        ocl::Kernel radix_sort(
            radix_kernel,
            radix_kernel_length,
            "sort"
        );
        radix_sort.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            t.restart();

            for (
                unsigned int bit_shift = 0;
                bit_shift < 8 * sizeof(unsigned int);
                bit_shift += nbits
            ) {
                // write zeros
                {
                    const unsigned int work_size = nwork_groups * (1 << nbits);
                    radix_write_zeros.exec(
                        gpu::WorkSize(work_group_size, work_size),
                        buf1_gpu,
                        buf_size
                    );
                }

                // count
                {
                    const unsigned int work_size = nwork_groups * work_group_size;
                    radix_count.exec(
                        gpu::WorkSize(work_group_size, work_size),
                        as_gpu,
                        n,
                        buf1_gpu,
                        bit_shift
                    );
                }

                // transpose
                {
                    const unsigned int M = nwork_groups;
                    const unsigned int K = 1 << nbits;
                    const unsigned int tile_size = std::min({M, K, 16u});
                    assert((tile_size & (tile_size - 1)) == 0 && "tile_size should be a power of 2");
                    radix_transpose.exec(
                        gpu::WorkSize(
                            tile_size,
                            tile_size,
                            tile_size * ((K + tile_size - 1) / tile_size),
                            tile_size * ((M + tile_size - 1) / tile_size)
                        ),
                        buf1_gpu,
                        buf2_gpu,
                        M,
                        K
                    );
                }

                // prefix sum
                {
                    for (int block_size = 2; block_size <= buf_size; block_size *= 2) {
                        unsigned int work_size = buf_size / block_size;
                        work_size = work_group_size * ((work_size + work_group_size - 1) / work_group_size);
                        radix_prefix_sum_pass1.exec(
                            gpu::WorkSize(work_group_size, work_size),
                            buf2_gpu,
                            buf_size,
                            block_size
                        );
                    }

                    for (int block_size = buf_size / 4; block_size > 0; block_size /= 2) {
                        unsigned int work_size = buf_size / (block_size * 2) - 1;
                        work_size = work_group_size * ((work_size + work_group_size - 1) / work_group_size);
                        radix_prefix_sum_pass2.exec(
                            gpu::WorkSize(work_group_size, work_size),
                            buf2_gpu,
                            buf_size,
                            block_size
                        );
                    }
                }

                // sort
                {
                    const unsigned int work_size = nwork_groups * work_group_size;
                    radix_sort.exec(
                        gpu::WorkSize(work_group_size, work_size),
                        as_gpu,
                        bs_gpu,
                        n,
                        buf2_gpu,
                        bit_shift
                    );
                }

                as_gpu.swap(bs_gpu);

                t.nextLap();
            }
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
