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


#define GROUP_SIZE 128
#define N_BITS 4
#define TILE_SIZE 16
#define UNSIGNED_INT_SIZE 32

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

    ocl::Kernel init_zeros(radix_kernel, radix_kernel_length, "init_zeros");
    ocl::Kernel get_counts(radix_kernel, radix_kernel_length, "get_counts");
    ocl::Kernel matrix_transpose(radix_kernel, radix_kernel_length, "matrix_transpose");
    ocl::Kernel prefix_sum_we_up(radix_kernel, radix_kernel_length, "pref_sum_we_up");
    ocl::Kernel prefix_sum_we_down(radix_kernel, radix_kernel_length, "pref_sum_we_down");
    ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");

    init_zeros.compile();
    get_counts.compile();
    matrix_transpose.compile();
    prefix_sum_we_up.compile();
    prefix_sum_we_down.compile();
    radix_sort.compile();

    const unsigned int c_width = 1 << N_BITS;
    const unsigned int c_height = (n + GROUP_SIZE - 1) / GROUP_SIZE;
    const unsigned int c_size = c_width * c_height;

    const unsigned int c_w_work_size = (c_width + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;
    const unsigned int c_h_work_size = (c_height + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;

    gpu::WorkSize transpose_work_size(TILE_SIZE, TILE_SIZE, c_w_work_size, c_h_work_size);

    gpu::gpu_mem_32u as_gpu, bs_gpu, counters, counters_tr, counters_tr_ps;

    counters.resizeN(c_size);
    counters_tr.resizeN(c_size);
    as_gpu.resizeN(n);
    bs_gpu.resizeN(n);

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            bs_gpu.writeN(as.data(), n);

            t.restart();
            for (int bitshift = 0; bitshift < UNSIGNED_INT_SIZE; bitshift += N_BITS) {
                init_zeros.exec(gpu::WorkSize(GROUP_SIZE, c_size), counters, c_size);
                get_counts.exec(gpu::WorkSize(GROUP_SIZE, n), as_gpu, counters, bitshift);
                matrix_transpose.exec(transpose_work_size, counters, counters_tr, c_height, c_width);

                int d = 1;
                for (; (1 << d) <= c_size; d++) {
                    prefix_sum_we_up.exec(gpu::WorkSize(GROUP_SIZE, c_size >> d), counters_tr, counters_tr, d, c_size);
                }

                d -= 1;
                for (; d > 0; d--) {
                    prefix_sum_we_down.exec(gpu::WorkSize(GROUP_SIZE, c_size >> d), counters_tr, counters_tr, d, c_size);
                }

                radix_sort.exec(gpu::WorkSize(GROUP_SIZE, n), as_gpu, bs_gpu, counters_tr, bitshift, n);

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
