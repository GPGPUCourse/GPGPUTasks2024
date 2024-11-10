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
        ocl::Kernel matrix_transpose_local_good_banks(radix_kernel, radix_kernel_length,
                                                             "matrix_transpose_local_good_banks");
        ocl::Kernel prefix_sum_efficient_first(radix_kernel, radix_kernel_length, "prefix_sum_efficient_first");
        ocl::Kernel prefix_sum_efficient_second(radix_kernel, radix_kernel_length,
                                                       "prefix_sum_efficient_second");
        ocl::Kernel set_zeros(radix_kernel, radix_kernel_length, "set_zeros");
        ocl::Kernel count(radix_kernel, radix_kernel_length, "count");
        ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");

        matrix_transpose_local_good_banks.compile();
        prefix_sum_efficient_first.compile();
        prefix_sum_efficient_second.compile();
        set_zeros.compile();
        count.compile();
        radix_sort.compile();

        unsigned int n_bits = 4;
        unsigned int n_digits = 1 << n_bits;
        unsigned int workgroup_size = 128;
        unsigned int work_groups = (n + workgroup_size - 1) / workgroup_size;
        unsigned int global_size = work_groups * workgroup_size;
        unsigned int counter_size = work_groups * n_digits;

        gpu::gpu_mem_32u counters;
        counters.resizeN(counter_size);
        gpu::gpu_mem_32u countersT;
        countersT.resizeN(counter_size);
        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);
        gpu::gpu_mem_32u bs_gpu;
        bs_gpu.resizeN(n);

        

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();
            
            for (int bit_shift = 0; bit_shift < sizeof(unsigned int) * 8; bit_shift += n_bits) 
            {
                set_zeros.exec(gpu::WorkSize(workgroup_size, counter_size), counters);
                count.exec(gpu::WorkSize(workgroup_size, n), as_gpu, counters, bit_shift, n_bits);
                matrix_transpose_local_good_banks.exec(gpu::WorkSize(16, 16, n_digits, work_groups), counters, countersT, n_digits, work_groups);
                
                for (unsigned int step = 1; step < counter_size; step *= 2) {
                    prefix_sum_efficient_first.exec(gpu::WorkSize(workgroup_size, counter_size / step / 2), countersT,
                                                    counter_size, step);
                }
                for (unsigned int step = counter_size / 4; step > 0; step /= 2) {
                    prefix_sum_efficient_second.exec(gpu::WorkSize(workgroup_size, counter_size / step / 2), countersT,
                                                     counter_size, step);
                }

                radix_sort.exec(gpu::WorkSize(workgroup_size, n), as_gpu, bs_gpu, countersT, bit_shift,
                                             n_bits, n);
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
