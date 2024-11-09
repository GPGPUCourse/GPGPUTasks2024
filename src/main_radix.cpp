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

void compute_prefix_sum(ocl::Kernel& prefix_sum, gpu::gpu_mem_32u& as_gpu, unsigned int n, unsigned int workgroup_size) {
    int i;
    for (i = 1; i < n; i <<= 1)
        prefix_sum.exec(gpu::WorkSize(workgroup_size, n / (2 * i)), as_gpu, as_gpu, n, (i << 1) - 1, i * 2, i);
    for (i >>= 2; i > 0; i >>= 1)
        prefix_sum.exec(gpu::WorkSize(workgroup_size, n / (2 * i) - 1), as_gpu, as_gpu, n, n - i - 1, i * (-2), i);
}


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    ocl::Kernel write_zeros(radix_kernel, radix_kernel_length, "write_zeros");
    ocl::Kernel counters_by_workgroup(radix_kernel, radix_kernel_length, "counters_by_workgroup");
    ocl::Kernel matrix_transpose(radix_kernel, radix_kernel_length, "matrix_transpose");
    ocl::Kernel prefix_sum(radix_kernel, radix_kernel_length, "prefix_sum");
    ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");

    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    const std::vector<unsigned int> cpu_reference = computeCPU(as);

    const unsigned int workgroup_size = 128;
    const unsigned int transpose_workgroup_size = 16;
    const unsigned int nbits = 4;
    const unsigned int counter_dim = 1 << nbits;
    const unsigned int n_workgroups = (n + workgroup_size - 1) / workgroup_size;
    const unsigned int counters_size = n_workgroups * counter_dim;

    gpu::gpu_mem_32u as_gpu, bs_gpu, counters_gpu, counters_t_gpu;
    as_gpu.resizeN(n);
    bs_gpu.resizeN(n);
    counters_gpu.resizeN(counters_size);
    counters_t_gpu.resizeN(counters_size);

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), as.size());
            t.restart();
            for (int bit_shift = 0; bit_shift < 32; bit_shift += nbits) {
                write_zeros.exec(gpu::WorkSize(workgroup_size, counters_size), counters_gpu, counters_size);
                
                counters_by_workgroup.exec(gpu::WorkSize(workgroup_size, n), as_gpu, counters_gpu, n, nbits, bit_shift);
                
                matrix_transpose.exec(
                    gpu::WorkSize(transpose_workgroup_size, transpose_workgroup_size, counter_dim, n_workgroups), 
                    counters_gpu, counters_t_gpu, counter_dim, n_workgroups
                );
                
                compute_prefix_sum(prefix_sum, counters_t_gpu, counters_size, workgroup_size);
                
                radix_sort.exec(
                    gpu::WorkSize(workgroup_size, n), 
                    as_gpu, bs_gpu, counters_t_gpu, counters_gpu, n, nbits, bit_shift, n_workgroups
                );
                
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
