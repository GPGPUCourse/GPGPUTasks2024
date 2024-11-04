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
#include <climits>

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

unsigned int get_near_upper_2_pow(int k) {
    --k;
    int length = 1;
    while ((k >> length) > 0) {
        k = k | (k >> length);
        ++length;
    }
    return k + 1;
}

inline void run_prefix_sum(unsigned two_powered_size, unsigned work_group_size, ocl::Kernel &prefix_sum, ocl::Kernel &refresh, gpu::gpu_mem_32u &input_gpu) {
    for (unsigned int block_size = 2; block_size <= two_powered_size; block_size *= 2) {
        const unsigned int global_work_size =
                (two_powered_size / block_size + work_group_size - 1) / work_group_size * work_group_size;
        prefix_sum.exec(gpu::WorkSize(work_group_size, global_work_size), input_gpu, two_powered_size, block_size);
    }
    for (unsigned int block_size = two_powered_size / 2; block_size >= 2; block_size /= 2) {
        const unsigned int global_work_size =
                (two_powered_size / block_size - 1 + work_group_size - 1) / work_group_size * work_group_size;
        refresh.exec(gpu::WorkSize(work_group_size, global_work_size), input_gpu, two_powered_size, block_size);
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
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    const std::vector<unsigned int> cpu_reference = computeCPU(as);

    // remove me
    // return 0;
    gpu::gpu_mem_32u as_gpu;
    gpu::gpu_mem_32u res_gpu;
    gpu::gpu_mem_32u buffer_gpu;
    as_gpu.resizeN(n);
    res_gpu.resizeN(n);
    constexpr unsigned nbits = 4;
    constexpr unsigned int_size = sizeof(unsigned int) * CHAR_BIT;
    static_assert(int_size % nbits == 0);
    const unsigned work_group_size = 128;
    unsigned work_group_need = (n + work_group_size - 1) / work_group_size;
    unsigned buffer_size = get_near_upper_2_pow(work_group_need * (1 << nbits));
    buffer_gpu.resizeN(buffer_size);
    unsigned n_work_size = (n + work_group_size - 1) / work_group_size * work_group_size;
    unsigned buffer_work_size = (buffer_size + work_group_size - 1) / work_group_size * work_group_size;

    auto defines = "-Dnbits=" + std::to_string(nbits) + " -Dwork_group_need=" + std::to_string(work_group_need) + " -Dwork_group_size=" + std::to_string(work_group_size);
    ocl::Kernel prefix_sum(radix_kernel, radix_kernel_length, "work_efficient_sum", defines);
    prefix_sum.compile();
    ocl::Kernel fill_with_zeros(radix_kernel, radix_kernel_length, "fill_with_zeros", defines);
    fill_with_zeros.compile();
    ocl::Kernel refresh(radix_kernel, radix_kernel_length, "refresh", defines);
    refresh.compile();
    ocl::Kernel work_group_counter(radix_kernel, radix_kernel_length, "work_group_counter", defines);
    work_group_counter.compile();
    ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort", defines);
    radix_sort.compile();
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            as_gpu.writeN(as.data(), n);
            t.restart();
            for (unsigned int bit_shift = 0; bit_shift < int_size; bit_shift += nbits) {
                fill_with_zeros.exec(gpu::WorkSize(work_group_size, buffer_work_size), buffer_gpu, buffer_size);
                work_group_counter.exec(gpu::WorkSize(work_group_size, n_work_size), as_gpu, n, bit_shift, buffer_gpu);
                run_prefix_sum(buffer_size, work_group_size, prefix_sum, refresh, buffer_gpu);
                radix_sort.exec(gpu::WorkSize(work_group_size, n_work_size), as_gpu, n, bit_shift, buffer_gpu, res_gpu);
                std::swap(as_gpu, res_gpu);
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
