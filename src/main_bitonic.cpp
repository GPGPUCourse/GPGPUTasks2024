#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/bitonic_cl.h"

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

unsigned int get_near_upper_2_pow(int k) {
    --k;
    int length = 1;
    while ((k >> length) > 0) {
        k = k | (k >> length);
        ++length;
    }
    return k + 1;
}

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
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    const std::vector<int> cpu_sorted = computeCPU(as);

    gpu::gpu_mem_32i as_gpu;
    unsigned int n_two_pow = get_near_upper_2_pow(n);

    as_gpu.resizeN(n_two_pow);

    {
        ocl::Kernel bitonic(bitonic_kernel, bitonic_kernel_length, "bitonic");
        bitonic.compile();
        ocl::Kernel fill_infinity(bitonic_kernel, bitonic_kernel_length, "fill_infinity");
        fill_infinity.compile();
        const unsigned int workGroupSize = 128;
        const unsigned int global_work_size = (n_two_pow + workGroupSize - 1) / workGroupSize * workGroupSize;
        const unsigned int global_fill_work_size = (n_two_pow - n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            if (global_fill_work_size != 0) {
                fill_infinity.exec(gpu::WorkSize(workGroupSize, global_fill_work_size), as_gpu, n, n_two_pow);
            }
            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            for (unsigned int half_superblock_size = 1; half_superblock_size < n_two_pow; half_superblock_size *= 2)
                for (unsigned int operation_index = 1; operation_index <= half_superblock_size; operation_index *= 2)
                    bitonic.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, n_two_pow, half_superblock_size, operation_index);
            t.nextLap();
        }

        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }


    as_gpu.readN(as.data(), n);

    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}