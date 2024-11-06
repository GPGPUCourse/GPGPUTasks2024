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

const int benchmarkingIters = 1;
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

unsigned int mylog(unsigned int n) {
    return 32 - __builtin_clz(n);
}

void execPrefixSum(ocl::Kernel &up_sweep, ocl::Kernel &down_sweep, gpu::gpu_mem_32u &as_gpu, unsigned int workSize, unsigned int workGroupSize) {
    unsigned int logWorkSize = mylog(workSize);
    for (int d = 0; d <= logWorkSize - 1; d++) {
        gpu::WorkSize ws{workGroupSize, (workSize >> (d + 1))};
        up_sweep.exec(ws, as_gpu, workSize, d);
    }
    for (int d = logWorkSize - 1; d >= 0; d--) {
        gpu::WorkSize ws{workGroupSize, (workSize >> (d + 1))};
        down_sweep.exec(ws, as_gpu, workSize, d);
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

    unsigned int workSize = n;
    unsigned int workGroupSize = 64;
    unsigned int bitsPerDigit = 4;

    unsigned int workGroupsCount = workSize / workGroupSize;
    unsigned digitsCount = (1 << bitsPerDigit);

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);
    gpu::gpu_mem_32u bs_gpu;
    bs_gpu.resizeN(n);
    gpu::gpu_mem_32u cs_gpu;
    cs_gpu.resizeN(workGroupsCount * digitsCount);

    ocl::Kernel count(radix_kernel, radix_kernel_length, "count");
    ocl::Kernel transpose(radix_kernel, radix_kernel_length, "transpose");
    ocl::Kernel up_sweep(radix_kernel, radix_kernel_length, "up_sweep");
    ocl::Kernel down_sweep(radix_kernel, radix_kernel_length, "down_sweep");
    ocl::Kernel move(radix_kernel, radix_kernel_length, "move");
    count.compile(true);
    up_sweep.compile(true);
    down_sweep.compile(true);
    move.compile(true);

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            t.restart();
            count.exec({workGroupSize, workSize}, as_gpu, cs_gpu, workSize, workGroupsCount, digitsCount, bitsPerDigit);
            transpose.exec({workGroupSize, workGroupsCount * digitsCount}, cs_gpu, workGroupsCount, digitsCount);
            execPrefixSum(up_sweep, down_sweep, cs_gpu, workGroupsCount * digitsCount, workGroupSize);
            move.exec({workGroupSize, workSize}, as_gpu, bs_gpu, cs_gpu, workSize, workGroupsCount, digitsCount, bitsPerDigit);
            t.nextLap();
        }
        t.stop();

        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_reference[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}
