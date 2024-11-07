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
#include <assert.h>

const int benchmarkingIters = 1;
const int benchmarkingItersCPU = 1;
const unsigned int n = 32;

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
    return 31 - __builtin_clz(n);
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
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max()) % 4;
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    const std::vector<unsigned int> cpu_reference = computeCPU(as);

    unsigned int workSize = n;
    unsigned int workGroupSize = 4;
    unsigned int nWorkGroups = workSize / workGroupSize;
    unsigned int bitsPerDigit = 2;
    unsigned int nDigits = (1 << bitsPerDigit);
    unsigned int tileSize = 2;

    assert(nDigits <= workGroupSize);
    assert(tileSize * tileSize == workGroupSize);

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(workSize);
    as_gpu.writeN(as.data(), workSize);
    gpu::gpu_mem_32u bs_gpu;
    bs_gpu.resizeN(workSize);
    gpu::gpu_mem_32u cs_gpu;
    cs_gpu.resizeN(nWorkGroups * nDigits);
    gpu::gpu_mem_32u cs_t_gpu;
    cs_t_gpu.resizeN(nWorkGroups * nDigits);

    std::string defines = "-DWORK_SIZE=" + std::to_string(workSize) +
                          " -DWORK_GROUP_SIZE=" + std::to_string(workGroupSize) +
                          " -DN_WORK_GROUPS=" + std::to_string(nWorkGroups) +
                          " -DBITS_PER_DIGIT=" + std::to_string(bitsPerDigit) +
                          " -DN_DIGITS=" + std::to_string(nDigits) +
                          " -DTILE_SIZE=" + std::to_string(tileSize);

    ocl::Kernel count(radix_kernel, radix_kernel_length, "count", defines);
    ocl::Kernel transpose(radix_kernel, radix_kernel_length, "transpose", defines);
    ocl::Kernel up_sweep(radix_kernel, radix_kernel_length, "up_sweep", defines);
    ocl::Kernel down_sweep(radix_kernel, radix_kernel_length, "down_sweep", defines);
    ocl::Kernel move(radix_kernel, radix_kernel_length, "move", defines);
    count.compile(true);
    up_sweep.compile(true);
    down_sweep.compile(true);
    move.compile(true);

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            t.restart();
            for (unsigned int digit_no = 0; digit_no < (32 / bitsPerDigit); digit_no++) {
                count.exec({workGroupSize, workSize}, as_gpu, cs_gpu, digit_no);
                transpose.exec({tileSize, tileSize, nDigits, nWorkGroups}, cs_gpu, cs_t_gpu);

                std::vector<unsigned int> debug_buf1;
                debug_buf1.resize(nWorkGroups * nDigits);
                cs_t_gpu.readN(debug_buf1.data(), nWorkGroups * nDigits);

                execPrefixSum(up_sweep, down_sweep, cs_t_gpu, nWorkGroups * nDigits, workGroupSize);

                std::vector<unsigned int> debug_buf2;
                debug_buf2.resize(nWorkGroups * nDigits);
                cs_t_gpu.readN(debug_buf2.data(), nWorkGroups * nDigits);

                move.exec({workGroupSize, workSize}, as_gpu, bs_gpu, cs_t_gpu, digit_no);
            }
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
