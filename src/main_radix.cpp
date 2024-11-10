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
const unsigned int n = 32 * 1024; //* 1024;
const unsigned int wg_size = 128;
const unsigned int wg_tp_size = 16;
const unsigned int bits = 4;

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

void prefix_sum(ocl::Kernel reduce, ocl::Kernel down_sweep, gpu::gpu_mem_32u& res_gpu, unsigned int n) {
    std::vector<gpu::gpu_mem_32u> as_gpus;
    for(int d = 1; d <= n; d <<= 1) {
        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n / d);
        as_gpus.push_back(std::move(as_gpu));
    }

    as_gpus[0] = res_gpu;
    for (int logd = 1; logd < as_gpus.size(); logd++) {
        const unsigned int curn = n / (1 << logd);
        reduce.exec(gpu::WorkSize(std::min(curn, wg_size), curn), as_gpus[logd - 1], as_gpus[logd]);
    }
    for (int logd = as_gpus.size() - 2; logd >= 0; logd--) {
        const unsigned int curn = n / (1 << logd);
        down_sweep.exec(gpu::WorkSize(std::min(curn, wg_size), curn), as_gpus[logd + 1], as_gpus[logd]);
    }

    res_gpu = as_gpus[0];
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
        ocl::Kernel assign_zeros(radix_kernel, radix_kernel_length, "assign_zeros");
        assign_zeros.compile();

        ocl::Kernel count_workgroup(radix_kernel, radix_kernel_length, "count_workgroup");
        count_workgroup.compile();

        ocl::Kernel matrix_transpose(radix_kernel, radix_kernel_length, "matrix_transpose");
        matrix_transpose.compile();

        ocl::Kernel reduce(radix_kernel, radix_kernel_length, "reduce");
        reduce.compile();

        ocl::Kernel down_sweep(radix_kernel, radix_kernel_length, "down_sweep");
        down_sweep.compile();

        ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");
        radix_sort.compile();

        const unsigned int wg_cnt = n / wg_size;
        const unsigned int cnt_size = wg_cnt * (1 << bits);

        gpu::gpu_mem_32u as_gpu, bs_gpu, cnt_gpu, cnt_gpu_tp;
        as_gpu.resizeN(n);
        bs_gpu.resizeN(n);
        cnt_gpu.resizeN(cnt_size);
        cnt_gpu_tp.resizeN(cnt_size);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), as.size());
            t.restart();

            for (int shift = 0; shift < 32; shift += bits) {
                assign_zeros.exec(gpu::WorkSize(wg_size, cnt_size), cnt_gpu);

                count_workgroup.exec(gpu::WorkSize(wg_size, n), as_gpu, cnt_gpu, shift);

                matrix_transpose.exec(gpu::WorkSize(wg_tp_size, wg_tp_size, 1 << bits, wg_cnt), cnt_gpu, cnt_gpu_tp, wg_cnt, 1 << bits);

                prefix_sum(reduce, down_sweep, cnt_gpu_tp, cnt_size);

                radix_sort.exec(gpu::WorkSize(wg_size, n), as_gpu, bs_gpu, cnt_gpu_tp, wg_cnt, shift);

                std::swap(as_gpu, bs_gpu);
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
