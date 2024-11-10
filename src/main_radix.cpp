#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "cl/radix_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>

const int benchmarkingIters = 1;
const int benchmarkingItersCPU = 1;
const unsigned int n = 32 * 1024 * 1024;
const uint32_t sortByNBits = 4;
const uint32_t tileSize = 16;
const int wgSize = 128;

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
        ocl::Kernel count_local_entries(radix_kernel, radix_kernel_length, "count_local_entries");
        ocl::Kernel transpose(radix_kernel, radix_kernel_length, "matrix_transpose");
        ocl::Kernel prefix_sum_up(radix_kernel, radix_kernel_length, "prefix_sum_up");
        ocl::Kernel prefix_sum_down(radix_kernel, radix_kernel_length, "prefix_sum_down");
        ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");

        count_local_entries.compile();
        transpose.compile();
        prefix_sum_down.compile();
        prefix_sum_up.compile();

        gpu::gpu_mem_32u as_gpu;
        gpu::gpu_mem_32u result;
        gpu::gpu_mem_32u group_cnt_gpu;
        gpu::gpu_mem_32u t_group_cnt_gpu;

        const unsigned int digits_in_mask = (1 << sortByNBits);

        as_gpu.resizeN(n);
        result.resizeN(n);
        group_cnt_gpu.resizeN(digits_in_mask * n / wgSize);
        t_group_cnt_gpu.resizeN(digits_in_mask * n / wgSize);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), as.size());
            t.restart();

            for (uint32_t i = 0; i < sizeof(*as.data()) * 8 / sortByNBits; ++i) {
                count_local_entries.exec(gpu::WorkSize{wgSize, n}, as_gpu, group_cnt_gpu, i * sortByNBits);
                transpose.exec(gpu::WorkSize{tileSize, tileSize, digits_in_mask, n / wgSize},
                               group_cnt_gpu, t_group_cnt_gpu, n / wgSize, digits_in_mask);

                const unsigned int workSizeGroupedBy = digits_in_mask * n / wgSize;
                for (uint32_t pow = 1; pow < workSizeGroupedBy; pow *= 2) {
                    prefix_sum_up.exec(gpu::WorkSize{wgSize, workSizeGroupedBy / (2 * pow)}, t_group_cnt_gpu, workSizeGroupedBy, pow);
                }
                for (uint32_t pow = workSizeGroupedBy / 4; pow >= 1; pow /= 2) {
                    prefix_sum_down.exec(gpu::WorkSize{wgSize, workSizeGroupedBy / (2 * pow)}, t_group_cnt_gpu, workSizeGroupedBy, pow);
                }
                radix_sort.exec(gpu::WorkSize{wgSize, n}, as_gpu, result, t_group_cnt_gpu, i * sortByNBits);
                std::swap(as_gpu, result);
            }


            t.nextLap();
        }
        t.stop();

        as_gpu.readN(as.data(), as.size());

        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    for (int i = 0; i < n; ++i) {
//        std::cout << as[i] << " " << cpu_reference[i] << std::endl;
        EXPECT_THE_SAME(as[i], cpu_reference[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}
