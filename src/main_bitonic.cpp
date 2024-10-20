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
    as_gpu.resizeN(n);

    {
        ocl::Kernel bitonic(bitonic_kernel, bitonic_kernel_length, "bitonic");
        bitonic.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            int wg_size = 128;
            gpu::WorkSize work_size(wg_size, n >> 1);
            for (int blocks_size = 1; blocks_size < n; blocks_size <<= 1) {
                for (int sub_blocks_size = blocks_size; sub_blocks_size > 0; sub_blocks_size >>= 1) {
                    bitonic.exec(work_size, as_gpu, blocks_size, sub_blocks_size);
                }
            }

            t.nextLap();
        }

        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

//    {
//        ocl::Kernel bitonic_on_shifts(bitonic_kernel, bitonic_kernel_length, "bitonic_on_shifts");
//        bitonic_on_shifts.compile();
//
//        timer t;
//        for (int iter = 0; iter < benchmarkingIters; ++iter) {
//            as_gpu.writeN(as.data(), n);
//            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
//
//            int wg_size = 128;
//            gpu::WorkSize work_size(wg_size, n >> 1);
//            for (int blocks_size_log = 0; (1 << blocks_size_log) < n; blocks_size_log += 1) {
//                for (int sub_blocks_size_log = blocks_size_log; sub_blocks_size_log > -1; sub_blocks_size_log -= 1) {
//                    bitonic_on_shifts.exec(work_size, as_gpu, blocks_size_log, sub_blocks_size_log);
//                }
//            }
//
//            t.nextLap();
//        }
//
//        std::cout << "(shifts) GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
//        std::cout << "(shifts) GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
//    }


    as_gpu.readN(as.data(), n);

    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}