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

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);
    as_gpu.writeN(as.data(), n);
    {
        ocl::Kernel reset_to_zeros(radix_kernel, radix_kernel_length, "reset_to_zeros");
        reset_to_zeros.compile();
        ocl::Kernel count(radix_kernel, radix_kernel_length, "count");
        count.compile();
        ocl::Kernel pref_sum(radix_kernel, radix_kernel_length, "pref_sum");
        pref_sum.compile();
        ocl::Kernel matrix_transpose(radix_kernel, radix_kernel_length, "matrix_transpose");
        matrix_transpose.compile();
        ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");
        radix_sort.compile();

        const int wg_size = 128;
        const int nbits = 4;
        const int uint_size = 32;

        gpu::gpu_mem_32u cs_gpu;
        cs_gpu.resizeN((1 << nbits) * n / wg_size);
        gpu::gpu_mem_32u cs_t_gpu;
        cs_t_gpu.resizeN((1 << nbits) * n / wg_size);
        gpu::gpu_mem_32u res_gpu;
        res_gpu.resizeN(n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            t.restart();
            for (int bit_shift = 0; bit_shift < uint_size; bit_shift += nbits) {
                reset_to_zeros.exec(gpu::WorkSize(128, (1 << nbits) * n / wg_size), cs_gpu);

                count.exec(gpu::WorkSize(wg_size, n), as_gpu, cs_gpu, bit_shift, nbits);

                matrix_transpose.exec(
                        gpu::WorkSize(16, 16, (1 << nbits), n / wg_size),
                        cs_gpu, cs_t_gpu,
                        (1 << nbits),
                        n / wg_size
                );

//                {
//                    std::vector<unsigned> cs(n / wg_size * (1 << nbits));
//                    cs_gpu.readN(cs.data(), cs.size());
//                    std::vector<unsigned> cs_t(cs.size());
//                    cs_t_gpu.readN(cs_t.data(), cs_t.size());
//
//                    for (int i = 0; i < n / wg_size; i++) {
//                        for (int j = 0; j < (1 << nbits); j++) {
//                            std::cout << cs[i * (1 << nbits) + j] << " ";
//                        }
//                        std::cout << std::endl;
//                    }
//                    std::cout << "=======================" << std::endl;
//                    for (int i = 0; i < (1 << nbits); i++) {
//                        for (int j = 0; j < (n / wg_size); j++) {
//                            std::cout << cs_t[i * (n / wg_size) + j] << " ";
//                        }
//                        std::cout << std::endl;
//                    }
//                }


                for (unsigned offset = 1; offset < (1 << nbits) * n / wg_size; offset <<= 1) {
                    gpu::WorkSize ws(wg_size, (1 << nbits) * n / wg_size / (offset << 1));
                    pref_sum.exec(ws, cs_t_gpu, offset, (1 << nbits) * n / wg_size, 0);
                }
                for (unsigned offset = (1 << nbits) * n / wg_size >> 2; offset > 0; offset >>= 1) {
                    gpu::WorkSize ws(wg_size, (1 << nbits) * n / wg_size / (offset << 1));
                    pref_sum.exec(ws, cs_t_gpu, offset, (1 << nbits) * n / wg_size, 1);
                }

//                {
//                    std::vector<unsigned> cs_t((1 << nbits) * n / wg_size);
//                    cs_t_gpu.readN(cs_t.data(), cs_t.size());
//
//                    for (int i = 0; i < (1 << nbits); i++) {
//                        for (int j = 0; j < (n / wg_size); j++) {
//                            std::cout << cs_t[i * (n / wg_size) + j] << " ";
//                        }
//                        std::cout << std::endl;
//                    }
//                }

                radix_sort.exec(
                        gpu::WorkSize(128, n),
                        as_gpu, res_gpu,
                        cs_t_gpu,
                        bit_shift, nbits
                );

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
