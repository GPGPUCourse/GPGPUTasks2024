#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_transpose_cl.h"
#include "libgpu/work_size.h"

#include <vector>
#include <iostream>
#include <stdexcept>

const int benchmarkingIters = 100;
const unsigned int M = 4096;
const unsigned int K = 4096;

void runTest(const std::string &kernel_name, const float *as)
{
    gpu::gpu_mem_32f as_gpu, as_t_gpu;
    as_gpu.resizeN(M*K);
    as_t_gpu.resizeN(K*M);

    as_gpu.writeN(as, M*K);

    ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, kernel_name);
    matrix_transpose_kernel.compile();

    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        unsigned int workGroupSizeDim = 16;
        unsigned int globalWorkSizeX = (K + workGroupSizeDim - 1) / workGroupSizeDim * workGroupSizeDim;
        unsigned int globalWorkSizeY = (M + workGroupSizeDim - 1) / workGroupSizeDim * workGroupSizeDim;
        gpu::WorkSize work_size(workGroupSizeDim, workGroupSizeDim, globalWorkSizeX, globalWorkSizeY);

        matrix_transpose_kernel.exec(work_size, as_gpu, as_t_gpu, M, K);

        t.nextLap();
    }

    std::cout << "[" << kernel_name << "]" << std::endl;
    std::cout << "    GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "    GPU: " << M*K/1000.0/1000.0 / t.lapAvg() << " millions/s" << std::endl;

    std::vector<float> as_t(M*K, 0);
    as_t_gpu.readN(as_t.data(), M*K);

    // Проверяем корректность результатов
    for (int j = 0; j < M; ++j) {
        for (int i = 0; i < K; ++i) {
            float a = as[j * K + i];
            float b = as_t[i * M + j];
            if (a != b) {
                throw std::runtime_error("Not the same!");
            }
        }
    }
}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    std::vector<float> as(M*K, 0);
    FastRandom r(M+K);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << std::endl;

    runTest("matrix_transpose_naive", as.data());
    runTest("matrix_transpose_local_bad_banks", as.data());
    runTest("matrix_transpose_local_good_banks", as.data());

    return 0;
}
