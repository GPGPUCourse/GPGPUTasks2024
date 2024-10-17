#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "cl/matrix_transpose_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>

const int benchmarkingIters = 100;
const unsigned int M = 4096;
const unsigned int K = 4096;

const int LOCAL_SIZE_X = 32;
const int LOCAL_SIZE_Y = 8; 

void runTest(const std::string &kernel_name, const float *as) {
    gpu::gpu_mem_32f as_gpu, as_t_gpu;
    as_gpu.resizeN(M * K);
    as_t_gpu.resizeN(K * M);

    as_gpu.writeN(as, M * K);

    ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, kernel_name);
    matrix_transpose_kernel.compile();

   

    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        gpu::WorkSize work_size(LOCAL_SIZE_X, LOCAL_SIZE_Y);
        matrix_transpose_kernel.exec(work_size, as_gpu, as_t_gpu, M, K);
        t.nextLap();
    }

    double avgTime = t.lapAvg();
    if (avgTime == 0) {
        std::cout << "Warning: Average time is zero. Check kernel execution time." << std::endl;
    } else {
        std::cout << "[" << kernel_name << "]" << std::endl;
        std::cout << "    GPU: " << avgTime << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "    GPU: " << M * K / 1000.0 / 1000.0 / avgTime << " millions/s" << std::endl;
    }



    std::vector<float> as_t(M * K, 0);
    as_t_gpu.readN(as_t.data(), M * K);

    std::cout << "Original Matrix (first 10x10):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            std::cout << as[i * K + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Transposed Matrix (first 10x10):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            std::cout << as_t[i * M + j] << " ";
        }
        std::cout << std::endl;
    }

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


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    std::vector<float> as(M * K, 0);
    FastRandom r(M + K);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << std::endl;

    // Вывод исходной матрицы
    //std::cout << "Original Matrix:" << std::endl;
    //for (unsigned int i = 0; i < M; ++i) {
    // for (unsigned int j = 0; j < K; ++j) {
    // std::cout << as[i * K + j] << " ";
    //}
    // std::cout << std::endl;
    // }

    runTest("matrix_transpose_naive", as.data());
    runTest("matrix_transpose_local_bad_banks", as.data());
    runTest("matrix_transpose_local_good_banks", as.data());

    return 0;
}