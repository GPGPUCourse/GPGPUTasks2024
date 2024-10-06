#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_multiplication_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>

const int benchmarkingIters = 10;
const int benchmarkingItersCPU = 1;
const unsigned int M = 1024;
const unsigned int K = 1024;
const unsigned int N = 1024;
const size_t gflops = ((size_t) M * K * N * 2) / (1000 * 1000 * 1000); // умножить на два, т.к. операция сложения и умножения

std::vector<float> computeCPU(const float *as, const float *bs)
{
    std::vector<float> cs(M*N, 0);

    timer t;
    for (int iter = 0; iter < benchmarkingItersCPU; ++iter) {
        for (int j = 0; j < M; ++j) {
            for (int i = 0; i < N; ++i) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += as[j * K + k] * bs[k * N + i];
                }
                cs[j * N + i] = sum;
            }
        }
        t.nextLap();
    }

    std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "CPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;

    return cs;
}

struct KernelConfig {
    std::string kernel_name;
    gpu::WorkSize work_size;
    std::string defines;
    std::string prefix;
};

KernelConfig makeNaiveConfig(unsigned int tile_size)
{
    std::string kernel_name = "matrix_multiplication_naive";
    gpu::WorkSize work_size(tile_size, tile_size, K, M);
    std::string defines;
    std::string prefix = "[naive, ts=" + std::to_string(tile_size) + "]";
    return KernelConfig{kernel_name, work_size, defines, prefix};
}

KernelConfig makeLocalConfig(unsigned int tile_size)
{
    std::string kernel_name = "matrix_multiplication_local";
    gpu::WorkSize work_size(tile_size, tile_size, K, M);
    std::string defines = "-DTILE_SIZE=" + std::to_string(tile_size);
    std::string prefix = "[local, ts=" + std::to_string(tile_size) + "]";
    return KernelConfig{kernel_name, work_size, defines, prefix};
}

KernelConfig makeLocalWPTConfig(unsigned int tile_size, unsigned int wpt)
{
    std::string kernel_name = "matrix_multiplication_local_wpt";
    gpu::WorkSize work_size(tile_size, tile_size / wpt, K, M / wpt);
    std::string defines = "-DTILE_SIZE=" + std::to_string(tile_size) + " -DWORK_PER_THREAD=" + std::to_string(wpt);
    std::string prefix = "[local wpt, ts=" + std::to_string(tile_size) + ", wpt=" + std::to_string(wpt) + "]";
    return KernelConfig{kernel_name, work_size, defines, prefix};
}

void runTest(const KernelConfig &config, const float *as, const float *bs, const float *cs_cpu_reference)
{
    gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;
    as_gpu.resizeN(M*K);
    bs_gpu.resizeN(K*N);
    cs_gpu.resizeN(M*N);

    as_gpu.writeN(as, M*K);
    bs_gpu.writeN(bs, K*N);

    ocl::Kernel matrix_multiplication_kernel(matrix_multiplication, matrix_multiplication_length, config.kernel_name, config.defines);
    matrix_multiplication_kernel.compile();

    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        matrix_multiplication_kernel.exec(config.work_size, as_gpu, bs_gpu, cs_gpu, M, K, N);
        t.nextLap();
    }

    std::cout << config.prefix << std::endl;
    std::cout << "    GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "    GPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;

    std::vector<float> cs(M*N, 0);
    cs_gpu.readN(cs.data(), M*N);

    // Проверяем корректность результатов
    double diff_sum = 0;
    for (int i = 0; i < M * N; ++i) {
        double a = cs[i];
        double b = cs_cpu_reference[i];
        if (a != 0.0 || b != 0.0) {
            double diff = fabs(a - b) / std::max(fabs(a), fabs(b));
            diff_sum += diff;
        }
    }

    double diff_avg = diff_sum / (M * N);
    std::cout <<"    Average difference: " << diff_avg * 100.0 << "%" << std::endl;
    if (diff_avg > 0.01) {
        throw std::runtime_error("Too big difference!");
    }
}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    std::vector<float> as(M*K, 0);
    std::vector<float> bs(K*N, 0);
    FastRandom r(M+K+N);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = r.nextf();
    }
    for (unsigned int i = 0; i < bs.size(); ++i) {
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << ", N=" << N << std::endl;

    const std::vector<float> cs_cpu_reference = computeCPU(as.data(), bs.data());

    runTest(makeNaiveConfig(4), as.data(), bs.data(), cs_cpu_reference.data());
    runTest(makeNaiveConfig(8), as.data(), bs.data(), cs_cpu_reference.data());
    runTest(makeNaiveConfig(16), as.data(), bs.data(), cs_cpu_reference.data());

    runTest(makeLocalConfig(4), as.data(), bs.data(), cs_cpu_reference.data());
    runTest(makeLocalConfig(8), as.data(), bs.data(), cs_cpu_reference.data());
    runTest(makeLocalConfig(16), as.data(), bs.data(), cs_cpu_reference.data());

    for (unsigned int tile_size : {4, 8, 16})
        for (unsigned int wpt : {2, 4, 8, 16})
            if (wpt <= tile_size)
                runTest(makeLocalWPTConfig(tile_size, wpt), as.data(), bs.data(), cs_cpu_reference.data());

    return 0;
}
