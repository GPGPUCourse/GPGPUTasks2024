#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "cl/sum_cl.h" 

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

void kernel_running(const std::string &kernel_name, int benchmarking_iters, gpu::gpu_mem_32u &gpu_ar, unsigned int n, unsigned int w, unsigned int g, unsigned int reference_sum) {
    ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernel_name);
    gpu::gpu_mem_32u g_sum;
    g_sum.resizeN(1);
    timer t;

    for (int iter = 0; iter < benchmarking_iters; ++iter) {
        unsigned int sum = 0;
        g_sum.writeN(&sum, 1);
        kernel.exec(gpu::WorkSize(w, g), gpu_ar, g_sum, n);
        g_sum.readN(&sum, 1);
        EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
        t.nextLap();
    }
    std::cout << std::endl << "Results:" << std::endl;
    std::cout << "GPU " + kernel_name + ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "GPU " + kernel_name + ": " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    
}

int main(int argc, char **argv) {
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100 * 1000 * 1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int)r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
#pragma omp parallel for reduction(+ : sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u gpu_ar;
        gpu_ar.resizeN(n);
        gpu_ar.writeN(as.data(), n);

        kernel_running("sum_1", benchmarkingIters, gpu_ar, n, 128, n, reference_sum);
        kernel_running("sum_2", benchmarkingIters, gpu_ar, n, 128, n / 64, reference_sum);
        kernel_running("sum_3", benchmarkingIters, gpu_ar, n, 128, n / 64, reference_sum);
        kernel_running("sum_4", benchmarkingIters, gpu_ar, n, 128, n, reference_sum);
        kernel_running("sum_5", benchmarkingIters, gpu_ar, n, 128, n, reference_sum);
    }

    return 0;
}
