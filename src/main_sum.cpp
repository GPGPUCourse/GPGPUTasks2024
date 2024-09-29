#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include "cl/sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
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
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    const auto run = [&](const std::string& kernel_name, gpu::WorkSize work_size){
        gpu::gpu_mem_32u as_gpu;
        gpu::gpu_mem_32u sum_gpu;

        as_gpu.resizeN(as.size());
        sum_gpu.resizeN(1);

        as_gpu.writeN(as.data(), as.size());

        ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernel_name);
        kernel.compile();

        const std::string platform_name = "GPU " + kernel_name;
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            sum_gpu.writeN(&sum, 1);
            kernel.exec(work_size, as_gpu, sum_gpu, n);
            sum_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, platform_name + " result should be consistent!");
            t.nextLap();
        }
        std::cout << platform_name << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << platform_name << ": " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    };

    run("sum_atomic", gpu::WorkSize(128, n));
    run("sum_loop_not_coalesced", gpu::WorkSize(128, n / 64));
    run("sum_loop_coalesced", gpu::WorkSize(128, n / 64));
    run("sum_main_thread", gpu::WorkSize(128, n));
    run("sum_tree", gpu::WorkSize(128, n));
}
