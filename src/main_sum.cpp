#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <numeric>
#include "libgpu/context.h"

#include "cl/sum_cl.h"
#include "libgpu/shared_device_buffer.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

class GPUKernelRunner {
public:
    GPUKernelRunner(std::vector<unsigned int> &t_arr, const int t_benchmarking_iters) : arr(t_arr),
                                                                                        benchmarking_iters(
                                                                                                t_benchmarking_iters) {
        n = t_arr.size();
        reference_sum = 0;
        std::for_each(t_arr.begin(), t_arr.end(), [&](int elem) {
            reference_sum += elem;
        });
    }

    void run(const std::string &kernel_name, gpu::WorkSize work_size) {
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernel_name);
        bool printLog = false;
        kernel.compile(printLog);

        gpu::gpu_mem_32u arr_gpu;
        gpu::gpu_mem_32u sum_gpu;
        arr_gpu.resizeN(n);
        sum_gpu.resizeN(1);

        arr_gpu.writeN(arr.data(), n);

        timer t;

        for (int i = 0; i < benchmarking_iters; ++i) {
            unsigned int sum = 0;
            sum_gpu.writeN(&sum, 1);
            kernel.exec(work_size,
                        arr_gpu,
                        sum_gpu,
                        n);
            sum_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU " + kernel_name + " result should be consistent!");
            t.nextLap();
        }

        std::cout << "GPU " + kernel_name + ":     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU " + kernel_name + ":     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s"
                  << std::endl;
    }

private:
    unsigned int n;
    int benchmarking_iters = 10;
    unsigned int reference_sum = 0;
    std::vector<unsigned int> &arr;
};


int main(int argc, char **argv) {
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100 * 1000 * 1000;
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
        std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
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
        std::cout << "CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::Context context;
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    context.init(device.device_id_opencl);
    context.activate();

    auto kernel_runner = GPUKernelRunner(as, benchmarkingIters);
    {
        kernel_runner.run("sum_atomic", gpu::WorkSize(128, n));
        kernel_runner.run("sum_cycle", gpu::WorkSize(128, n / 64));
        kernel_runner.run("sum_cycle_coalesced", gpu::WorkSize(128, n / 64));
        kernel_runner.run("sum_one_main_thread", gpu::WorkSize(128, n));
        kernel_runner.run("sum_tree", gpu::WorkSize(128, n));
    }
}
