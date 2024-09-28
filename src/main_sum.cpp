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

class KernelRunner
{
public:
    KernelRunner(std::vector<unsigned int> &as, unsigned int n, unsigned int reference_sum, int benchmarkingIters)
        : as(as), n(n), reference_sum(reference_sum), benchmarkingIters(benchmarkingIters) {
    }

    void run(const char *name) {
        timer t;
        gpu::gpu_mem_32u as_gpu;
        gpu::gpu_mem_32u sum_gpu;
        as_gpu.resizeN(n);
        sum_gpu.resizeN(1);
        as_gpu.writeN(as.data(), n);
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, name);
        kernel.compile();
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            sum_gpu.writeN(&sum, 1);
            kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, n), as_gpu, n, sum_gpu);
            sum_gpu.readN(&sum, 1);
            t.nextLap();
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
        }
        std::cout << "GPU " << name << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << '\n';
        std::cout << "GPU " << name << ": " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

private:
    std::vector<unsigned int> &as;
    unsigned int n;
    unsigned int reference_sum;
    unsigned int benchmarkingIters;
    static const unsigned int WORKGROUP_SIZE = 128;
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
        // TODO: implement on OpenCL
        // gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        KernelRunner runner{as, n, reference_sum, benchmarkingIters};
        runner.run("sum1");
        runner.run("sum2");
        runner.run("sum3");
        runner.run("sum4");
        runner.run("sum5");
    }
}
