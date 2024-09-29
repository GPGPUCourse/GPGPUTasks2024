#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
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

namespace {

constexpr int benchmarkingIters = 10;
constexpr unsigned int n = 100*1000*1000;

void executeKernel(const char* name,
                   const gpu::gpu_mem_32u& memory,
                   gpu::WorkSize ws,
                   unsigned int expected) {
    ocl::Kernel kernel(sum_kernel, sum_kernel_length, name);
    gpu::gpu_mem_32u result_gpu;
    result_gpu.resizeN(1);

    timer t;
    std::string errorMessage = std::string("GPU ") + name + " result should be consistent!";
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        unsigned int sum = 0;

        result_gpu.writeN(&sum, 1);
        kernel.exec(ws, memory, result_gpu, n);
        result_gpu.readN(&sum, 1);

        EXPECT_THE_SAME(expected, sum, errorMessage);

        t.nextLap();
    }

    std::cout << "GPU " << name << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "GPU " << name << ": " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
}

}  // namespace

int main(int argc, char **argv)
{
    unsigned int reference_sum = 0;
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

    {
        const unsigned int workGroupSize = 128u;
        const unsigned int valuesPerWorkItem = 64u;
        const unsigned int workItems = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;

        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u array;
        array.resizeN(n);
        array.writeN(as.data(), n);

        executeKernel("global_add_scan", array, gpu::WorkSize(workGroupSize, workItems), reference_sum);
        executeKernel("simple_cycle_scan", array, gpu::WorkSize(workGroupSize, workItems / valuesPerWorkItem), reference_sum);
        executeKernel("coalesced_cycle_scan", array, gpu::WorkSize(workGroupSize, workItems / valuesPerWorkItem), reference_sum);
        executeKernel("local_leader_synchronization_scan", array, gpu::WorkSize(workGroupSize, workItems), reference_sum);
        executeKernel("divide_and_conquer_scan", array, gpu::WorkSize(workGroupSize, workItems), reference_sum);
    }
}
