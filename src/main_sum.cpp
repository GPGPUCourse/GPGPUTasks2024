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

void run_kernel_benchmark(
    unsigned int n, const std::vector<unsigned int> &as,
    unsigned int reference_sum, int benchmarkingIters,
    gpu::Device device, ocl::Kernel kernel, const std::string kernel_name,
    unsigned int work_group_size
) {
    const unsigned int global_work_size = (n + work_group_size - 1) / work_group_size * work_group_size;

    std::cout << std::endl;
    std::cout << "Global work size is " << global_work_size << std::endl;

    kernel.compile(false);
    {
        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            gpu::gpu_mem_32u sum_gpu;
            sum_gpu.resizeN(1);
            sum_gpu.writeN(&sum, 1);

            kernel.exec(
                gpu::WorkSize(work_group_size, global_work_size),
                as_gpu, sum_gpu, n
            );
            sum_gpu.readN(&sum, 1);

            EXPECT_THE_SAME(reference_sum, sum, "GPU " + kernel_name + " result should be consistent!");

            t.nextLap();
        }

        std::cout << "GPU " + kernel_name + ":     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU " + kernel_name + ":     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
}


int main(int argc, char **argv)
{
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

        std::cout << std::endl;
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

        std::cout << std::endl;
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // TODO: implement on OpenCL

        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        const unsigned int values_per_workitem = 32;
        const unsigned int work_group_size = 64;

        ocl::Kernel global_atomic(sum_kernel, sum_kernel_length, "sum_gpu_1");
        run_kernel_benchmark(
            n, as, reference_sum,
            benchmarkingIters, device, global_atomic, "global_atomic",
            work_group_size
        );

        ocl::Kernel loop_sum(sum_kernel, sum_kernel_length, "sum_gpu_2");
        run_kernel_benchmark(
            n, as, reference_sum,
            benchmarkingIters, device, loop_sum, "loop_sum",
            16 * values_per_workitem
        );

        ocl::Kernel coalesced_loop_sum(sum_kernel, sum_kernel_length, "sum_gpu_3");
        run_kernel_benchmark(
            n, as, reference_sum,
            benchmarkingIters, device, coalesced_loop_sum, "coalesced_loop_sum",
            16 * values_per_workitem
        );

        ocl::Kernel local_memory_sum(sum_kernel, sum_kernel_length, "sum_gpu_4");
        run_kernel_benchmark(
            n, as, reference_sum,
            benchmarkingIters, device, local_memory_sum, "local_memory_sum",
            work_group_size
        );

        ocl::Kernel tree_sum(sum_kernel, sum_kernel_length, "sum_gpu_5");
        run_kernel_benchmark(
            n, as, reference_sum,
            benchmarkingIters, device, tree_sum, "tree_sum",
            work_group_size
        );
    }
}
