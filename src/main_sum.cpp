#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>

#include "cl/sum_cl.h"
#include <libgpu/shared_device_buffer.h>


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
    {
        // TODO: implement on OpenCL
        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);
        gpu::gpu_mem_32u res_gpu;
        res_gpu.resizeN(1);
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_1");
        kernel.compile();
        unsigned int workGroupSize = 128;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        as_gpu.writeN(as.data(), n);
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            uint32_t answer = 0;
            res_gpu.writeN(&answer, 1);
            kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu, res_gpu, n);
            res_gpu.readN(&answer, 1);
            EXPECT_THE_SAME(reference_sum, answer, "GPU OpenCL1 result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU 1 base: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU 1 base: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
    {
        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);
        gpu::gpu_mem_32u res_gpu;
        res_gpu.resizeN(1);

        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_2");
        kernel.compile();

        unsigned int workGroupSize = 128;
        unsigned int values_per_workitem = 64;
        unsigned int global_work_size = (n / values_per_workitem + workGroupSize - 1) / workGroupSize * workGroupSize;
        as_gpu.writeN(as.data(), n);
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            uint32_t answer = 0;
            res_gpu.writeN(&answer, 1);
            kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu, res_gpu, n);
            res_gpu.readN(&answer, 1);
            EXPECT_THE_SAME(reference_sum, answer, "GPU OpenCL1 result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU 2 cycle: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU 2 cycle: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
    {
        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);
        gpu::gpu_mem_32u res_gpu;
        res_gpu.resizeN(1);

        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_3");
        kernel.compile();

        unsigned int workGroupSize = 128;
        unsigned int values_per_workitem = 64;
        unsigned int global_work_size = (n / values_per_workitem + workGroupSize - 1) / workGroupSize * workGroupSize;
        as_gpu.writeN(as.data(), n);
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            uint32_t answer = 0;
            res_gpu.writeN(&answer, 1);
            kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu, res_gpu, n);
            res_gpu.readN(&answer, 1);
            EXPECT_THE_SAME(reference_sum, answer, "GPU OpenCL1 result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU 3 coalesced: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU 3 coalesced: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);
        gpu::gpu_mem_32u res_gpu;
        res_gpu.resizeN(1);

        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_4");
        kernel.compile();

        unsigned int workGroupSize = 128;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        as_gpu.writeN(as.data(), n);
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            uint32_t answer = 0;
            res_gpu.writeN(&answer, 1);
            kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu, res_gpu, n);
            res_gpu.readN(&answer, 1);
            EXPECT_THE_SAME(reference_sum, answer, "GPU OpenCL1 result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU 4 main thread: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU 4 main thread: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);
        gpu::gpu_mem_32u res_gpu;

        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_5");
        kernel.compile();

        unsigned int workGroupSize = 128;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        res_gpu.resizeN(global_work_size / workGroupSize);
        gpu::gpu_mem_32u tmp_gpu;
        tmp_gpu.resizeN(global_work_size / workGroupSize);
        timer t;
        for (int iter = 0; iter < 1; ++iter) {
            uint32_t answer = 0;
            int m = n;
            as_gpu.writeN(as.data(), n);
            for (int size = global_work_size; m > 1; size /= workGroupSize, std::swap(tmp_gpu, res_gpu), m = size, size = (size + workGroupSize - 1) / workGroupSize * workGroupSize) {
                if (size == global_work_size)
                    kernel.exec(gpu::WorkSize(workGroupSize, size),
                        as_gpu, res_gpu, m);
                else
                    kernel.exec(gpu::WorkSize(workGroupSize, size),
                        tmp_gpu, res_gpu, m);
            }
            std::swap(tmp_gpu, res_gpu);
            res_gpu.readN(&answer, 1);
            EXPECT_THE_SAME(reference_sum, answer, "GPU OpenCL1 result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU 5 tree: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU 5 tree: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
}
