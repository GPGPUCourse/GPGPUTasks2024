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

    {
        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u as_gpu, sum_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);
        sum_gpu.resizeN(1);

        unsigned int workGroupSize = 128;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        unsigned int VALUES_PER_WORKITEM = 64;
        unsigned int taskSize = (global_work_size + VALUES_PER_WORKITEM - 1) / VALUES_PER_WORKITEM;

        {
            ocl::Kernel sum_k(sum_kernel, sum_kernel_length, "sum_gpu_atomic");
            sum_k.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                sum_gpu.writeN(&sum, 1);
                sum_k.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu, sum_gpu, n);
                
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU (atomic):     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (atomic):     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel sum_k(sum_kernel, sum_kernel_length, "sum_gpu_atomic_loop");
            sum_k.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                sum_gpu.writeN(&sum, 1);
                sum_k.exec(gpu::WorkSize(workGroupSize, taskSize),
                        as_gpu, sum_gpu, n);
                
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU (atomic loop):     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (atomic loop):     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel sum_k(sum_kernel, sum_kernel_length, "sum_gpu_atomic_loop_coalesced");
            sum_k.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                sum_gpu.writeN(&sum, 1);
                sum_k.exec(gpu::WorkSize(workGroupSize, taskSize),
                        as_gpu, sum_gpu, n);
                
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU (atomic loop coalesced):     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (atomic loop coalesced):     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel sum_k(sum_kernel, sum_kernel_length, "sum_gpu_atomic_local_mem");
            sum_k.compile();
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                sum_gpu.writeN(&sum, 1);
                sum_k.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu, sum_gpu, n);
                
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU (atomic local mem):     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (atomic local mem):     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel sum_k(sum_kernel, sum_kernel_length, "sum_gpu_atomic_tree");
            sum_k.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                sum_gpu.writeN(&sum, 1);
                sum_k.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu, sum_gpu, n);
                
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU (atomic tree):     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (atomic tree):     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
