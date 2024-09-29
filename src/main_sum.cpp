#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include "cl/sum_cl.h"
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <list>
#include <string>

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

struct kernel_info {
    std::string name;
    gpu::WorkSize work_size;
};

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
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u buffer;
        gpu::gpu_mem_32u result;

        buffer.resizeN(n);
        result.resizeN(1);

        unsigned int work_group_size = 128;
        unsigned int global_work_size = (n + work_group_size - 1) / work_group_size * work_group_size;
        unsigned int n_work_groups = global_work_size / work_group_size;

        buffer.writeN(as.data(), n);
        const unsigned int init = 0;

        std::list<kernel_info> kernels = {
            {"atomic_sum", gpu::WorkSize(work_group_size, global_work_size)},
            {"loop_sum", gpu::WorkSize(work_group_size, global_work_size / 32)},
        };

        for(const auto& info : kernels) {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, info.name.data());
            kernel.compile();

            timer t;
            unsigned int sum = 0;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                result.writeN(&init, 1);
                kernel.exec(info.work_size, buffer, result, n);
                t.nextLap();
            }
            result.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            std::cout << "GPU " << info.name << " : " <<  t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU " << info.name << " : " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl << std::endl;
        }
    }
}
