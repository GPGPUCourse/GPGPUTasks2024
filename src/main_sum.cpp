#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

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

// I want std::string_view..
void runKernel(int benchmarkingIters, unsigned int reference_sum, unsigned int n, std::vector<unsigned int> &as, gpu::WorkSize size, const char* name) {
    gpu::gpu_mem_32u gpuAS, gpuSum;
    gpuAS.resizeN(n);
    gpuSum.resizeN(1);
    gpuAS.writeN(as.data(), n);
    ocl::Kernel kernel(sum_kernel, sum_kernel_length, name);
    kernel.compile();

    timer t;
    for (int i = 0; i < benchmarkingIters; ++i) {
        unsigned sum = 0;
        gpuSum.writeN(&sum, 1);
        kernel.exec(size, gpuAS, n, gpuSum);
        gpuSum.readN(&sum, 1);
        t.nextLap();
        EXPECT_THE_SAME(reference_sum, sum, "OpenCL Results must be consistent");
    }

    std::cout << "GPU kernel '" << name << "' : " << t.lapAvg() << "+-" << t.lapStd() << "s\n"
    << "  Time: " << (n / 1000. / 1000.) / t.lapAvg() << " millions/s\n";
}


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

//    {
//        timer t;
//        for (int iter = 0; iter < benchmarkingIters; ++iter) {
//            unsigned int sum = 0;
//            for (int i = 0; i < n; ++i) {
//                sum += as[i];
//            }
//            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
//            t.nextLap();
//        }
//        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
//        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
//    }

//    {
//        timer t;
//        for (int iter = 0; iter < benchmarkingIters; ++iter) {
//            unsigned int sum = 0;
//            #pragma omp parallel for reduction(+:sum)
//            for (int i = 0; i < n; ++i) {
//                sum += as[i];
//            }
//            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
//            t.nextLap();
//        }
//        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
//        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
//    }

    {
        // TODO: implement on OpenCL
         gpu::Device device = gpu::chooseGPUDevice(argc, argv);
         gpu::Context context;
         context.init(device.device_id_opencl);
         context.activate();

         {
             runKernel(benchmarkingIters, reference_sum, n, as, gpu::WorkSize(128, n), "sum_global_atomic");
             runKernel(benchmarkingIters, reference_sum, n, as, gpu::WorkSize(128, n / 64), "sum_loop");
             runKernel(benchmarkingIters, reference_sum, n, as, gpu::WorkSize(128, n / 64), "sum_loop_coalesced");
             runKernel(benchmarkingIters, reference_sum, n, as, gpu::WorkSize(128, n), "sum_local_mem");
             runKernel(benchmarkingIters, reference_sum, n, as, gpu::WorkSize(128, n), "sum_tree");
         }
    }
}

