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

struct tester {
    tester(
        int argc, 
        char** argv, 
        std::vector<unsigned int>& as, 
        int benchmarkingIters, 
        unsigned int referenceSum, 
        unsigned int n) 
            : 
            as(as), 
            benchmarkingIters(benchmarkingIters), 
            referenceSum(referenceSum), 
            n(n),
            context()
    {
        device = gpu::chooseGPUDevice(argc, argv);
        context.init(device.device_id_opencl);
        context.activate();
        
        as_buf.resize(sizeof(unsigned int) * n);
        as_buf.write(as.data(), sizeof(unsigned int) * n);

        result_buf.resize(sizeof(unsigned int));
    }

    void execute_and_time(std::string kernel_name, int values_per_workitem = 1) {
        unsigned int zero = 0;

        ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernel_name);
        kernel.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;

            result_buf.write(&zero, sizeof(unsigned int));
            kernel.exec(gpu::WorkSize(128, n / values_per_workitem), result_buf, as_buf, n);
            result_buf.read(&sum, sizeof(unsigned int));
            
            EXPECT_THE_SAME(referenceSum, sum, kernel_name + " result should be consistent!");
            t.nextLap();
        }
        std::cout << kernel_name + ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << kernel_name + ": " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
    

private:
    const std::vector<unsigned int>& as;
    const int benchmarkingIters;
    const unsigned int referenceSum;
    const unsigned int n;
    gpu::Device device;
    gpu::Context context;
    gpu::gpu_mem_any as_buf;
    gpu::gpu_mem_any result_buf;
};

#define VALUES_PER_WORKITEM 32

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
        tester t(argc, argv, as, benchmarkingIters, reference_sum, n);
        t.execute_and_time("sum_atomic");
        t.execute_and_time("sum_for_loop", VALUES_PER_WORKITEM);
        t.execute_and_time("sum_for_loop_coalesced", VALUES_PER_WORKITEM);
        t.execute_and_time("sum_local_mem_single_thread");
        t.execute_and_time("sum_local_mem_tree");
    }
}
