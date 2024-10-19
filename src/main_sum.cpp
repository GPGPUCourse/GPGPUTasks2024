#include "libgpu/context.h"
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "sum_kernel_runner.h"
#include "stats_decorator.h"

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

         StatsGenerator stats_generator(benchmarkingIters);

         const unsigned int wg_size = 128;
         const unsigned int values_per_work_item = 32;

         SumKernelRunner global_runner(device, context, "sum_global_add", n, n, wg_size);
         stats_generator.runBenchmark(n, global_runner);

         SumKernelRunner loop_runner(device, context, "sum_loop", (n + values_per_work_item - 1) / values_per_work_item, n, wg_size);
         stats_generator.runBenchmark(n, loop_runner);

         SumKernelRunner coalesced_runner(device, context, "coalesced", (n + values_per_work_item - 1) / values_per_work_item, n, wg_size);
         stats_generator.runBenchmark(n, coalesced_runner);

         SumKernelRunner local_mem_runner(device, context, "local_memory", n, n, wg_size);
         stats_generator.runBenchmark(n, local_mem_runner);

         SumKernelRunner tree_runner(device, context, "tree_sum", n, n, wg_size);
         stats_generator.runBenchmark(n, tree_runner);
    }
}
