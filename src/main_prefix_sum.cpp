#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/prefix_sum_cl.h"


const int benchmarkingIters = 10;
const int benchmarkingItersCPU = 10;

const unsigned int min_n = (4096);
const unsigned int max_n = (1 << 23);

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
	if (a != b) {
		std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
		throw std::runtime_error(message);
	}
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

std::vector<unsigned int> computeCPU(const std::vector<unsigned int> &as)
{
    const unsigned int n = as.size();

    std::vector<unsigned int> bs(n);
    timer t;
    for (int iter = 0; iter < benchmarkingItersCPU; ++iter) {
        for (int i = 0; i < n; ++i) {
            bs[i] = as[i];
            if (i) {
                bs[i] += bs[i-1];
            }
        }
        t.nextLap();
    }

    std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

    return bs;
}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();


    for (int seed = 0; seed < 1; seed++) {
        for (unsigned int n = min_n; n <= max_n; n *= 4) {
            std::cout << "______________________________________________" << std::endl;
            unsigned int values_range = std::min<unsigned int>(1023, std::numeric_limits<int>::max() / n);
            std::cout << "n=" << n << " values in range: [" << 0 << "; " << values_range << "]" << std::endl;

            std::vector<unsigned int> as(n, 0);
            FastRandom r(n);
            for (int i = 0; i < n; ++i) {
                as[i] = r.next(0, values_range);
            }

            const std::vector<unsigned int> cpu_reference = computeCPU(as);

// prefix sum
#if 1
            {
                gpu::gpu_mem_32u as_gpu, bs_gpu;
                as_gpu.resizeN(n); bs_gpu.resizeN(n);

                ocl::Kernel prefixSum(prefix_sum_kernel, prefix_sum_kernel_length, "global_prefix_sum");
                prefixSum.compile();
                std::vector<unsigned int> res(n);

                const unsigned int groupSize = std::min(128u, n);
                const unsigned int workSize = gpu::divup(n, groupSize) * groupSize;

                timer t;
                for (int iter = 0; iter < benchmarkingIters; ++iter) {
                    as_gpu.writeN(as.data(), n);
                    t.restart();

                    unsigned int step = 1;
                    while (step < n){
                        prefixSum.exec(gpu::WorkSize(groupSize, workSize), n, as_gpu, step, bs_gpu);
                        std::swap(as_gpu, bs_gpu);
                        step *= 2;
                    }

                    t.nextLap();
                }

                std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
                std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

                as_gpu.readN(res.data(), n);

                for (int i = 0; i < n; ++i) {
                    EXPECT_THE_SAME(cpu_reference[i], res[i], "GPU result should be consistent!");
                }

            }
#endif

// work-efficient prefix sum
#if 1
            {
                const unsigned int groupSize = std::min(128u, n);

                gpu::gpu_mem_32u as_gpu;
                as_gpu.resizeN(n);

                std::string defines = "-DGROUP_SIZE=" + std::to_string(groupSize);
                ocl::Kernel prefix_up(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_up", defines);
                prefix_up.compile();
                ocl::Kernel prefix_down(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_down", defines);
                prefix_down.compile();

                timer t;
                for (int iter = 0; iter < benchmarkingIters; ++iter) {
                    as_gpu.writeN(as.data(), n);

                    t.restart();

                    unsigned int global_step = 1;
                    while (global_step < n) {
                        const unsigned int workSize =
                                (n + 2 * groupSize * global_step - 1) / (2 * groupSize * global_step) * groupSize;
                        prefix_up.exec(
                                gpu::WorkSize(groupSize, workSize),
                                as_gpu, (int) n, (int) global_step
                        );
                        global_step *= 2 * groupSize;
                    }

                    global_step /= (2 * groupSize * 2 * groupSize);

                    while (global_step > 0) {
                        const unsigned int workSize =
                                (n + 2 * groupSize * global_step - 1) / (2 * groupSize * global_step) * groupSize * 2;
                        prefix_down.exec(
                                gpu::WorkSize(2 * groupSize, workSize),
                                as_gpu, (int) n, (int) global_step
                        );
                        global_step /= (2 * groupSize);
                    }

                    t.nextLap();
                }

                std::cout << "GPU [work-efficient]: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
                std::cout << "GPU [work-efficient]: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s"
                          << std::endl;

                std::vector<unsigned int> res(n);
                as_gpu.readN(res.data(), n);

                for (int i = 0; i < n; ++i) {
                    EXPECT_THE_SAME(cpu_reference[i], res[i], "GPU result should be consistent! index: " + std::to_string(i) + " seed: " + std::to_string(seed) + " gs: " + std::to_string(groupSize));
                }
            }
#endif
        }
    }
}
