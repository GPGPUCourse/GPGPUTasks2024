#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/prefix_sum_cl.h"


const int benchmarkingIters = 10;
const int benchmarkingItersCPU = 10;
const unsigned int max_n = (1 << 24);

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
    unsigned int local_size = 128;
    context.init(device.device_id_opencl);
    context.activate();
    for (unsigned int n = 4096; n <= max_n; n *= 4) {
        std::cout << "______________________________________________" << std::endl;
        unsigned int values_range = std::min<unsigned int>(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << 0 << "; " << values_range << "]" << std::endl;

        std::vector<unsigned int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = r.next(0, values_range);
        }

        const std::vector<unsigned int> cpu_reference = computeCPU(as);

        // work-efficient prefix sum
        {
            std::vector<unsigned int> res(n);
            gpu::gpu_mem_32u as_gpu;
            as_gpu.resizeN(n);
            ocl::Kernel upsweep(prefix_sum_kernel, prefix_sum_kernel_length, "upsweep");
            ocl::Kernel downsweep(prefix_sum_kernel, prefix_sum_kernel_length, "downsweep");
            ocl::Kernel set_zero(prefix_sum_kernel, prefix_sum_kernel_length, "set_zero");
            upsweep.compile();
            downsweep.compile();
            set_zero.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu.writeN(as.data(), n);
                t.restart();

                unsigned int logn = (int)ceil(log2(n));
                for (int i = 0; i < logn; i++) {
                    unsigned int shift = 1 << (i + 1);
                    unsigned int global_size = (n - 1) / shift + 1;
                    upsweep.exec(gpu::WorkSize(local_size, global_size), as_gpu, shift, n);
                }

                set_zero.exec(gpu::WorkSize(1, 1), as_gpu, n);

                for (int i = logn - 1; i >= 0; i--) {
                    unsigned int shift = 1 << (i + 1);
                    unsigned int global_size = (n - 1) / shift + 1;
                    downsweep.exec(gpu::WorkSize(local_size, global_size), as_gpu, shift, n);
                }
                t.nextLap();
            }
            as_gpu.readN(res.data(), n - 1, 1);
            res[n - 1] = res[n - 2] + as[n - 1];

            std::cout << "GPU [work-efficient]: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU [work-efficient]: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

            for (int i = 0; i < n; ++i) {
                EXPECT_THE_SAME(cpu_reference[i], res[i], "GPU result should be consistent!");
            }
        }
    }
}
