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

unsigned int get_near_upper_2_pow(int k) {
    --k;
    int length = 1;
    while ((k >> length) > 0) {
        k = k | (k >> length);
        ++length;
    }
    return k + 1;
}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
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

// prefix sum
#if 0
        {
            std::vector<unsigned int> res(n);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                // TODO
                t.restart();
                // TODO
                t.nextLap();
            }

            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

            for (int i = 0; i < n; ++i) {
                EXPECT_THE_SAME(cpu_reference[i], res[i], "GPU result should be consistent!");
            }
        }
#endif

// work-efficient prefix sum
#if 1
        {
            gpu::gpu_mem_32u as_gpu;
            std::vector<unsigned int> res(n);
            unsigned int n_two_pow = get_near_upper_2_pow(n);
            as_gpu.resizeN(n_two_pow);

            ocl::Kernel prefix_sum(prefix_sum_kernel, prefix_sum_kernel_length, "work_efficient_sum");
            prefix_sum.compile();
            ocl::Kernel fill_with_zeros(prefix_sum_kernel, prefix_sum_kernel_length, "fill_with_zeros");
            fill_with_zeros.compile();
            ocl::Kernel refresh(prefix_sum_kernel, prefix_sum_kernel_length, "refresh");
            refresh.compile();

            const unsigned int workGroupSize = 128;
            const unsigned int global_fill_work_size = (n_two_pow - n + workGroupSize - 1) / workGroupSize * workGroupSize;
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu.writeN(as.data(), n);
                if (global_fill_work_size != 0) {
                    fill_with_zeros.exec(gpu::WorkSize(workGroupSize, global_fill_work_size), as_gpu, n, n_two_pow);
                }
                t.restart();
                for (unsigned int block_size = 2; block_size <= n_two_pow; block_size *= 2) {
                    const unsigned int global_work_size = (n_two_pow / block_size + workGroupSize - 1) / workGroupSize * workGroupSize;
                    prefix_sum.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, n_two_pow, block_size);
                }
                for (unsigned int block_size = n_two_pow / 2; block_size >= 2; block_size /= 2) {
                    const unsigned int global_work_size = (n_two_pow / block_size - 1 + workGroupSize - 1) / workGroupSize * workGroupSize;
                    refresh.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, n_two_pow, block_size);
                }
                t.nextLap();
            }
            as_gpu.readN(res.data(), n);
            std::cout << "GPU [work-efficient]: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU [work-efficient]: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

            for (int i = 0; i < n; ++i) {
                EXPECT_THE_SAME(cpu_reference[i], res[i], "GPU result should be consistent!");
            }
        }
#endif
	}
}
