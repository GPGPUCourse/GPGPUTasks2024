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
            // std::cout << as[i] << " ";
		}
        // std::cout << std::endl;

        const std::vector<unsigned int> cpu_reference = computeCPU(as);

// prefix sum
#if 1
        {
            std::vector<unsigned int> res(n);
            ocl::Kernel prefix_sum_step2(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum_step2");
            prefix_sum_step2.compile();
            gpu::gpu_mem_32u a_gpu, b_gpu;
            a_gpu.resizeN(n);
            b_gpu.resizeN(n);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                a_gpu.writeN(as.data(), n);
                t.restart();

                for (int offset = 1; offset < n; offset *= 2) {
                    a_gpu.copyToN(b_gpu, n);
                    prefix_sum_step2.exec(
                        gpu::WorkSize(128, n - offset),
                        a_gpu,
                        b_gpu,
                        offset,
                        -offset,
                        n
                    );

                    // b_gpu.readN(res.data(), n);
                    // for (int i = 0; i < n; ++i) {
                    //     std::cout << res[i]  << " ";
                    // }
                    // std::cout << std::endl;

                    std::swap(a_gpu, b_gpu);
                }

                t.nextLap();
            }

            a_gpu.readN(res.data(), n);

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
            std::vector<unsigned int> res(n);
            ocl::Kernel prefix_sum_step(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum_step");
            prefix_sum_step.compile();
            gpu::gpu_mem_32u a_gpu;
            a_gpu.resizeN(n);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                a_gpu.writeN(as.data(), n);
                t.restart();

                int i;
                for (i = 1; (1 << i) <= n; ++i) {
                    int work = n >> i;
                    int stride_log = i;
                    int offset = (1 << i) - 1;
                    int add_offset = -(1 << (i - 1));
                    // std::printf("i=%d work=%d stride=%d offset=%d add=%d\n", i, work, 1 << stride_log, offset, add_offset);
                    prefix_sum_step.exec(gpu::WorkSize(128, work), a_gpu, stride_log, offset, add_offset, n);
                }
                for (i -= 2; i > 0; --i) {
                    int work = (n >> i) - 1;
                    int stride_log = i;
                    int offset = (3 << (i - 1)) - 1;
                    int add_offset = -(1 << (i - 1));
                    // std::printf("i=%d work=%d stride=%d offset=%d add=%d\n", i, work, 1 << stride_log, offset, add_offset);
                    prefix_sum_step.exec(gpu::WorkSize(128, work), a_gpu, stride_log, offset, add_offset, n);
                }

                t.nextLap();
            }

            a_gpu.readN(res.data(), n);

            std::cout << "GPU [work-efficient]: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU [work-efficient]: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

            for (int i = 0; i < n; ++i) {
                EXPECT_THE_SAME(cpu_reference[i], res[i], "GPU result should be consistent!");
            }
        }
#endif
	}
}
