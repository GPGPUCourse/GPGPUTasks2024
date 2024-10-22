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

constexpr int ilog2(int x, int acc = 0) {
    return x <= 1 ? acc : ilog2(x >> 1, acc + 1);
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
        {
            ocl::Kernel prefix_sum1(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum1");
            prefix_sum1.compile();

            std::vector<unsigned int> res(n);
            gpu::gpu_mem_32u as_gpu;
            gpu::gpu_mem_32u bs_gpu;
            as_gpu.resizeN(n);
            bs_gpu.resizeN(n);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu.writeN(as.data(), n);
                t.restart();
                for (int logStride = 0; (1 << logStride) < n; ++logStride) {
                    prefix_sum1.exec(gpu::WorkSize(128, n), as_gpu, bs_gpu, 1 << logStride, n);
                    std::swap(as_gpu, bs_gpu);
                }
                t.nextLap();
            }

            as_gpu.readN(res.data(), n);

            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

            for (int i = 0; i < n; ++i) {
                EXPECT_THE_SAME(cpu_reference[i], res[i], "GPU result should be consistent!");
            }
        }

// work-efficient prefix sum
        {
            ocl::Kernel prefix_sum2(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum2");
            prefix_sum2.compile();

            std::vector<unsigned int> res(n);
            gpu::gpu_mem_32u as_gpu;
            as_gpu.resizeN(n);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu.writeN(as.data(), n);
                t.restart();
                int lg = ilog2(n - 1) + 1;
                for (int logStride = 0; (1 << logStride) < n; ++logStride) {
                    prefix_sum2.exec(gpu::WorkSize(128, n >> (logStride + 1)), as_gpu, 0, 1 << logStride, n);
                }
                for (int logStride = lg - 2; logStride >= 0; --logStride) {
                    prefix_sum2.exec(gpu::WorkSize(128, (n - (1 << logStride)) >> (logStride + 1)), as_gpu, (1 << logStride), 1 << (logStride), n);

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
	}

    // Сравнение реализаций:
    // Обе реализации уступают вычислениям на процессоре с -O3 (по крайней мере на моём устройстве).
    // 2ая реализация начинает обгонять первую почти сразу же на CPU и в районе n=1048576 на GPU.
    // 2ая реализация делает в 2 раза больше запусков кренелов с меньшим WorkSize и очень нелокально обращается к памяти,
    // из-за чего накладные расходы в ней значительно больше, чем в 1ой реализации
}
