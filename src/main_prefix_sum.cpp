#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/prefix_sum_cl.h"


const int benchmarkingIters = 1;
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

    ocl::Kernel basic_prefix_sum_kernel(prefix_sum_kernel, prefix_sum_kernel_length, "basic_prefix_sum");
    ocl::Kernel work_efficient_prefix_sum_kernel(prefix_sum_kernel, prefix_sum_kernel_length, "work_efficient_prefix_sum");

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

	    gpu::gpu_mem_32u as_gpu;
	    as_gpu.resizeN(n);
	    gpu::gpu_mem_32u bs_gpu;
	    bs_gpu.resizeN(n);

	    {
	        gpu::WorkSize workSize(256, n);
	        std::vector<unsigned int> res(n);

	        timer t;
	        for (int i = 0; i < benchmarkingIters; ++i) {
	            as_gpu.writeN(as.data(), n);
	            t.restart();
	            for (unsigned int step = 1; step < n; step *= 2) {
	                basic_prefix_sum_kernel.exec(workSize, as_gpu, bs_gpu, step, n);
	                std::swap(as_gpu, bs_gpu);
	            }
	            t.nextLap();
	        }

	        std::cout << "GPU [non-work-efficient]: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
	        std::cout << "GPU [non-work-efficient]: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
	        as_gpu.readN(res.data(), n);

	        for (int i = 0; i < n; ++i) {
	            EXPECT_THE_SAME(res[i], cpu_reference[i], "GPU results should be equal to CPU results!");
	        }
	    }

        {
            std::vector<unsigned int> res(n);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu.writeN(as.data(), n);
                unsigned int last_step;
                t.restart();
                // наверное лучше workSize снаружи ограничивать, чем внутри кернела делать проверку и return
                // (workitem'ов же явно больше, чем потоков, мб так драйвер лучше работу распределит)
                for (unsigned int step = 1; step < n; step *= 2) {
                    work_efficient_prefix_sum_kernel.exec(gpu::WorkSize(256, n / step / 2), as_gpu, step, n, 0);
                    last_step = step;
                }
                for (unsigned int step = last_step / 2; step > 0; step /= 2) {
                    work_efficient_prefix_sum_kernel.exec(gpu::WorkSize(256, n / step / 2), as_gpu, step, n, 1);
                }
                t.nextLap();
            }

            std::cout << "GPU [work-efficient]: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU [work-efficient]: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
	        as_gpu.readN(res.data(), n);

            for (int i = 0; i < n; ++i) {
                EXPECT_THE_SAME(res[i], cpu_reference[i], "GPU result should be consistent!");
            }
        }
	}
}
