#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>

const int benchmarkingIters = 10;
const int benchmarkingItersCPU = 1;
const unsigned int n = 32 * 1024 * 1024;

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

std::vector<unsigned int> computeCPU(const std::vector<unsigned int> &as)
{
    std::vector<unsigned int> cpu_sorted;

    timer t;
    for (int iter = 0; iter < benchmarkingItersCPU; ++iter) {
        cpu_sorted = as;
        t.restart();
        std::sort(cpu_sorted.begin(), cpu_sorted.end());
        t.nextLap();
    }
    std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

    return cpu_sorted;
}

template<typename T>
std::ostream &operator<<(std::ostream &o, const std::vector<T> &v) {
    for (int i = 0; i < v.size(); ++i) {
        o << v[i] << (i == v.size() - 1 ? "" : i % 8 == 7 ? " | " : i % 4 == 3 ? " . " : " ");
    }
    return o;
}

template<typename T>
std::ostream &operator<<(std::ostream &o, const gpu::shared_device_buffer_typed<T> &v) {
    std::vector<T> buffer(v.number());
    v.readN(buffer.data(), v.number());
    return o << "gpu[" << buffer << "]";
    // return o << "gpu[]";
}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    const std::vector<unsigned int> cpu_reference = computeCPU(as);


    {
        timer t;

        const int SHIFT = 4; // 4
        const int NUM_BUCKETS = 1 << SHIFT;
        const int WG_SIZE = 128; // 128
        const int WG2_SIZE = 16; // 16

        gpu::gpu_mem_32u as_gpu;
        gpu::gpu_mem_32u counts_gpu;
        gpu::gpu_mem_32u counts_t_gpu;
        gpu::gpu_mem_32u bs_gpu;

        ocl::Kernel radix_step_phase1(radix_kernel, radix_kernel_length, "radix_step_phase1");
        ocl::Kernel transpose(radix_kernel, radix_kernel_length, "transpose");
        ocl::Kernel prefix_sum_step(radix_kernel, radix_kernel_length, "prefix_sum_step");
        ocl::Kernel radix_step_phase2(radix_kernel, radix_kernel_length, "radix_step_phase2");
        radix_step_phase1.compile();
        transpose.compile();
        prefix_sum_step.compile();
        radix_step_phase2.compile();

        int num_chunks = n / WG_SIZE;
        int counts_size = num_chunks * NUM_BUCKETS;

        as_gpu.resizeN(n);
        bs_gpu.resizeN(n);
        counts_gpu.resizeN(counts_size);
        counts_t_gpu.resizeN(counts_size);

        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();
            
            // std::cout << "Initial array: " << as_gpu << "\n";

            for (int shift = 0; shift < 32; shift += SHIFT) {
                radix_step_phase1.exec(gpu::WorkSize(WG_SIZE, n), as_gpu, counts_gpu, n, shift);

                // std::cout << "Pre-sorted: " << as_gpu << "\n";
                // std::cout << "Counts: " << counts_gpu << "\n";

                transpose.exec(gpu::WorkSize(WG2_SIZE, WG2_SIZE, NUM_BUCKETS, num_chunks), counts_gpu, counts_t_gpu, num_chunks, NUM_BUCKETS);

                // std::cout << "Transposed counts: " << counts_t_gpu << "\n";

                int i;
                for (i = 1; (1 << i) <= counts_size; ++i) {
                    int work = counts_size >> i;
                    int stride_log = i;
                    int offset = (1 << i) - 1;
                    int add_offset = -(1 << (i - 1));
                    // std::printf("i=%d work=%d stride=%d offset=%d add=%d\n", i, work, 1 << stride_log, offset, add_offset);
                    prefix_sum_step.exec(gpu::WorkSize(WG_SIZE, work), counts_t_gpu, stride_log, offset, add_offset, counts_size, counts_size - 1);
                }
                for (i -= 2; i > 0; --i) {
                    int work = (counts_size >> i) - 1;
                    int stride_log = i;
                    int offset = (3 << (i - 1)) - 1;
                    int add_offset = -(1 << (i - 1));
                    // std::printf("i=%d work=%d stride=%d offset=%d add=%d\n", i, work, 1 << stride_log, offset, add_offset);
                    prefix_sum_step.exec(gpu::WorkSize(WG_SIZE, work), counts_t_gpu, stride_log, offset, add_offset, counts_size, counts_size - 1);
                }

                // std::cout << "Prefix sum: " << counts_t_gpu << "\n";

                for (i = 1; (1 << i) <= NUM_BUCKETS; ++i) {
                    int work = counts_size >> i;
                    int stride_log = i;
                    int offset = (1 << i) - 1;
                    int add_offset = -(1 << (i - 1));
                    prefix_sum_step.exec(gpu::WorkSize(WG_SIZE, work), counts_gpu, stride_log, offset, add_offset, counts_size, NUM_BUCKETS - 1);
                }
                for (i -= 2; i > 0; --i) {
                    int work = (counts_size >> i) - 1;
                    int stride_log = i;
                    int offset = (3 << (i - 1)) - 1;
                    int add_offset = -(1 << (i - 1));
                    prefix_sum_step.exec(gpu::WorkSize(WG_SIZE, work), counts_gpu, stride_log, offset, add_offset, counts_size, NUM_BUCKETS - 1);
                }

                // std::cout << "Local sum: " << counts_gpu << "\n";

                radix_step_phase2.exec(gpu::WorkSize(WG_SIZE, n), as_gpu, bs_gpu, counts_gpu, counts_t_gpu, n, shift);

                // std::cout << "Result: " << bs_gpu << "\n";

                std::swap(as_gpu, bs_gpu);

                // std::cout << "\n";
            }

            t.nextLap();
        }
        t.stop();

        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // std::cout << "Expected: " << cpu_reference << "\n";

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_reference[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}
