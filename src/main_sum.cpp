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

#define DIV_UP(a, b) ((a) + (b) - 1) / (b)

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

        gpu::gpu_mem_32u device_as;
        device_as.resizeN(n);
        device_as.writeN(as.data(), n);

        gpu::gpu_mem_32u device_sum;
        device_sum.resizeN(1);

        auto benchmark_kernel = [&] (const char *entry_point_name, unsigned int work_space_size)
        {
            std::cout << std::endl << entry_point_name << ':' << std::endl;

            ocl::Kernel kernel(sum_kernel, sum_kernel_length, entry_point_name);
            kernel.compile(false);

            auto work_size = gpu::WorkSize(64, work_space_size);

            timer t;
            for (int i = 0; i < benchmarkingIters; i++) {
                unsigned int zero = 0;
                device_sum.writeN(&zero, 1);
                kernel.exec(work_size, device_as, n, device_sum);

                unsigned int sum = 0;
                device_sum.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "CPU and GPU results must be the same.");

                t.nextLap();
            }

            std::cout << "GPU OCL: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU OCL: " << n / (1000000.0 * t.lapAvg()) << " millions/s" << std::endl;
        };

        benchmark_kernel("sum_global_atomic", DIV_UP(n, 64) * 64);
        benchmark_kernel("sum_global_atomic_with_loop",           DIV_UP(DIV_UP(n, 64), 64) * 64);
        benchmark_kernel("sum_global_atomic_with_loop_coalesced", DIV_UP(DIV_UP(n, 64), 64) * 64);
        benchmark_kernel("sum_local_memory_with_main_thread", DIV_UP(n, 64) * 64);
        benchmark_kernel("sum_tree", DIV_UP(n, 64) * 64);
    }

    //
    // Сразу отметим, что даже не самая производительная видеокарта справляется с этой задачей в десятки раз быстрее центрального процессора.
    //
    // Первая версия самая медленная, что ожидаемо, так как мы платим за глобальную синхронизацию.
    //
    // Вторая версия чуть лучше, видимо, повлияло то, что мы в 64 раза реже обращаемся к atomic_add.
    //
    // Третья версия самая производительная, что подтверждает сведения с лекции о важности coalesced memory access.
    //
    // Четвёртая версия ещё медленнее, чем вторая. Скорее всего, это вызвано тем, что потоки совершают слишком мало полезной работы перед тем как
    // начинают ждать барьер, и тем, что даже очень хорошее расположение элементов в памяти (здесь buffer даже может поместиться в L1 cache) не
    // компенсирует ожидание сложения всех чисел в локальном буффере главным потоком, пока все остальные простаивают.
    //
    // Пятая версия работает так же как плохо, как и первая. Вероятно, из-за того, что сложение -- слишком простая операция и затраты на барьеры
    // компенсируют все преимущества по расположению элементов в памяти, которые мы могли бы здесь получить. Возможно, если бы вместо сложения
    // мы выполняли бы более сложную операцию, то вариант с деревом оказался бы быстрее первого варианта.
    //
}

