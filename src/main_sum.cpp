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

constexpr bool BUILD_LOG = false;

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

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
        std::cout << "CPU:                  " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:                  " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
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
        std::cout << "CPU OMP:              " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP:              " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // implement on OpenCL
        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), as.size());
        gpu::gpu_mem_32u sum_gpu;
        sum_gpu.resizeN(1);

        constexpr int N_VARIANTS = 4;
        constexpr int WORK_PER_ITEM = 64;
        constexpr int WORKGROUP_SIZE = 128;

        for (int variant = 0; variant < N_VARIANTS; ++variant) {

            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum",
                               "-D WORKGROUP_SIZE=" + std::to_string(WORKGROUP_SIZE) +
                               " -D WORK_PER_ITEM=" + std::to_string(WORK_PER_ITEM) +
                               " -D VARIANT" + std::to_string(variant));
            kernel.compile(BUILD_LOG);
            unsigned int localN = (variant == 1 || variant == 2) ? n / WORK_PER_ITEM : n;
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                sum_gpu.writeN(&sum, 1);

                kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, localN), as_gpu, n, sum_gpu);
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "OpenCL result should be consistent!");
                t.nextLap();
            }
            std::cout << "OpenCL variant " << variant << ":     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "OpenCL variant " << variant << ":     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s"
                      << std::endl;
        }

        {
            gpu::gpu_mem_32u bs_gpu, cs_gpu;
            bs_gpu.resizeN((n + 1) / 2);
            cs_gpu.resizeN((n + 1) / 2);

            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "tree_sum", "-D WORKGROUP_SIZE=" + std::to_string(WORKGROUP_SIZE) +" -D VARIANT0");
            kernel.compile(BUILD_LOG);

            timer t;

            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int cur_n = n;

                kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, cur_n), as_gpu, cur_n, bs_gpu);
                cur_n = (cur_n + 1) / 2;

                while (cur_n > 1) {
                    kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, cur_n), bs_gpu, cur_n, cs_gpu);
                    cur_n = (cur_n + 1) / 2;
                    std::swap(bs_gpu, cs_gpu);
                }

                unsigned int sum;
                bs_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "OpenCL result should be consistent!");
                t.nextLap();
            }
            std::cout << "OpenCL variant tree:  " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "OpenCL variant tree:  " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s"
                      << std::endl;
        }

        {
            gpu::gpu_mem_32u bs_gpu, cs_gpu;
            bs_gpu.resizeN((n + 1) / 2);
            cs_gpu.resizeN((n + 1) / 2);

            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "tree_sum2", "-D WORKGROUP_SIZE=" + std::to_string(WORKGROUP_SIZE) +" -D VARIANT0");
            kernel.compile(BUILD_LOG);

            timer t;

            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int cur_n = n;

                kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, cur_n), as_gpu, cur_n, bs_gpu);
                cur_n = (cur_n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

                while (cur_n > 1) {
                    kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, cur_n), bs_gpu, cur_n, cs_gpu);
                    cur_n = (cur_n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                    std::swap(bs_gpu, cs_gpu);
                }

                unsigned int sum;
                bs_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "OpenCL result should be consistent!");
                t.nextLap();
            }
            std::cout << "OpenCL variant tree2: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "OpenCL variant tree2: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s"
                      << std::endl;
        }
    }

    // Анализ вариантов:
    // 0) Базлайн на GPU работает почти на треть быстрее варианта с векторизацией на CPU (на CPU он очень медленный),
    //    что, видимо, связанно с оптимизациями atomic_add
    // 1) Вариант с батчем значений -- лучший на CPU и 2ой в общем зачёте. Не очень понятно, почему OMP не делает так же,
    //    но тут компилятор прекрасно справляется с компиляцией этого варианта в 512-битные векторные операции.
    //    С точки зрения CPU, в этом варианте данные уложены оптимально (последовательно), а локальной памяти в CPU нет,
    //    поэтому этот вариант работает на нём лучше, чем все остальные
    // 2) Вариант с оптимизаций доступа к данным для GPU закономерно выигрывает,
    //    конкуренцию ему мог бы составить только вариант с деревом, но, похоже, длина тестового примера слишком маленькая,
    //    чтобы atomic_add начал проигрывать логарифму перезапусков кернела
    // 3) Ожидаемо сопоставим с вариантом (1)
    // tree) На CPU вариант без синхронизации (кроме вызовов кернела) в 2 раза лучше,
    //       на GPU -- наоборот вариант с большим числом итераций в кернеле в 2 раза быстрее.
    //       Но они в любом случае проигрывают варианту с батчем и оптимальной укладкой данных.
    //       Возможно, драйвер сам оптимизирует atomic_add до чего-то похожего на это дерево
    //       (скорости на (tree2) и (1) на GPU как раз очень похожи)
}
