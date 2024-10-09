#include <atomic>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <thread>
#include <vector>

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

int main(int argc, char **argv) {
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100 * 1000 * 1000;
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
        std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
#pragma omp parallel for reduction(+ : sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            std::atomic<unsigned int> atomic_sum(0);
            int num_threads = std::thread::hardware_concurrency();
            std::vector<std::thread> threads;

            auto sum_thread = [&as, &atomic_sum, n, num_threads](int start, int end) {
                unsigned int local_sum = 0;
                for (int i = start; i < end; ++i) {
                    local_sum += as[i];
                }
                atomic_sum.fetch_add(local_sum);
            };

            int chunk_size = n / num_threads;
            for (int i = 0; i < num_threads; ++i) {
                int start = i * chunk_size;
                int end = (i == num_threads - 1) ? n : (i + 1) * chunk_size;
                threads.emplace_back(sum_thread, start, end);
            }

            for (auto &thread : threads) {
                thread.join();
            }

            EXPECT_THE_SAME(reference_sum, atomic_sum.load(), "Atomic result should be consistent!");
            t.nextLap();
        }
        std::cout << "Atomic:  " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "Atomic:  " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // TODO: implement on OpenCL
        // gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    }
}