#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/merge_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>

const int benchmarkingIters = 1;
const int benchmarkingItersCPU = 1;
const unsigned int n = 1024 * 1024 + 33 + 70 + 256;

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

std::vector<int> computeCPU(const std::vector<int> &as)
{
    std::vector<int> cpu_sorted;

    timer t;
    for (int iter = 0; iter < benchmarkingItersCPU; ++iter) {
        cpu_sorted = as;
        t.restart();
        std::sort(cpu_sorted.begin(), cpu_sorted.end());
        t.nextLap();
    }
//    std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
//    std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

    return cpu_sorted;
}

std::vector<int> computeCPU2(const std::vector<int> &as)
{

    int mx = as[0];
    for (int i = 1; i < as.size(); ++i) {
        mx = std::max(mx, as[i]);
    }
    std::vector<int> counters(mx + 1, 0);
    for (int i = 0; i < as.size(); ++i) {
        counters[as[i]]++;
    }
    std::vector<int> cpu_sorted;
    cpu_sorted.reserve(n);
    for(int i = 0; i < counters.size(); ++i){
        for (int j = 0; j < counters[i]; ++j) {
            cpu_sorted.push_back(i);
        }
    }

    return cpu_sorted;
}


void log_array(gpu::gpu_mem_32i &vec, const unsigned int n){
    std::vector<int> vec_cpu(n);
    vec.readN(vec_cpu.data(), n);
    for(auto &elem: vec_cpu){
        std::cout << elem << " ";
    }
    std::cout << std::endl;
}

void test_result(gpu::gpu_mem_32i &gt, gpu::gpu_mem_32i &test, const unsigned int n, const unsigned int block_size, int seed){
    std::vector<int> gt_cpu(n), test_cpu(n);
    gt.readN(gt_cpu.data(), n);
    test.readN(test_cpu.data(), n);
    for (int i = 0; i < gt_cpu.size(); ++i) {
        if (gt_cpu[i] != test_cpu[i]){
            std::cerr << "problem on seed: " << seed << " block_size: " << block_size;
            std::cerr << std::endl;
            exit(1);
        }
    }

}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    std::vector<int> as(n), bs(n);


    gpu::gpu_mem_32i as_gpu;
    as_gpu.resizeN(n);

    gpu::gpu_mem_32i bs_gpu;
    bs_gpu.resizeN(n);

    gpu::gpu_mem_32i bs_gpu2;
    bs_gpu2.resizeN(n);

    const unsigned int groupSize = std::min((int) 128, (int) n);
    const unsigned int workSize = (n + groupSize - 1) / groupSize * groupSize;
    gpu::gpu_mem_32i ind_gpu;
    ind_gpu.resizeN(workSize / groupSize * 2);

    std::string defines = "-DGROUP_SIZE=" + std::to_string(groupSize);
    ocl::Kernel calculate_indices(merge_kernel, merge_kernel_length, "calculate_indices", defines);
    ocl::Kernel merge_local(merge_kernel, merge_kernel_length, "merge_local", defines);
    calculate_indices.compile();
    merge_local.compile();

    ocl::Kernel merge_global(merge_kernel, merge_kernel_length, "merge_global");
    merge_global.compile();


    for (int seed = 0; seed < 10000; ++seed) {

    FastRandom r(n + seed);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.next(1, 3);
        bs[i] = 0;
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    const std::vector<int> cpu_sorted = computeCPU2(as);


    bs_gpu.readN(bs.data(), n);


//    {
//        const unsigned int groupSize = std::min(128u, n);
//        const unsigned int workSize = (n + groupSize - 1) / groupSize * groupSize;
//
//        ocl::Kernel merge_global(merge_kernel, merge_kernel_length, "merge_global");
//        merge_global.compile();
//
//        timer t;
//        for (int iter = 0; iter < benchmarkingIters; ++iter) {
//            as_gpu.writeN(as.data(), n);
//            t.restart();
//            unsigned int block_size = 2;
//            while (block_size / 2 < n) {
//                merge_global.exec(gpu::WorkSize(groupSize, workSize), as_gpu, bs_gpu, n, block_size);
//                std::swap(as_gpu, bs_gpu);
//                block_size *= 2;
//            }
//            t.nextLap();
//        }
//        std::cout << "GPU global: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
//        std::cout << "GPU global: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
//        as_gpu.readN(as.data(), n);
//        for (int i = 0; i < n; ++i) {
//            EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results! ");
//        }
//    }

    // remove me for task 5.2
//    return 0;
//
    {


        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();
            unsigned int block_size = 2;
            while (block_size / 2 < n){
                unsigned int log = (unsigned int) 1;
                if (log){
                    calculate_indices.exec(gpu::WorkSize(groupSize, workSize / groupSize), as_gpu, ind_gpu, n, block_size);
                    merge_local.exec(gpu::WorkSize(groupSize, workSize), as_gpu, ind_gpu, bs_gpu, n, block_size, log);
                    std::cout << "block_size: " << block_size << std::endl;
                    log_array(as_gpu, n);
                    log_array(ind_gpu, workSize / groupSize * 2);
                    log_array(bs_gpu, n);

                    merge_global.exec(gpu::WorkSize(groupSize, workSize), as_gpu, bs_gpu2, n, block_size);
                    log_array(bs_gpu2, n);
                } else {


                    calculate_indices.exec(gpu::WorkSize(groupSize, workSize / groupSize), as_gpu, ind_gpu, n, block_size);
                    merge_local.exec(gpu::WorkSize(groupSize, workSize), as_gpu, ind_gpu, bs_gpu, n, block_size, log);

//                    merge_global.exec(gpu::WorkSize(groupSize, workSize), as_gpu, bs_gpu2, n, block_size);
//                    test_result(bs_gpu2, bs_gpu, block_size, n, seed);
                }
                std::swap(as_gpu, bs_gpu);
                block_size *= 2;
            }
            t.nextLap();
        }
//        std::cout << "GPU local: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
//        std::cout << "GPU local: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
        as_gpu.readN(as.data(), n);

        for (int i = 0; i < n; ++i) {
            EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results! " + std::to_string(seed) + " " + std::to_string(i));
        }
    }
    }

    return 0;
}
