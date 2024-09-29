#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "cl/sum_cl.h" 

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
        as[i] = (unsigned int)r.next(0, std::numeric_limits<unsigned int>::max() / n);
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
    
    try {
        std::string kernelSource = loadKernel("sum.cl");

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found.");
        
        cl::Platform platform = platforms[0];

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) throw std::runtime_error("No OpenCL devices found.");
        
        cl::Device device = devices[0]; // Use first GPU device

        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        cl::Program::Sources sources(1, std::make_pair(kernelSource.c_str(), kernelSource.length()));
        cl::Program program(context, sources);
        program.build({device});
        
        cl::Buffer bufferData(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * n, as.data());
        cl::Buffer bufferResult(context, CL_MEM_WRITE_ONLY, sizeof(unsigned int));

        cl::Kernel kernel(program, "sum_1");
        kernel.setArg(0, bufferData);
        kernel.setArg(1, bufferResult);
        kernel.setArg(2, n);

        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            queue.enqueueWriteBuffer(bufferResult, CL_TRUE, 0, sizeof(unsigned int), &sum);

            size_t globalWorkSize = 256 * 256; 
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NullRange);

            queue.enqueueReadBuffer(bufferResult, CL_TRUE, 0, sizeof(unsigned int), &sum);

            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
        }

        std::cout << "GPU:    " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU:    " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;

    } catch (const cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")" << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
