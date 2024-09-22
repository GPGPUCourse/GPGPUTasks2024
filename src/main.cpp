#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/fast_random.h>
#include <libutils/timer.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>


template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // 1 Выбираем устройство

    cl_uint platformCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformCount));
    std::vector<cl_platform_id> platforms(platformCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformCount, platforms.data(), nullptr));

    cl_device_id selectedDevice = nullptr;
    bool gpuFound = false;

    for (cl_uint i = 0; i < platformCount; ++i) {
        cl_platform_id platform = platforms[i];

        cl_uint deviceCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount));
        std::vector<cl_device_id> devices(deviceCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr));

        for (cl_uint j = 0; j < deviceCount; ++j) {
            cl_device_id device = devices[j];

            size_t deviceTypeSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, 0, nullptr, &deviceTypeSize));
            cl_device_type deviceType = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, deviceTypeSize, &deviceType, nullptr));

            if (selectedDevice == nullptr and deviceType == CL_DEVICE_TYPE_CPU) {
                selectedDevice = device;
            } else if (not gpuFound and deviceType == CL_DEVICE_TYPE_GPU) {
                selectedDevice = device;
                gpuFound = true;
            }
        }
    }

    // 2 Создаем контекст с выбранным устройством
    cl_int errcode = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &selectedDevice, nullptr, nullptr, &errcode);
    OCL_SAFE_CALL(errcode);

    // 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    cl_command_queue queue = clCreateCommandQueue(context, selectedDevice, 0, &errcode);
    OCL_SAFE_CALL(errcode);

    unsigned int n = 100 * 1000 * 1000;
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    // 4 Создаем три буфера в памяти устройства
    size_t dataSize = n * sizeof(float);
    cl_mem as_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dataSize, as.data(), &errcode);
    OCL_SAFE_CALL(errcode);
    cl_mem bs_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dataSize, bs.data(), &errcode);
    OCL_SAFE_CALL(errcode);
    cl_mem cs_gpu = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, nullptr, &errcode);
    OCL_SAFE_CALL(errcode);

    // 6 Выполним (5)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        // std::cout << kernel_sources << std::endl;
    }

    // 7 Создаем OpenCL-подпрограмму с исходниками кернела
    const char* source = kernel_sources.c_str();
    size_t sourceSize = kernel_sources.size();
    cl_program program = clCreateProgramWithSource(context, 1, &source, &sourceSize, &errcode);
    OCL_SAFE_CALL(errcode);

    // 8 Скомпилируем программу и напечатаем в консоль лог компиляции
    errcode = clBuildProgram(program, 1, &selectedDevice, nullptr, nullptr, nullptr);
    if (errcode != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, selectedDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, selectedDevice, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cout << "Build log:" << std::endl;
        std::cout << log.data() << std::endl;
        OCL_SAFE_CALL(errcode);
    }

    // 9 Создаем OpenCL-kernel в созданной подпрограмме
    cl_kernel kernel = clCreateKernel(program, "aplusb", &errcode);
    OCL_SAFE_CALL(errcode);

    // 10 Выставляем все аргументы в кернеле через clSetKernelArg
    {
        unsigned int i = 0;
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &as_gpu));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &bs_gpu));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &cs_gpu));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(unsigned int), &n));
    }

    // 12 Запускаем выполнения кернела
    {
        size_t workGroupSize = 128;
        size_t global_work_size = ((n + workGroupSize - 1) / workGroupSize) * workGroupSize;
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, &workGroupSize, 0, nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            OCL_SAFE_CALL(clReleaseEvent(event));
            t.nextLap();
        }
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        // 13 Рассчитаем достигнутые гигафлопcы
        double gflops = n / (t.lapAvg() * 1e9);
        std::cout << "GFlops: " << gflops << std::endl;

        // 14 Рассчитаем используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        double totalBytes = 3.0 * n * sizeof(float); // 3 accesses per element
        double bandwidthGBs = totalBytes / (t.lapAvg() * 1024 * 1024 * 1024);
        std::cout << "VRAM bandwidth: " << bandwidthGBs << " GB/s" << std::endl;
    }

    // 15 Скачиваем результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM)
    {
        timer t;
        size_t dataSize = n * sizeof(float);
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueReadBuffer(queue, cs_gpu, CL_TRUE, 0, dataSize, cs.data(), 0, nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            OCL_SAFE_CALL(clReleaseEvent(event));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        double totalBytes = n * sizeof(float);
        double bandwidthGBs = totalBytes / (t.lapAvg() * 1024 * 1024 * 1024);
        std::cout << "VRAM -> RAM bandwidth: " << bandwidthGBs << " GB/s" << std::endl;
    }

    // 16 Сверяем результаты вычислений со сложением чисел на процессоре
    {
        float epsilon = 1e-6f;
        for (unsigned int i = 0; i < n; ++i) {
            if (std::fabs(cs[i] - (as[i] + bs[i])) > epsilon) {
                throw std::runtime_error("CPU and GPU results differ at index " + std::to_string(i));
            }
        }
    }

    // Освобождаем ресурсы
    clReleaseMemObject(as_gpu);
    clReleaseMemObject(bs_gpu);
    clReleaseMemObject(cs_gpu);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
