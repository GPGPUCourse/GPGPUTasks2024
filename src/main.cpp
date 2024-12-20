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

    unsigned int n;
    std::vector<float> as;
    std::vector<float> bs;
    std::vector<float> cs;
    std::string kernel_sources;

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue commandQueue;
    cl_mem as_buffer;
    cl_mem bs_buffer;
    cl_mem cs_buffer;
    cl_program program;
    cl_kernel kernel;

    { // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
        n = 100 * 1000 * 1000;
        as = std::vector<float>(n, 0);
        bs = std::vector<float>(n, 0);
        cs = std::vector<float>(n, 0);
        FastRandom r(n);
        for (unsigned int i = 0; i < n; ++i) {
            as[i] = r.nextf();
            bs[i] = r.nextf();
        }
        std::cout << "Data generated for n=" << n << "!" << std::endl;
    }

    { // Prepare kernel source
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        std::cout << "============ KERNEL ============" << std::endl;
        std::cout << kernel_sources << std::endl;
        std::cout << "================================" << std::endl;
    }

    { // Get platform
        cl_uint num_platforms;
        OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &num_platforms));
        std::vector<cl_platform_id> platforms(num_platforms);
        OCL_SAFE_CALL(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));
        platform = platforms.at(0);
    }

    { // Get device
        cl_uint devicesSize;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 0, nullptr, &devicesSize));
        std::vector<cl_device_id> devices(devicesSize);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, devicesSize, devices.data(), nullptr));
        device = devices.at(0);
    }

    { // Create context
        cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0 };
        cl_int errcode_ret;
        context = clCreateContext(properties, 1, &device, nullptr, nullptr, &errcode_ret);
        OCL_SAFE_CALL(errcode_ret);
    }

    { // Create command queue
        cl_int commandQueueError;
        commandQueue = clCreateCommandQueue(context, device, 0, &commandQueueError);
        OCL_SAFE_CALL(commandQueueError);
    }

    { // Create program
        cl_int error;
        const char *strings[] = { kernel_sources.c_str() };
        size_t lengths[] = { kernel_sources.length() };
        program = clCreateProgramWithSource(context, 1, strings, nullptr, &error);
        OCL_SAFE_CALL(error);
        error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

        if (error != CL_SUCCESS) {
            std::cout << "Error building program" << std::endl;

            size_t buildLogSize;
            OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &buildLogSize));
            std::vector<char> buildLog(buildLogSize);
            OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog.data(), nullptr));
            std::cout << "=== Build Log: ===" << std::endl;
            std::cout << buildLog.data() << std::endl;
            OCL_SAFE_CALL(error);
        }
    }

    { // Create kernel
        cl_int error;
        kernel = clCreateKernel(program, "aplusb", &error);
        OCL_SAFE_CALL(error);
    }

    { // Create buffers
        cl_int error;

        as_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, &error);
        OCL_SAFE_CALL(error);
        bs_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, &error);
        OCL_SAFE_CALL(error);
        cs_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, &error);
        OCL_SAFE_CALL(error);

        error = clEnqueueWriteBuffer(commandQueue, as_buffer, CL_TRUE, 0, sizeof(float) * n, as.data(), 0, NULL, NULL);
        OCL_SAFE_CALL(error);
        error = clEnqueueWriteBuffer(commandQueue, bs_buffer, CL_TRUE, 0, sizeof(float) * n, bs.data(), 0, NULL, NULL);
        OCL_SAFE_CALL(error);
    }

    { // Set up all args
        OCL_SAFE_CALL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &as_buffer));
        OCL_SAFE_CALL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bs_buffer));
        OCL_SAFE_CALL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &cs_buffer));
        OCL_SAFE_CALL(clSetKernelArg(kernel, 3, sizeof(unsigned int), &n));
    }

    // TODO 12 Запустите выполнения кернела:
    // - С одномерной рабочей группой размера 128
    // - В одномерном рабочем пространстве размера roundedUpN, где roundedUpN - наименьшее число, кратное 128 и при этом не меньшее n
    // - см. clEnqueueNDRangeKernel
    // - Обратите внимание, что, чтобы дождаться окончания вычислений (чтобы знать, когда можно смотреть результаты в cs_gpu) нужно:
    //   - Сохранить событие "кернел запущен" (см. аргумент "cl_event *event")
    //   - Дождаться завершения полунного события - см. в документации подходящий метод среди Event Objects
    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;// Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            size_t global = (n + 127) / 128 * 128;
            size_t local = 128;
            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, &global, &local, 0, nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            t.nextLap();// При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        // TODO 13 Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "GFlops: " << (float) n / 1e9 / t.lapAvg() << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << (float) (3 * n * sizeof(float)) / t.lapAvg() / (1024*1024*1024) << " GB/s" << std::endl;
    }

    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            clEnqueueReadBuffer(commandQueue, cs_buffer, CL_TRUE, 0, n * sizeof(float), cs.data(), 0, NULL, NULL);
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << (float) (n * sizeof(float)) / t.lapAvg() / (1024*1024*1024) << " GB/s" << std::endl;
    }

    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    {
        for (unsigned int i = 0; i < n; ++i) {
           if (cs[i] != as[i] + bs[i]) {
               throw std::runtime_error("CPU and GPU results differ!");
           }
       }
    }

    return 0;
}
