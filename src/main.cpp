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

#include "utils.h"

bool selectAvailableDeviceFromPlatforms(
    const std::vector<cl_platform_id>& platforms,
    cl_device_type priorityDeviceTypes,
    cl_platform_id& platform,
    cl_device_id& device
) {
    bool foundDevice = false;
    auto platformsCount = platforms.size();
    cl_device_id deviceId;

    for (int platformIndex = 0; platformIndex < platformsCount && !foundDevice; ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        platform = platforms[platformIndex];
        std::cout << "\tPlatform name: " << getPlatformProperty<std::string>(platform, CL_PLATFORM_NAME) << std::endl;

        try {
            deviceId = getDeviceFromPlatform(platform, priorityDeviceTypes);
            foundDevice = true;
        }
        catch (std::exception &e) {
            std::cout << e.what() << std::endl;
        }
    }

    if (foundDevice) device = deviceId;
    return foundDevice;
}

void selectAvailableDevice(cl_device_type priorityDeviceTypes, cl_platform_id& platform, cl_device_id& device) {
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    // Iterating over available platforms
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    bool foundDevice = selectAvailableDeviceFromPlatforms(platforms, priorityDeviceTypes, platform, device);

    if (!foundDevice) {
        std::cout << "No devices with specified types found. Fallback to CPU and GPU types." << std::endl;
        foundDevice = selectAvailableDeviceFromPlatforms(platforms, CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU, platform, device);
    }

    if (!foundDevice) {
        throw std::runtime_error("No devices for OpenCL available.");
    }

    std::cout << "Selected device: " << getDeviceProperty<std::string>(device, CL_DEVICE_NAME) << std::endl;
}

int main() {
    try {
        // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
        if (!ocl_init())
            throw std::runtime_error("Can't init OpenCL driver!");

        // TODO 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
        // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
        cl_platform_id platform;
        cl_device_id device;
        selectAvailableDevice(CL_DEVICE_TYPE_GPU, platform, device);

        // TODO 2 Создайте контекст с выбранным устройством
        // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
        // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
        // код по переданному аргументом errcode_ret указателю)
        cl_int ret = 0;

        cl_context context;
        cl_context_properties context_properties[] = {
            CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0
        };

        context = clCreateContext(context_properties, 1, &device, nullptr, nullptr, &ret);
        OCL_CHECK_CALL_RET(ret);

        // Контекст и все остальные ресурсы следует освобождать с помощью clReleaseContext/clReleaseQueue/clReleaseMemObject... (да, не очень RAII, но это лишь пример)

        // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
        // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
        // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
        cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, &ret);
        OCL_CHECK_CALL_RET(ret);

        // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
        unsigned int n = 100 * 1000 * 1000;
        std::vector<float> as(n, 0);
        std::vector<float> bs(n, 0);
        std::vector<float> cs(n, 0);
        FastRandom r(n);
        for (unsigned int i = 0; i < n; ++i) {
            as[i] = r.nextf();
            bs[i] = r.nextf();
        }
        std::cout << "Data generated for n=" << n << "!" << std::endl;

        // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
        // См. Buffer Objects -> clCreateBuffer
        // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт, что чисел в каждом массиве n штук
        // Данные в as и bs можно прогрузить этим же методом, скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг, на это указывающий)
        // или же через метод Buffer Objects -> clEnqueueWriteBuffer
        cl_mem bufa = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(as[0]), as.data(), &ret);
        OCL_CHECK_CALL_RET(ret);
        cl_mem bufb = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(bs[0]), bs.data(), &ret);
        OCL_CHECK_CALL_RET(ret);
        cl_mem bufc = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(cs[0]), cs.data(), &ret);
        OCL_CHECK_CALL_RET(ret);

        // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
        // затем убедитесь, что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания),
        // напечатав исходники в консоль (if проверяет, что удалось считать хоть что-то)
        std::string kernelSources;
        {
            std::ifstream file("src/cl/aplusb.cl");
            kernelSources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            if (kernelSources.empty()) {
                throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
            }
            // std::cout << kernel_sources << std::endl;
        }

        // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
        // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
        // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель
        const char *kernelSourcesPtr = kernelSources.c_str();
        size_t kernelSourcesSize = kernelSources.size();
        cl_program program = clCreateProgramWithSource(context, 1, &kernelSourcesPtr, &kernelSourcesSize, &ret);
        OCL_CHECK_CALL_RET(ret);

        // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
        // см. clBuildProgram
        ret = clBuildProgram(program, 1, &device, "", nullptr, nullptr);

        // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
        // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
        // см. clGetProgramBuildInfo
        //    size_t log_size = 0;
        //    std::vector<char> log(log_size, 0);
        //    if (log_size > 1) {
        //        std::cout << "Log:" << std::endl;
        //        std::cout << log.data() << std::endl;
        //    }
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize, 0);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        if (logSize > 0) {
            std::cout << "=====================" << std::endl;
            std::cout << "Log:" << std::endl;
            std::cout << log.data() << std::endl;
            std::cout << "=====================" << std::endl;
        }

        OCL_CHECK_CALL_RET(ret);

        // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
        // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
        cl_kernel kernel = clCreateKernel(program, "aplusb", &ret);
        OCL_CHECK_CALL_RET(ret);

        // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
        {
            unsigned int i = 0;
            ret = clSetKernelArg(kernel, i++, sizeof(cl_mem), &bufa);
            OCL_CHECK_CALL_RET(ret);
            ret = clSetKernelArg(kernel, i++, sizeof(cl_mem), &bufb);
            OCL_CHECK_CALL_RET(ret);
            ret = clSetKernelArg(kernel, i++, sizeof(cl_mem), &bufc);
            OCL_CHECK_CALL_RET(ret);
            ret = clSetKernelArg(kernel, i++, sizeof(unsigned int), &n);
            OCL_CHECK_CALL_RET(ret);
        }

        // TODO 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

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
                // clEnqueueNDRangeKernel...
                cl_event event;
                ret = clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, &global_work_size, &workGroupSize, 0, nullptr, &event);
                OCL_CHECK_CALL_RET(ret);

                // clWaitForEvents...
                ret = clWaitForEvents(1, &event);
                OCL_CHECK_CALL_RET(ret);

                t.nextLap();// При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
                clReleaseEvent(event);
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
            std::cout << "GFlops: " << n / t.lapAvg() / 1e9 << std::endl;

            // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
            // - Всего элементов в массивах по n штук
            // - Размер каждого элемента sizeof(float)=4 байта
            // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
            // - В гигабайте 1024*1024*1024 байт
            // - Среднее время выполнения кернела равно t.lapAvg() секунд
            std::cout << "VRAM bandwidth: " << (3 * n * sizeof(float)) / t.lapAvg() / (1024 * 1024 * 1024) << " GB/s" << std::endl;
        }

        // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
        {
            timer t;
            for (unsigned int i = 0; i < 20; ++i) {
                // clEnqueueReadBuffer...
                ret = clEnqueueReadBuffer(commandQueue, bufc, CL_TRUE, 0, cs.size() * sizeof(cs[0]), cs.data(), 0, nullptr, nullptr);
                OCL_CHECK_CALL_RET(ret);

                t.nextLap();
            }
            std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "VRAM -> RAM bandwidth: " << (3 * n * sizeof(float)) / t.lapAvg() / (1024 * 1024 * 1024) << " GB/s" << std::endl;
        }

        // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
        for (unsigned int i = 0; i < n; ++i) {
            if (cs[i] != as[i] + bs[i]) {
                throw std::runtime_error("CPU and GPU results differ!");
            }
        }


        // TODO 17 Clear allocated resources (context, etc.)
        // Release memory objects (buffers)
        ret = clReleaseMemObject(bufa);
        OCL_CHECK_CALL_RET(ret);
        ret = clReleaseMemObject(bufb);
        OCL_CHECK_CALL_RET(ret);
        ret = clReleaseMemObject(bufc);
        OCL_CHECK_CALL_RET(ret);

        // Release kernel
        ret = clReleaseKernel(kernel);
        OCL_CHECK_CALL_RET(ret);

        // Release program
        ret = clReleaseProgram(program);
        OCL_CHECK_CALL_RET(ret);

        // Release command queue
        ret = clReleaseCommandQueue(commandQueue);
        OCL_CHECK_CALL_RET(ret);

        // Release context
        ret = clReleaseContext(context);
        OCL_CHECK_CALL_RET(ret);
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        throw;
    }

    return 0;
}
