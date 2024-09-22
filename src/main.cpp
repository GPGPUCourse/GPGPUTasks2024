#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/fast_random.h>
#include <libutils/timer.h>

#include <cassert>
#include <cstring>
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

void checkErrorCode(const cl_int errorCode, const int lineNum = 0) {
    if (errorCode != NULL) {
        std::cout << "Got error code: " << errorCode << " at line " << lineNum << std::endl;
        exit(-1);
    }
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

    // TODO 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));
    cl_platform_id platform = platforms[0];

    cl_uint devicesCount = 0;
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

    std::vector<cl_device_id> devices(devicesCount);
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
    cl_device_id device = devices[0];

    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)

    std::vector<cl_context_properties> contextProperties = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
    cl_int errorCode = NULL;
    cl_context context = clCreateContext(contextProperties.data(), devicesCount, devices.data(), nullptr, nullptr, &errorCode);

    checkErrorCode(errorCode);

    // Контекст и все остальные ресурсы следует освобождать с помощью clReleaseContext/clReleaseQueue/clReleaseMemObject... (да, не очень RAII, но это лишь пример)

    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач

    cl_command_queue commandQueue = clCreateCommandQueue(context, device, NULL, &errorCode);

    checkErrorCode(errorCode);

    unsigned int n = 100*1000 * 1000;
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

    // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт, что чисел в каждом массиве n штук
    // Данные в as и bs можно прогрузить этим же методом, скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг, на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer

    size_t bufferSize = n * sizeof(float);
    cl_mem aBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, nullptr, &errorCode);
    checkErrorCode(errorCode, 103);
    cl_mem bBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, nullptr, &errorCode);
    checkErrorCode(errorCode, 104);
    cl_mem cBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &errorCode);
    checkErrorCode(errorCode, 105);

    errorCode = clEnqueueWriteBuffer(commandQueue, aBuffer, CL_TRUE, 0, bufferSize, as.data(), 0, nullptr, nullptr);
    checkErrorCode(errorCode, 110);
    errorCode = clEnqueueWriteBuffer(commandQueue, bBuffer, CL_TRUE, 0, bufferSize, bs.data(), 0, nullptr, nullptr);
    checkErrorCode(errorCode, 112);
    errorCode = clEnqueueWriteBuffer(commandQueue, cBuffer, CL_TRUE, 0, bufferSize, cs.data(), 0, nullptr, nullptr);
    checkErrorCode(errorCode, 114);

    // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь, что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания),
    // напечатав исходники в консоль (if проверяет, что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        std::cout << kernel_sources << std::endl;
    }

    // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель
    std::vector<const char*> strings = { kernel_sources.c_str() };
    std::vector<size_t> lengths = { strlen(strings[0]) };
    cl_program program = clCreateProgramWithSource(context, 1, strings.data(), lengths.data(), &errorCode);
    checkErrorCode(errorCode, 136);

    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram

    errorCode = clBuildProgram(program, 1, devices.data(), nullptr, nullptr, nullptr);

    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo
    if (errorCode != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size, 0);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        if (log_size > 1) {
            std::cout << "Log:" << std::endl;
            std::cout << log.data() << std::endl;
        }
    }
    else {
        std::cout << "Succesfull build!" << std::endl;
    }

    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
    char kernelName[] = "aplusb";
    cl_kernel kernel = clCreateKernel(program, kernelName, &errorCode);
    checkErrorCode(errorCode, 164);

    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
        unsigned int i = 0;
        errorCode = clSetKernelArg(kernel, i++, sizeof(cl_mem), &aBuffer);
        checkErrorCode(errorCode, 170);
        errorCode = clSetKernelArg(kernel, i++, sizeof(cl_mem), &bBuffer);
        checkErrorCode(errorCode, 172);
        errorCode = clSetKernelArg(kernel, i++, sizeof(cl_mem), &cBuffer);
        checkErrorCode(errorCode, 174);
        errorCode = clSetKernelArg(kernel, i++, sizeof(unsigned int), &n);
        checkErrorCode(errorCode, 176);
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
        cl_event event;
        for (unsigned int i = 0; i < 20; ++i) {
            errorCode = clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, &global_work_size, &workGroupSize, NULL, nullptr, &event);
            checkErrorCode(errorCode, 196);
            errorCode = clWaitForEvents(1, &event);
            checkErrorCode(errorCode, 198);
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
        double gflops = n / pow(10, 9) / t.lapAvg();
        std::cout << "GFlops: " << round(gflops) << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        double vRAMBandwidth = 3*n*sizeof(float) / 1024 / 1024 / 1024 / t.lapAvg();
        std::cout << "VRAM bandwidth: " << vRAMBandwidth << " GB/s" << std::endl;
        errorCode = clReleaseEvent(event);
        checkErrorCode(errorCode, 223);
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        cl_event event;
        for (unsigned int i = 0; i < 20; ++i) {
            errorCode = clEnqueueReadBuffer(commandQueue, cBuffer, CL_TRUE, 0, bufferSize, cs.data(), 0, nullptr, &event);
            checkErrorCode(errorCode, 232);
            errorCode = clWaitForEvents(1, &event);
            checkErrorCode(errorCode, 234);
            t.nextLap();
        }
        double vRAMtoRAMBandwidth = 3*n*sizeof(float) / 1024 / 1024 / 1024 / t.lapAvg();
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << vRAMtoRAMBandwidth << " GB/s" << std::endl;
        errorCode = clReleaseEvent(event);
        checkErrorCode(errorCode, 241);
    }

    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }
    errorCode = clReleaseMemObject(aBuffer);
    checkErrorCode(errorCode, 251);
    errorCode = clReleaseMemObject(bBuffer);
    checkErrorCode(errorCode, 253);
    errorCode = clReleaseMemObject(cBuffer);
    checkErrorCode(errorCode, 255);

    errorCode = clReleaseKernel(kernel);
    checkErrorCode(errorCode, 258);

    errorCode = clReleaseCommandQueue(commandQueue);
    checkErrorCode(errorCode, 262);

    errorCode = clReleaseContext(context);
    checkErrorCode(errorCode, 264);

    errorCode = clReleaseProgram(program);
    checkErrorCode(errorCode, 267);

    return 0;
}
