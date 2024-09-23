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

bool find_gpu(cl_device_id& device, cl_platform_id& platform, std::vector<cl_platform_id>& platforms) {
    for (auto cur_platform : platforms) {

        cl_uint devices_cnt = 0;
        auto found = clGetDeviceIDs(cur_platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &devices_cnt);
        if (found == CL_DEVICE_NOT_FOUND || devices_cnt == 0) {
            continue;
        }

        std::vector<cl_device_id> platform_devices(devices_cnt);
        OCL_SAFE_CALL(clGetDeviceIDs(cur_platform,
                                     CL_DEVICE_TYPE_GPU,
                                     devices_cnt,
                                     platform_devices.data(),
                                     nullptr));

        device = platform_devices[0];
        platform = cur_platform;
        return true;
    }
    return false;
}

bool find_cpu(cl_device_id& device, cl_platform_id& platform, std::vector<cl_platform_id>& platforms) {
    for (auto cur_platform: platforms) {
        cl_uint devices_cnt = 0;
        auto found = clGetDeviceIDs(cur_platform, CL_DEVICE_TYPE_CPU, 0, nullptr, &devices_cnt);
        if (found == CL_DEVICE_NOT_FOUND || devices_cnt == 0) {
            continue;
        }

        std::vector<cl_device_id> platform_devices(devices_cnt);
        OCL_SAFE_CALL(clGetDeviceIDs(cur_platform,
                                     CL_DEVICE_TYPE_CPU,
                                     devices_cnt,
                                     platform_devices.data(),
                                     nullptr));

        device = platform_devices[0];
        platform = cur_platform;
        return true;
    }
    return false;
}

#define free_resource(ctx, commandQueue, as, bs, cs, kernel, program ) \
    OCL_SAFE_CALL(clReleaseContext(ctx)); \
    OCL_SAFE_CALL(clReleaseCommandQueue(commandQueue)); \
    OCL_SAFE_CALL(clReleaseMemObject(as_gpu)); \
    OCL_SAFE_CALL(clReleaseMemObject(bs_gpu)); \
    OCL_SAFE_CALL(clReleaseMemObject(cs_gpu)); \
    OCL_SAFE_CALL(clReleaseKernel(kernel)); \
    OCL_SAFE_CALL(clReleaseProgram(program));  \
    printf("Resources are free\n");     \


bool find_device (cl_device_id& device, cl_platform_id& platform) {
    cl_uint platforms_cnt = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platforms_cnt));
    std::vector<cl_platform_id> platforms(platforms_cnt);
    OCL_SAFE_CALL(clGetPlatformIDs(platforms_cnt, platforms.data(), nullptr));

    return find_gpu(device, platform, platforms) || find_cpu(device, platform, platforms);
}

int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // TODO 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
    cl_device_id device = 0;
    cl_platform_id platform = 0;
    if (!find_device(device, platform)) {
        std::cout << "Can`t find any platform\n";
        return 1;
    }

    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
    cl_int err = 0;
    cl_context ctx = clCreateContext(props, 1, &device, nullptr, nullptr, &err);
    OCL_SAFE_CALL(err);

    // Контекст и все остальные ресурсы следует освобождать с помощью clReleaseContext/clReleaseQueue/clReleaseMemObject... (да, не очень RAII, но это лишь пример)

    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    cl_command_queue commandQueue = clCreateCommandQueue(ctx, device, 0, &err);
    OCL_SAFE_CALL(err);
    unsigned int n = 1000 * 1000  * 100;
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
    cl_mem as_gpu = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), as.data(), &err);
    OCL_SAFE_CALL(err);
    cl_mem bs_gpu= clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), bs.data(), &err);
    OCL_SAFE_CALL(err);
    cl_mem cs_gpu = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, &err);
    OCL_SAFE_CALL(err);
//    std::cout << "141" << std::endl;
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
//         std::cout << kernel_sources << std::endl;
    }

//    std::cout << "154" << std::endl;

    // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель

    cl_program program;
    const char* src = kernel_sources.c_str();
    size_t size = kernel_sources.size();
    program = clCreateProgramWithSource(ctx,1, &src, &size, &err);
    OCL_SAFE_CALL(err);
    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram
    OCL_SAFE_CALL( clBuildProgram(program, 1, &device, "", nullptr, nullptr));
    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo
    //    size_t log_size = 0;
    //    std::vector<char> log(log_size, 0);
    //    if (log_size > 1) {
    //        std::cout << "Log:" << std::endl;
    //        std::cout << log.data() << std::endl;
    //    }
//    std::cout << "177" << std::endl;
    size_t log_size = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
    std::vector<char> log(log_size, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));
    if (log_size > 1) {
        std::cout << "Log:" << std::endl;
        std::cout << log.data() << std::endl;
    }

//    std::cout << "188" << std::endl;

    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects

    auto kernel = clCreateKernel(program, "aplusb", &err);
    OCL_SAFE_CALL(err);
//    std::cout << "192" << std::endl;
    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
         unsigned int i = 0;
         clSetKernelArg(kernel, i++, sizeof(cl_mem*), &as_gpu);
         clSetKernelArg(kernel, i++, sizeof(cl_mem*), &bs_gpu);
         clSetKernelArg(kernel, i++, sizeof(cl_mem*), &cs_gpu);
         clSetKernelArg(kernel, i++, sizeof(unsigned int), &n);
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
            // clWaitForEvents...
            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, &global_work_size, &workGroupSize, 0, nullptr, &event));
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
        std::cout << "GFlops: " << (double)n / t.lapAvg() / 1000000000 << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << 3.0 * n * sizeof(float) / (1024*1024*1024) / t.lapAvg() << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            // clEnqueueReadBuffer...
            cl_event event;
            OCL_SAFE_CALL(clEnqueueReadBuffer(commandQueue, cs_gpu, CL_TRUE, 0,  n * sizeof(float), cs.data(), 0, nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << (double)n * sizeof(float) / (1024 * 1024 * 1024) / t.lapAvg() << " GB/s" << std::endl;
    }

    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
        for (unsigned int i = 0; i < n; ++i) {
            if (cs[i] != as[i] + bs[i]) {
                throw std::runtime_error("CPU and GPU results differ!");
                free_resource(ctx, commandQueue, as_gpu, bs_gpu, cs_gpu, kernel, program)
            }
        }
        free_resource(ctx, commandQueue, as_gpu, bs_gpu, cs_gpu, kernel, program)
    return 0;
}
