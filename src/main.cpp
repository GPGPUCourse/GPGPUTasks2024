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

std::pair<cl_platform_id, cl_device_id> choose_device() {
    cl_uint platforms_count = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platforms_count));

    assert(platforms_count > 0);

    std::vector<cl_platform_id> platform_ids(platforms_count);
    OCL_SAFE_CALL(clGetPlatformIDs(platforms_count, platform_ids.data(), nullptr));

    cl_platform_id cpu_platform_id;
    cl_device_id cpu_device_id;

    bool has_found_cpu = false;

    for (auto platform_id: platform_ids) {
        cl_uint devices_count = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &devices_count));

        std::vector<cl_device_id> platform_device_ids(devices_count, nullptr);
        OCL_SAFE_CALL(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, devices_count, platform_device_ids.data(), NULL));

        for (auto device_id: platform_device_ids) {
            cl_device_type device_type;
            OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_TYPE, 8, &device_type, NULL));

            switch (device_type)
            {
            case CL_DEVICE_TYPE_GPU:
                return std::make_pair(platform_id, device_id);
            case CL_DEVICE_TYPE_CPU:
                if (has_found_cpu) {
                    break;
                }
                has_found_cpu = true;
                cpu_platform_id = platform_id;
                cpu_device_id = device_id;
                break;
            default:
                break;
            }
        }
    }   

    assert(has_found_cpu);

    return std::make_pair(cpu_platform_id, cpu_device_id);
}

int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // TODO 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)

    auto device_choice = choose_device();
    cl_platform_id platform_id = device_choice.first;
    cl_device_id device_id = device_choice.second;

    size_t device_name_size = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, nullptr, &device_name_size));
    std::vector<unsigned char> device_name(device_name_size, 0);
    OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_NAME, device_name_size, device_name.data(), nullptr));
    std::cout << "Device: " << device_name.data() << std::endl;

    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        printf("create context error %d", err);
        return err;
    } 

    // Контекст и все остальные ресурсы следует освобождать с помощью clReleaseContext/clReleaseQueue/clReleaseMemObject... (да, не очень RAII, но это лишь пример)

    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач

    cl_command_queue command_queue = clCreateCommandQueue(ctx, device_id, 0, &err);
    if (err != CL_SUCCESS) {
        printf("create command queue error %d", err);
        return err;
    } 

    unsigned int n = 100*1000*1000;
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

    cl_mem as_buf = clCreateBuffer(ctx, (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), sizeof(float) * n, as.data(), &err);
    if (err != CL_SUCCESS) {
        printf("create buffer as error %d", err);
        return err;
    }

    cl_mem bs_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, bs.data(), &err);
    if (err != CL_SUCCESS) {
        printf("create buffer bs error %d", err);
        return err;
    }

    cl_mem cs_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * n, nullptr, &err);
    if (err != CL_SUCCESS) {
        printf("create buffer cs error %d", err);
        return err;
    }


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
        // std::cout << kernel_sources << std::endl;
    }

    // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель

    auto kernel_sources_chars = kernel_sources.c_str();
    cl_program program = clCreateProgramWithSource(ctx, 1, &kernel_sources_chars, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("create program from sources error %d", err);
        return err;
    }

    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram

    err = clBuildProgram(program, 1, &device_id, nullptr, NULL, nullptr);
    if (err != CL_SUCCESS) {
        if (err == CL_BUILD_PROGRAM_FAILURE) {
            size_t len = 0;
            OCL_SAFE_CALL(clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len));
            std::vector<char> build_log(len, 0);
            OCL_SAFE_CALL(clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, build_log.data(), NULL));

            printf("build failure, build log: %s", build_log.data());

            return err;
        }

        printf("build program error %d", err);
        return err;
    }

    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo
    //    size_t log_size = 0;
    //    std::vector<char> log(log_size, 0);
    //    if (log_size > 1) {
    //        std::cout << "Log:" << std::endl;
    //        std::cout << log.data() << std::endl;
    //    }

    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects

    cl_kernel kernel = clCreateKernel(program, "aplusb", &err);

    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
        unsigned int i = 0;
        clSetKernelArg(kernel, i++, sizeof(as_buf), &as_buf);
        clSetKernelArg(kernel, i++, sizeof(bs_buf), &bs_buf);
        clSetKernelArg(kernel, i++, sizeof(cs_buf), &cs_buf);
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
            size_t work_group_size = 128;

            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &workGroupSize, 0, NULL, &event));

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
        std::cout << "GFlops: " << (n / t.lapAvg()) / 1e9 << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << (3 * n * sizeof(float)) / t.lapAvg() / (1 << 30) << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(command_queue, cs_buf, CL_TRUE, 0, n * sizeof(float), cs.data(), 0, NULL, NULL));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << (n * sizeof(float)) / t.lapAvg() / (1 << 30) << " GB/s" << std::endl;
    }

    OCL_SAFE_CALL(clReleaseKernel(kernel));
    OCL_SAFE_CALL(clReleaseProgram(program));
    OCL_SAFE_CALL(clReleaseMemObject(as_buf));
    OCL_SAFE_CALL(clReleaseMemObject(bs_buf));
    OCL_SAFE_CALL(clReleaseMemObject(cs_buf));
    OCL_SAFE_CALL(clReleaseCommandQueue(command_queue));
    OCL_SAFE_CALL(clReleaseContext(ctx));

    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            printf("%f is not equal to %f", cs[i], as[i] + bs[i]);
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    return 0;
}
