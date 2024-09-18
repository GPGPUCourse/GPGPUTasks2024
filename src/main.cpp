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
#define OCL_SAFE_CALL2(expr) { cl_int err; expr; reportError(err, __FILE__, __LINE__) } void(0)

template<typename F>
class DeferredCall {
    F f;

public:
    explicit DeferredCall(F f) : f(f) {}
    ~DeferredCall() { f(); }
};

template<typename F>
DeferredCall<F> makeDeferred(F f) {
    return DeferredCall<F>(f);
}

#define token_paste(a, b) a ## b
#define token_paste2(a, b) token_paste(a, b)
#define defer(code) auto token_paste2(_super_magic_deferred_, __LINE__) = makeDeferred([&](){code;})

int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // TODO 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
    cl_uint platform_count = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platform_count));
    std::vector<cl_platform_id> platforms(platform_count);
    OCL_SAFE_CALL(clGetPlatformIDs(platform_count, platforms.data(), nullptr));
    std::vector<cl_device_id> cpus;
    std::vector<cl_device_id> gpus;
    for (int platform_index = 0; platform_index < platform_count; ++platform_index) {
        cl_uint device_count = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count));

        if (device_count == 0)
            continue;

        std::vector<cl_device_id> devices(device_count);
        OCL_SAFE_CALL(clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr));

        for (int device_index = 0; device_index < device_count; ++device_index) {
            cl_device_id device = devices[device_index];

            cl_bool available;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(available), &available, nullptr));
            if (!available) {
                continue;
            }

            cl_device_type device_type;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, nullptr));
            if (device_type & CL_DEVICE_TYPE_CPU) {
                cpus.push_back(device);
            }
            if (device_type & CL_DEVICE_TYPE_GPU) {
                gpus.push_back(device);
            }
        }
    }

    std::cout << "Found " << cpus.size() << " CPUs and " << gpus.size() << " GPUs" << std::endl;

    cl_device_id device;
    if (!gpus.empty()) {
        device = gpus[0];
        std::cout << "Using GPU: ";
    } else if (!cpus.empty()) {
        device = cpus[0];
        std::cout << "Using CPU: ";
    } else {
        throw std::runtime_error("No devices found!");
    }

    size_t device_name_size = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &device_name_size));
    std::vector<char> device_name(device_name_size);
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, device_name_size, device_name.data(), nullptr));
    std::cout << device_name.data() << std::endl;

    cl_platform_id platform;
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, nullptr));

    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)

    // Контекст и все остальные ресурсы следует освобождать с помощью clReleaseContext/clReleaseQueue/clReleaseMemObject... (да, не очень RAII, но это лишь пример)

    cl_int errcode_ret = 0;
    cl_context ctx = clCreateContext(nullptr, 1, &device, [](const char errinfo[], const void *, size_t, void *) {
        std::cerr << "OpenCL error: " << errinfo << std::endl;
    }, nullptr, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    defer(clReleaseContext(ctx));

    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    defer(clReleaseCommandQueue(queue));

    // unsigned int n = 1000 * 1000;
    unsigned int n = 1000 * 1000 * 100;
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

    std::size_t buf_size = n * sizeof(float);

    // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт, что чисел в каждом массиве n штук
    // Данные в as и bs можно прогрузить этим же методом, скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг, на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    cl_mem a_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size, as.data(), &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    defer(clReleaseMemObject(a_buf));

    cl_mem b_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size, bs.data(), &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    defer(clReleaseMemObject(b_buf));

    cl_mem c_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, buf_size, nullptr, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    defer(clReleaseMemObject(c_buf));

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
    const char *kernel_sources_ptr = kernel_sources.c_str();
    cl_program program = clCreateProgramWithSource(ctx, 1, &kernel_sources_ptr, nullptr, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    defer(clReleaseProgram(program));

    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram
    cl_int build_error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo
    if (build_error == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size = 0;
        OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
        std::vector<char> log(log_size, 0);
        OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));
        std::cout << "Log:" << std::endl;
        std::cout << log.data() << std::endl;
        return 1;
    } else {
        OCL_SAFE_CALL(build_error);
    }

    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
    cl_kernel kernel = clCreateKernel(program, "aplusb", &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    defer(clReleaseKernel(kernel));

    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
        unsigned int i = 0;
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem *), &a_buf));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem *), &b_buf));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem *), &c_buf));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_uint), &n));
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
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, &workGroupSize, 0, nullptr, &event));
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
        std::cout << "GFlops: " << (double)n / t.lapAvg() / 1e9 << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << (double)(3 * buf_size) / (1024 * 1024 * 1024) / t.lapAvg() << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            // clEnqueueReadBuffer...
            OCL_SAFE_CALL(clEnqueueReadBuffer(queue, c_buf, CL_TRUE, 0, buf_size, cs.data(), 0, nullptr, nullptr));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << (double)buf_size / (1024 * 1024 * 1024) / t.lapAvg() << " GB/s" << std::endl;
    }

    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    return 0;
}
