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

static std::vector<unsigned char>
getProgramBuildInfo_(cl_program program, cl_device_id device, cl_program_build_info param, const char *filename, int line) {
    size_t size = 0;
    reportError(clGetProgramBuildInfo(program, device, param, 0, nullptr, &size), filename, line);
    std::vector<unsigned char> data(size);
    reportError(clGetProgramBuildInfo(program, device,param, size, data.data(), nullptr), filename, line);
    return data;
}

#define getProgramBuildInfo(program, device, param) getProgramBuildInfo_(program, device, param, __FILE__, __LINE__)

static std::vector<cl_platform_id> getPlatformIds_(const char *filename, int line) {
    cl_uint size = 0;
    reportError(clGetPlatformIDs(0, nullptr, &size), filename, line);
    std::vector<cl_platform_id> ids(size);
    reportError(clGetPlatformIDs(size, ids.data(), nullptr), filename, line);
    return ids;
}
#define getPlatformIds() getPlatformIds_(__FILE__, __LINE__)

static std::vector<cl_device_id> getDeviceIds_(cl_platform_id platform, cl_device_type type, const char *filename, int line) {
    cl_uint size = 0;
    cl_int ret = clGetDeviceIDs(platform, type, 0, nullptr, &size);
    if (ret == CL_DEVICE_NOT_FOUND) return {};
    else OCL_SAFE_CALL(ret);
    std::vector<cl_device_id> ids(size);
    reportError(clGetDeviceIDs(platform, type, size, ids.data(), nullptr), filename, line);
    return ids;
}
#define getDeviceIds(platform, type) getDeviceIds_(platform, type, __FILE__, __LINE__)

template<class T>
static T getDeviceInfoParam_(cl_device_id device, cl_device_info param, const char *filename, int line) {
    T res = 0;
    reportError(clGetDeviceInfo(device, param, sizeof(T), &res, nullptr), filename, line);
    return res;
}
#define getDeviceInfoParam(device, param, type) getDeviceInfoParam_<type>(device, param, __FILE__, __LINE__)


int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
    cl_device_id device = nullptr;
    {
        std::vector<cl_platform_id> platforms = getPlatformIds();
        for (cl_platform_id pl : platforms) {
            std::vector<cl_device_id> devs = getDeviceIds(pl, CL_DEVICE_TYPE_GPU);
            if (!devs.empty()) {
                device = devs.front();
                break;
            }
            devs = getDeviceIds(pl, CL_DEVICE_TYPE_ALL);
            if (!devs.empty()) {
                device = devs.front();
            }
        }

        auto type = getDeviceInfoParam(device, CL_DEVICE_TYPE, cl_device_type);
        if (type & CL_DEVICE_TYPE_GPU) {
            std::cout << "Using GPU device" << std::endl;
        } else if (type & CL_DEVICE_TYPE_CPU) {
            std::cout << "Using CPU device" << std::endl;
        } else if (type & CL_DEVICE_TYPE_ACCELERATOR) {
            std::cout << "Using accelerator device" << std::endl;
        } else if (type == CL_DEVICE_TYPE_DEFAULT) {
            std::cout << "Using \"default\" device" << std::endl;
        } else {
            std::cout << "Using futuristic device" << std::endl;
        }

        if (!device) {
            throw std::runtime_error("Can't find any OpenCL device");
        }
    }

    // 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    cl_context ctx;
    {
        cl_int ret_code;
        ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret_code);
        OCL_SAFE_CALL(ret_code);
    }

    // Контекст и все остальные ресурсы следует освобождать с помощью clReleaseContext/clReleaseQueue/clReleaseMemObject... (да, не очень RAII, но это лишь пример)

    // 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    cl_command_queue queue;
    {
        cl_int ret_code;
        queue = clCreateCommandQueue(ctx, device, 0, &ret_code);
        OCL_SAFE_CALL(ret_code);
    }

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

    // 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт, что чисел в каждом массиве n штук
    // Данные в as и bs можно прогрузить этим же методом, скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг, на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    cl_mem asGpu, bsGpu, csGpu;
    {
        cl_int ret_code;
        asGpu = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, as.size() * sizeof(float), as.data(), &ret_code);
        OCL_SAFE_CALL(ret_code);
        bsGpu = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bs.size() * sizeof(float), bs.data(), &ret_code);
        OCL_SAFE_CALL(ret_code);
        csGpu = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, cs.size() * sizeof(float), nullptr, &ret_code);
        OCL_SAFE_CALL(ret_code);
    }

    // 6 Выполните TO-DO 5 (реализуйте кернел в src/cl/aplusb.cl)
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

    // 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель
    cl_program prog;
    {
        cl_int ret_code;
        const char *src = kernel_sources.c_str();
        size_t len = kernel_sources.size();
        prog = clCreateProgramWithSource(ctx, 1, &src, &len, &ret_code);
        OCL_SAFE_CALL(ret_code);
    }
    // 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram
    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo
    {
        cl_int res = clBuildProgram(prog, 1, &device, "", nullptr, nullptr);
        std::vector<unsigned char> buildLog = getProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG);
        std::cout << "Build log:\n" << buildLog.data() << "\n<Build log end>\n" << std::endl;
        OCL_SAFE_CALL(res);
    }

    // 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
    cl_kernel kernel;
    {
        cl_int ret_code;
        kernel = clCreateKernel(prog, "aplusb", &ret_code);
        OCL_SAFE_CALL(ret_code);
    }

    // 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
        unsigned int i = 0;
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(asGpu), &asGpu));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(bsGpu), &bsGpu));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(csGpu), &csGpu));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(n), &n));
    }

    // 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

    // 12 Запустите выполнения кернела:
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
            cl_event ev;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, &workGroupSize, 0, nullptr, &ev));
            OCL_SAFE_CALL(clWaitForEvents(1, &ev));
            OCL_SAFE_CALL(clReleaseEvent(ev));
            t.nextLap();// При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        // 13 Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "GFlops: " << n / t.lapAvg() / 1e9 << std::endl;

        // 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << 3 * n * sizeof(float) / t.lapAvg() / (1 << 30) << " GB/s" << std::endl;
    }

    // 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(queue, csGpu, CL_TRUE, 0, cs.size() * sizeof(float), cs.data(), 0, nullptr, nullptr));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << n * sizeof(float) / t.lapAvg() / (1 << 30) << " GB/s" << std::endl;
    }

    OCL_SAFE_CALL(clReleaseKernel(kernel));
    OCL_SAFE_CALL(clReleaseProgram(prog));
    OCL_SAFE_CALL(clReleaseMemObject(asGpu));
    OCL_SAFE_CALL(clReleaseMemObject(bsGpu));
    OCL_SAFE_CALL(clReleaseMemObject(csGpu));
    OCL_SAFE_CALL(clReleaseCommandQueue(queue));
    OCL_SAFE_CALL(clReleaseContext(ctx));

    // 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    return 0;
}
