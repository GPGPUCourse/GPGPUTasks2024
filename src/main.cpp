#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/fast_random.h>
#include <libutils/timer.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <iomanip>
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

void reportErrorCustom(cl_int err, const std::string &filename, int line, const std::string &msg) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL: " + msg + " (" + to_string(err) + ") encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}
#define OCL_SAFE_CALL_MESSAGE(expr, msg) reportErrorCustom(expr, __FILE__, __LINE__, msg)

struct platformDevices {
    cl_platform_id platform;
    cl_uint n_devices;
    cl_device_id *devices;

    platformDevices(cl_platform_id platform, cl_uint n_devices, cl_device_id *devices) : platform(platform),
                                                                                         n_devices(n_devices),
                                                                                         devices(devices) {}
};

cl_context createContextFromPlatformAndDevices(cl_platform_id platform, cl_uint n_devices, cl_device_id *devices) {
    cl_int err;
    std::vector<cl_context_properties> props = {
            cl_context_properties(CL_CONTEXT_PLATFORM), cl_context_properties(platform), 0
    };
    cl_context context = clCreateContext(props.data(), n_devices, devices, nullptr, nullptr, &err);
    OCL_SAFE_CALL_MESSAGE(err, "cannot create context");
    return context;
}

platformDevices getPlatformDevicesByType(cl_device_type device_type) {

    cl_uint n_platforms = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &n_platforms));
    std::vector<cl_platform_id> platforms(n_platforms);
    OCL_SAFE_CALL(clGetPlatformIDs(n_platforms, platforms.data(), &n_platforms));

    for (cl_platform_id platform: platforms) {

        cl_uint n_found_devices = 0;
        cl_int err_code = clGetDeviceIDs(platform, device_type, 0, nullptr, &n_found_devices);
        if (err_code == CL_DEVICE_NOT_FOUND){
            continue;
        }
        OCL_SAFE_CALL(err_code);
        if (n_found_devices > 0) {
            auto *devices = new cl_device_id[n_found_devices];
            OCL_SAFE_CALL(clGetDeviceIDs(platform, device_type, n_found_devices, devices, &n_found_devices));
            return {platform, n_found_devices, devices};
        }

    }

    return {nullptr, 0, nullptr};
}

cl_command_queue createCommandQueueOnDevice(cl_context context, cl_device_id device) {
    cl_int err_command_queue;
    cl_command_queue_properties cq_props = 0;
    cl_command_queue queue = clCreateCommandQueue(context, device, cq_props, &err_command_queue);
    OCL_SAFE_CALL_MESSAGE(err_command_queue, "cannot create command queue");
    return queue;
}

cl_mem createReadBufferFromHost(cl_context context, size_t size, void *data) {
    cl_int alloc_erc;
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, data, &alloc_erc);
    OCL_SAFE_CALL_MESSAGE(alloc_erc, "cannot create read buffer");
    return buffer;
}

cl_mem createWriteBuffer(cl_context context, size_t size) {
    cl_int alloc_erc;
    cl_mem buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, nullptr, &alloc_erc);
    OCL_SAFE_CALL_MESSAGE(alloc_erc, "cannot create write buffer");
    return buffer;
}

cl_program createProgramFromFile(cl_context context, const std::string& filename){
    // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь, что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания),
    // напечатав исходники в консоль (if проверяет, что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        std::ifstream file(filename);
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
//         std::cout << kernel_sources << std::endl;
    }

    // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель

    cl_int create_erc;
    size_t ab_size = kernel_sources.size();
    const char *ab_sources = kernel_sources.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &ab_sources, &ab_size, &create_erc);
    OCL_SAFE_CALL_MESSAGE(create_erc, "cannot create program from sources");

    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram
    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo

    cl_uint n_devices;
    OCL_SAFE_CALL(clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &n_devices, nullptr));

    cl_device_id devices[n_devices];
    OCL_SAFE_CALL(clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id) * n_devices, &devices, nullptr));

    std::string build_options = "";
    try {
        OCL_SAFE_CALL(clBuildProgram(program, n_devices, devices, build_options.c_str(), nullptr, nullptr));
    } catch (std::runtime_error &e) {
        size_t log_size;
        OCL_SAFE_CALL(clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
        std::vector<char> log(log_size);
        OCL_SAFE_CALL(clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log.data(), &log_size));
        if (log_size > 1) {
            std::cout << "Log:" << std::endl;
            std::cout << log.data() << std::endl;
        }
    }

    return program;
}

cl_kernel createKernelFromProgram(cl_program program, const std::string &name, const std::vector<std::pair<size_t, void*>> &args){
    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
    cl_int kernel_create_erc;
    cl_kernel kernel = clCreateKernel(program, name.c_str(), &kernel_create_erc);
    OCL_SAFE_CALL_MESSAGE(kernel_create_erc, "cannot create kernel");


    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
//    {
//        unsigned int i = 0;
//        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &a_gpu));
//        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &b_gpu));
//        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &c_gpu));
//
//        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(unsigned int), &n));
//    }
    for (unsigned int arg_id = 0; arg_id < args.size(); ++arg_id) {
        OCL_SAFE_CALL(clSetKernelArg(kernel, arg_id, args[arg_id].first, args[arg_id].second));
    }
    return kernel;
}

int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // TODO 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
    platformDevices platform_devices = getPlatformDevicesByType(CL_DEVICE_TYPE_GPU);
    if (platform_devices.n_devices == 0) {
        std::cout << "WARN! Cannot find GPU device, switching to CPU." << std::endl;
        platform_devices = getPlatformDevicesByType(CL_DEVICE_TYPE_CPU);
        if (platform_devices.n_devices == 0) {
            throw std::runtime_error("Cannot find either GPU or CPU device, check your configuration!");
        }
    }

    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    cl_context context = createContextFromPlatformAndDevices(platform_devices.platform, platform_devices.n_devices,
                                                             platform_devices.devices);

    // Контекст и все остальные ресурсы следует освобождать с помощью clReleaseContext/clReleaseQueue/clReleaseMemObject... (да, не очень RAII, но это лишь пример)

    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    cl_command_queue queue = createCommandQueueOnDevice(context, platform_devices.devices[0]);


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

    // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт, что чисел в каждом массиве n штук
    // Данные в as и bs можно прогрузить этим же методом, скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг, на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer

    cl_mem a_gpu = createReadBufferFromHost(context, sizeof(float) * n, as.data());
    cl_mem b_gpu = createReadBufferFromHost(context, sizeof(float) * n, bs.data());
    cl_mem c_gpu = createWriteBuffer(context, sizeof(float) * n);

    cl_program ab_program = createProgramFromFile(context, "src/cl/aplusb.cl");
    cl_kernel ab_kernel = createKernelFromProgram(ab_program, "aplusb", {
            {sizeof(cl_mem), &a_gpu},
            {sizeof(cl_mem), &b_gpu},
            {sizeof(cl_mem), &c_gpu},
            {sizeof(unsigned int), &n},
    });

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
            cl_event finish_event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(queue, ab_kernel, 1, nullptr, &global_work_size, &workGroupSize, 0, nullptr, &finish_event));
            // clWaitForEvents...
            OCL_SAFE_CALL(clWaitForEvents(1, &finish_event));
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
        long double gflops = n / (long double) t.lapAvg() / powl(10, 9) ;
        std::cout << "GFlops: " << gflops << std::setprecision(4) << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        long double bandwith = 3 * n * sizeof(float) / (long double) (1 << 30) / t.lapAvg();
        std::cout << "VRAM bandwidth: " << bandwith << std::setprecision(4) << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            // clEnqueueReadBuffer...
            OCL_SAFE_CALL(clEnqueueReadBuffer(queue, c_gpu, true, 0, sizeof(float) * n, cs.data(), 0, nullptr, nullptr));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        long double bandwith = n * sizeof(float) / (long double) (1 << 30) / t.lapAvg();
        std::cout << "VRAM -> RAM bandwidth: " << bandwith << std::setprecision(4) << " GB/s" << std::endl;
    }

    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    OCL_SAFE_CALL(clReleaseKernel(ab_kernel));
    OCL_SAFE_CALL(clReleaseProgram(ab_program));
    OCL_SAFE_CALL(clReleaseMemObject(a_gpu));
    OCL_SAFE_CALL(clReleaseMemObject(b_gpu));
    OCL_SAFE_CALL(clReleaseMemObject(c_gpu));
    OCL_SAFE_CALL(clReleaseCommandQueue(queue));
    OCL_SAFE_CALL(clReleaseContext(context));

    return 0;
}
