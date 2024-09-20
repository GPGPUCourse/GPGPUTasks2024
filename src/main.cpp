#include <CL/cl.h>
#include <libclew/ocl_init.h>

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
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // Откройте
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
    // Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
    // Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    // Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        // Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
        // Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
        size_t platformNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
        // TODO 1.1
        // Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
        // Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
        // Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
        // Откройте таблицу с кодами ошибок:
        // libs/clew/CL/cl.h:103
        // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
        // Найдите там нужный код ошибки и ее название
        // Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
        // в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
        // Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

        // TODO 1.2
        // Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        std::vector<unsigned char> platformName(platformNameSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), NULL));
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        size_t platformVendorSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
        std::vector<unsigned char> platformVendor(platformVendorSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), NULL));
        std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &devicesCount));

        std::vector<cl_device_id> platform_devices_id(devicesCount, nullptr);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, platform_devices_id.data(), NULL));
        std::cout << "    Devices count: " << devicesCount << std::endl;

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {

            // TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            // - Тип устройства (видеокарта/процессор/что-то странное)
            // - Размер памяти устройства в мегабайтах
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными

            cl_device_id cur_id = platform_devices_id[deviceIndex];

            std::cout << "    Device ID: " << cur_id << std::endl;
            
            size_t device_name_size = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(cur_id, CL_DEVICE_NAME, 0, nullptr, &device_name_size));
            std::vector<unsigned char> device_name(device_name_size, 0);
            OCL_SAFE_CALL(clGetDeviceInfo(cur_id, CL_DEVICE_NAME, device_name_size, device_name.data(), nullptr));
            std::cout << "        Device name: " << device_name.data() << std::endl;

            cl_device_type device_type;
            OCL_SAFE_CALL(clGetDeviceInfo(cur_id, CL_DEVICE_TYPE, 8, &device_type, NULL));

            std::string device_type_name = "";
            switch (device_type)
            {
            case CL_DEVICE_TYPE_GPU:
                device_type_name = "GPU";
                break;
            case CL_DEVICE_TYPE_CPU:
                device_type_name = "CPU";
                break;
            case CL_DEVICE_TYPE_ACCELERATOR:
                device_type_name = "ACCELERATOR";
                break;
            }
            std::cout << "        Device type: " << device_type_name << std::endl;

            cl_ulong global_mem_size_in_bytes;
            OCL_SAFE_CALL(clGetDeviceInfo(cur_id, CL_DEVICE_GLOBAL_MEM_SIZE, 8, &global_mem_size_in_bytes, NULL));
            std::cout << "        Device global memory size: " << global_mem_size_in_bytes / 1000000 << std::endl;

            cl_uint compute_units_max;
            OCL_SAFE_CALL(clGetDeviceInfo(cur_id, CL_DEVICE_MAX_COMPUTE_UNITS, 4, &compute_units_max, NULL));
            std::cout << "        Device max compute units: " << compute_units_max << std::endl;

            size_t max_work_group_size;
            OCL_SAFE_CALL(clGetDeviceInfo(cur_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, 8, &max_work_group_size, NULL));

            std::cout << "        Device max work group size: " << max_work_group_size << std::endl;
        }
    }

    return 0;
}