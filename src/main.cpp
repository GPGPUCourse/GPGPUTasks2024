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

static std::vector<uint8_t>
getPlatformInfo_(cl_platform_id platform, cl_platform_info param, const char *filename, int line) {
    size_t size = 0;
    reportError(clGetPlatformInfo(platform, param, 0, nullptr, &size), filename, line);
    std::vector<uint8_t> data(size);
    reportError(clGetPlatformInfo(platform, param, size, data.data(), nullptr), filename, line);
    return data;
}

#define getPlatformInfo(platform, param) getPlatformInfo_(platform, param, __FILE__, __LINE__)

static std::vector<uint8_t>
getDeviceInfo_(cl_device_id device, cl_device_info param, const char *filename, int line) {
    size_t size = 0;
    reportError(clGetDeviceInfo(device, param, 0, nullptr, &size), filename, line);
    std::vector<uint8_t> data(size);
    reportError(clGetDeviceInfo(device, param, size, data.data(), nullptr), filename, line);
    return data;
}

#define getDeviceInfo(device, param) getDeviceInfo_(device, param, __FILE__, __LINE__)


template<class T>
static T getDeviceInfoParam_(cl_device_id device, cl_device_info param, const char *filename, int line) {
    T res = 0;
    reportError(clGetDeviceInfo(device, param, sizeof(T), &res, nullptr), filename, line);
    return res;
}

#define getDeviceInfoParam(type, device, param) getDeviceInfoParam_<type>(device, param, __FILE__, __LINE__)


static std::string map_device_type(cl_device_type type) {
    bool sep = false;
    std::string res;

#define LOCAL_SEP \
    if (sep) res.push_back('|'); \
    else sep = true;

    if (type & CL_DEVICE_TYPE_CPU) {
        LOCAL_SEP
        res += "CPU";
    }
    if (type & CL_DEVICE_TYPE_GPU) {
        LOCAL_SEP
        res += "GPU";
    }
    if (type & CL_DEVICE_TYPE_ACCELERATOR) {
        LOCAL_SEP
        res += "Accelerator";
    }
    if (!sep) {
        res = "Custom";
    }

    // no CL_DEVICE_TYPE_CUSTOM, old cl.h header?

    return res;

#undef LOCAL_SEP
}

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
        // CL_INVALID_VALUE

        // Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        std::vector<unsigned char> platformName(platformNameSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        // Запросите и напечатайте так же в консоль вендора данной платформы
        std::cout << "    Platform vendor: " << getPlatformInfo(platform, CL_PLATFORM_VENDOR).data() << std::endl;

        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));


        const char *tab = "    ";
        std::cout << tab << "Number of devices: " << devicesCount << "\n";
        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            // Запросите и напечатайте в консоль:
            // - Название устройства
            // - Тип устройства (видеокарта/процессор/что-то странное)
            // - Размер памяти устройства в мегабайтах
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            std::cout << tab << "Device #" << (deviceIndex + 1) << "/" << devicesCount << "\n";

            std::cout << tab << tab << "Device name: " << getDeviceInfo(devices[deviceIndex], CL_DEVICE_NAME).data()
                      << "\n";

            std::cout << tab << tab << "Device type: "
                      << map_device_type(getDeviceInfoParam(cl_device_type, devices[deviceIndex], CL_DEVICE_TYPE))
                      << "\n";

            std::cout << tab << tab << "Device memory: " <<
                      static_cast<double>(getDeviceInfoParam(cl_ulong, devices[deviceIndex],
                                                             CL_DEVICE_GLOBAL_MEM_SIZE)) / 1024.0 / 1024.0
                      << "MB\n";

            std::cout << tab << tab << "Is device little endian: " << std::boolalpha
                      << static_cast<bool>(getDeviceInfoParam(cl_bool, devices[deviceIndex], CL_DEVICE_ENDIAN_LITTLE))
                      << "\n";

            std::cout << tab << tab << "Max supported OpenCL C version: "
                      << getDeviceInfo(devices[deviceIndex], CL_DEVICE_OPENCL_C_VERSION).data() << "\n";

            std::cout << tab << tab << "Max work group size: "
                      << getDeviceInfoParam(size_t, devices[deviceIndex], CL_DEVICE_MAX_WORK_GROUP_SIZE) << "\n";
        }
        std::cout.flush();
    }

    return 0;
}
