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

    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


int main() {
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        size_t platformNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));

        // 1.1
        //OCL_SAFE_CALL(clGetPlatformInfo(platform, 239, 0, nullptr, &platformNameSize));
        // Найдите там нужный код ошибки и ее название:
        // -30 CL_INVALID_VALUE

        std::vector<unsigned char> platformName(platformNameSize, 0);

        // 1.2
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));

        std::cout << "    Platform name: " << platformName.data() << std::endl;

        // 1.3
        size_t vendorNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorNameSize));
        std::vector<char> vendorName(vendorNameSize);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorNameSize, vendorName.data(), nullptr));
        std::cout << "    Vendor: " << vendorName.data() << std::endl;

        // 2.1
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::cout << "    Number of devices: " << devicesCount << std::endl;


        // 2.2
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            std::cout << "    Device ID: " << deviceIndex << std::endl;

            // - Название устройства
            size_t deviceNameSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
            std::vector<char> deviceName(deviceNameSize);
            OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
            std::cout << "        Device name: " << deviceName.data() << std::endl;

            // - Тип устройства (видеокарта/процессор/что-то странное)
            size_t deviceTypeSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_TYPE, 0, nullptr, &deviceTypeSize));
            cl_device_type deviceType = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_TYPE, deviceTypeSize, &deviceType, nullptr));
            if (deviceType == CL_DEVICE_TYPE_DEFAULT) {
                std::cout << "        Device type: DEFAULT" << std::endl;
            } else if (deviceType == CL_DEVICE_TYPE_CPU ) {
                std::cout << "        Device type: CPU" << std::endl;
            } else if (deviceType == CL_DEVICE_TYPE_GPU ) {
                std::cout << "        Device type: GPU" << std::endl;
            } else if (deviceType == CL_DEVICE_TYPE_ACCELERATOR ) {
                std::cout << "        Device type: ACCELERATOR" << std::endl;
            }

            // - Размер памяти устройства в мегабайтах
            size_t deviceMemSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_GLOBAL_MEM_SIZE, 0, nullptr, &deviceMemSize));
            cl_long deviceGlobalMemSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_GLOBAL_MEM_SIZE, deviceMemSize, &deviceGlobalMemSize, nullptr));
            std::cout << "        Device global memory size: " << deviceGlobalMemSize / 1048576  << " MB"<< std::endl;

            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            // - Тип кэша устройства
            size_t deviceCacheTypeSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, 0, nullptr, &deviceCacheTypeSize));
            cl_device_type deviceCacheType = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, deviceCacheTypeSize, &deviceCacheType, nullptr));
            if (deviceCacheType == CL_READ_ONLY_CACHE) {
                std::cout << "        Device global memory cache type: Read-Only" << std::endl;
            } else if (deviceCacheType == CL_READ_WRITE_CACHE ) {
                std::cout << "        Device global memory cache type: Write" << std::endl;
            }
            // - Размер линии кэша
            size_t deviceCLSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, 0, nullptr, &deviceCLSize));
            cl_long deviceCacheLineSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, deviceCLSize, &deviceCacheLineSize, nullptr));
            std::cout << "        Device global cache-line size: " << deviceCacheLineSize << " B"
            << std::endl;
        }
    }

    return 0;
}
