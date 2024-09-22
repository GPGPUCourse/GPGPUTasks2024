#include <iostream>
#include <stdlib.h>
#include <vector>
#include <CL/cl.h>

using namespace std;

int main() {
    //  the number of platforms
    cl_uint platformCount = 0;
    cl_int result = clGetPlatformIDs(0, nullptr, &platformCount);
    if (result != CL_SUCCESS) {
        std::cerr << "Error getting platform count: " << result << std::endl;
        return -1;
    }

    // номер платформ
    std::cout << "Number of OpenCL platforms: " << platformCount << std::endl;

    // если платформа есть то вывод IDs и имени
    if (platformCount > 0) {
        cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
        result = clGetPlatformIDs(platformCount, platforms, nullptr);
        if (result != CL_SUCCESS) {
            std::cerr << "Error getting platform IDs: " << result << std::endl;
            free(platforms);
            return -1;
        }

        // Вывести платформу ID и имя
        for (cl_uint i = 0; i < platformCount; ++i) {
            std::cout << "Platform #" << (i + 1) << ": " << platforms[i] << std::endl;

            // получить длину имени платформы
            size_t platformNameSize;
            result = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, nullptr, &platformNameSize);
            if (result != CL_SUCCESS) {
                std::cerr << "Error getting platform name size: " << result << std::endl;
                continue;
            }


            std::vector<char> platformName(platformNameSize);
            result = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr);
            if (result != CL_SUCCESS) {
                std::cerr << "Error getting platform name: " << result << std::endl;
                continue;
            }

            // имя платформы
            std::cout << "    Platform name: " << platformName.data() << std::endl;

            // вендор и версия
            size_t platformVendorSize;
            result = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize);
            if (result != CL_SUCCESS) {
                std::cerr << "Error getting platform vendor size: " << result << std::endl;
                continue;
            }
            std::vector<char> platformVendor(platformVendorSize);
            result = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr);
            if (result != CL_SUCCESS) {
                std::cerr << "Error getting platform vendor: " << result << std::endl;
                continue;
            }
            std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

            size_t platformVersionSize;
            result = clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 0, nullptr, &platformVersionSize);
            if (result != CL_SUCCESS) {
                std::cerr << "Error getting platform version size: " << result << std::endl;
                continue;
            }
            std::vector<char> platformVersion(platformVersionSize);
            result = clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, platformVersionSize, platformVersion.data(), nullptr);
            if (result != CL_SUCCESS) {
                std::cerr << "Error getting platform version: " << result << std::endl;
                continue;
            }
            std::cout << "    Platform version: " << platformVersion.data() << std::endl;

            // количество устройств платформы
            cl_uint devicesCount = 0;
            result = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount);
            if (result != CL_SUCCESS) {
                std::cerr << "Error getting device count: " << result << std::endl;
                continue;
            }

            // количество устройств
            std::cout << "    Number of devices: " << devicesCount << std::endl;

            if (devicesCount > 0) {
                cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * devicesCount);
                result = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, devicesCount, devices, nullptr);
                if (result != CL_SUCCESS) {
                    std::cerr << "Error getting device IDs: " << result << std::endl;
                    free(devices);
                    continue;
                }

                // данные об устройстве
                for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
                    // Get the device name
                    size_t deviceNameSize;
                    result = clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_NAME, 0, nullptr, &deviceNameSize);
                    if (result != CL_SUCCESS) {
                        std::cerr << "Error getting device name size: " << result << std::endl;
                        continue;
                    }
                    std::vector<char> deviceName(deviceNameSize);
                    result = clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr);
                    if (result != CL_SUCCESS) {
                        std::cerr << "Error getting device name: " << result << std::endl;
                        continue;
                    }
                    std::cout << "        Device #" << (deviceIndex + 1) << ": " << deviceName.data() << std::endl;

                    // получить тип устройства
                    cl_device_type deviceType;
                    result = clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr);
                    if (result != CL_SUCCESS) {
                        std::cerr << "Error getting device type: " << result << std::endl;
                        continue;
                    }
                    std::string deviceTypeStr;
                    if (deviceType & CL_DEVICE_TYPE_CPU) deviceTypeStr += "CPU ";
                    if (deviceType & CL_DEVICE_TYPE_GPU) deviceTypeStr += "GPU ";
                    if (deviceType & CL_DEVICE_TYPE_ACCELERATOR) deviceTypeStr += "Accelerator ";
                    if (deviceTypeStr.empty()) deviceTypeStr = "Unknown";
                    std::cout << "            Device type: " << deviceTypeStr << std::endl;

                    // получить размер памяти в мб
                    cl_ulong globalMemSize;
                    result = clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, nullptr);
                    if (result != CL_SUCCESS) {
                        std::cerr << "Error getting global memory size: " << result << std::endl;
                        continue;
                    }
                    std::cout << "            Global memory size: " << globalMemSize / (1024 * 1024) << " MB" << std::endl;

                    // получить max модулей
                    cl_uint maxComputeUnits;
                    result = clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, nullptr);
                    if (result != CL_SUCCESS) {
                        std::cerr << "Error getting maximum compute units: " << result << std::endl;
                        continue;
                    }
                    std::cout << "            Maximum compute units: " << maxComputeUnits << std::endl;


                    cl_uint maxClockFrequency;
                    result = clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxClockFrequency), &maxClockFrequency, nullptr);
                    if (result != CL_SUCCESS) {
                        std::cerr << "Error getting maximum clock frequency: " << result << std::endl;
                        continue;
                    }
                    std::cout << "            Maximum clock frequency: " << maxClockFrequency << " MHz" << std::endl;
                }

                free(devices);
            }
        }

        free(platforms);
    }

    return 0;
}
