//
// Created by dmitriiart on 22.09.2024.
//
#pragma once

#include <CL/cl.h>

#include <sstream>
#include <stdexcept>


template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

inline void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)
#define OCL_CHECK_CALL_RET(ret) reportError(ret, __FILE__, __LINE__); ret = 0


inline std::string getDeviceType(cl_device_type deviceType) {
    switch (deviceType) {
        case CL_DEVICE_TYPE_CPU: return "CPU";
        case CL_DEVICE_TYPE_GPU: return "GPU";
        case CL_DEVICE_TYPE_ACCELERATOR: return "ACCELERATOR";
        default: return "UNKNOWN(" + std::to_string(deviceType) + ")";
    }
}

template<class RetType>
RetType getPlatformProperty(cl_platform_id platform, cl_platform_info paramName) {
    RetType paramValue;
    OCL_SAFE_CALL(clGetPlatformInfo(platform, paramName, sizeof(paramValue), &paramValue, nullptr));
    return paramValue;
}

template<>
inline std::string getPlatformProperty<std::string>(cl_platform_id platform, cl_platform_info paramName) {
    size_t paramSize = 0;
    OCL_SAFE_CALL(clGetPlatformInfo(platform, paramName, 0, nullptr, &paramSize));
    std::vector <unsigned char> paramValue(paramSize, 0);
    OCL_SAFE_CALL(clGetPlatformInfo(platform, paramName, paramValue.size() * sizeof(paramValue[0]), paramValue.data(), nullptr));

    return std::string{paramValue.begin(), paramValue.end()};
}


template<class RetType>
RetType getDeviceProperty(cl_device_id device, cl_device_info paramName) {
    RetType paramValue;
    OCL_SAFE_CALL(clGetDeviceInfo(device, paramName, sizeof(paramValue), &paramValue, nullptr));

    return paramValue;
}

template<>
std::string getDeviceProperty<std::string>(cl_device_id device, cl_device_info paramName) {
    size_t paramSize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, paramName, 0, nullptr, &paramSize));
    std::vector <unsigned char> paramValue(paramSize, 0);
    OCL_SAFE_CALL(clGetDeviceInfo(device, paramName, paramValue.size() * sizeof(paramValue[0]), paramValue.data(), nullptr));

    return std::string{paramValue.begin(), paramValue.end()};
}



/**
 * @return device inside a `platform` with type of specified inside a `priorityDeviceTypes`.
 * @throws if no such devices exist, `std::runtime_error` is thrown.
 */
inline cl_device_id getDeviceFromPlatform(cl_platform_id platform, cl_device_type priorityDeviceTypes) {
    // Finding a prioritized device type (if no such type, then picking any)
    cl_uint devicesCount = 0;
    OCL_SAFE_CALL(clGetDeviceIDs(platform, priorityDeviceTypes, 0, nullptr, &devicesCount));

    if (devicesCount == 0) {
        throw std::runtime_error(
            getPlatformProperty<std::string>(platform, CL_PLATFORM_NAME) +
            ": No OpenCL devices found with type " +
            getDeviceType(priorityDeviceTypes)
        );
    }

    std::vector<cl_device_id> devices(devicesCount);
    OCL_SAFE_CALL(clGetDeviceIDs(platform, priorityDeviceTypes, devicesCount, devices.data(), nullptr));

    return devices[0];
}




