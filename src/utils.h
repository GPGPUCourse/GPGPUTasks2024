#ifndef UTILS_H
#define UTILS_H

#include <CL/cl.h>

#include <fstream>
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

    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

inline void checkErrWithMsg(cl_int err, const std::string& err_message) {
    if (CL_SUCCESS == err) {
        return;
    }

    throw std::runtime_error(err_message + ", err_code = " + to_string(err));
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

#endif // UTILS_H
