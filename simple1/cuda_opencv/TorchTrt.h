#ifndef TORCHTRT_H
#define TORCHTRT_H

#include <string>

class TorchTrt
{
public:
    TorchTrt();

    static void test1(const std::string& trt_ts_module_path);
};

#endif // TORCHTRT_H
