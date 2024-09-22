#ifdef __CLION_IDE__
    // Этот include виден только для CLion парсера, это позволяет IDE "знать" ключевые слова вроде __kernel, __global
    // а также уметь подсказывать OpenCL методы, описанные в данном инклюде (такие как get_global_id(...) и get_local_id(...))
    #include "clion_defines.cl"
#endif

#line 8 // Седьмая строчка теперь восьмая (при ошибках компиляции в логе компиляции будут указаны корректные строчки благодаря этой директиве)

// 5 Реализуем кернел:
__kernel void aplusb(__global const float* as, __global const float* bs, __global float* cs, unsigned int n) {
    unsigned int gid = get_global_id(0);

    if (gid < n) {
        cs[gid] = as[gid] + bs[gid];
    }
}
