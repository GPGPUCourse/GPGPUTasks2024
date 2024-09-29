OpenCL devices:
  Device #0: CPU. AMD Ryzen 7 5800X 8-Core Processor             . Intel(R) Corporation. Total memory: 32670 Mb
  Device #1: GPU. NVIDIA GeForce RTX 4080. Total memory: 16375 Mb
Using device #0: CPU. AMD Ryzen 7 5800X 8-Core Processor             . Intel(R) Corporation. Total memory: 32670 Mb
CPU: 0.317833+-0.0131075 s
CPU: 31.463 GFlops
    Real iterations fraction: 56.2638%
Invalid Parameter - 100
Building kernels for AMD Ryzen 7 5800X 8-Core Processor             ...
Kernels compilation done in 0.18 seconds
Device 1
        Program build log:
Compilation started
Compilation done
Linking started
Linking done
Device build started
Options used by backend compiler:  -D WARP_SIZE=1
Device build done
Kernel "mandelbrot" was successfully vectorized (8)
Done.

GPU: 0.0281667+-0.000372678 s
GPU: 381.21 GFlops
    Real iterations fraction: 56.2657%
Invalid Parameter - 100
GPU vs CPU average results difference: 0.942446%