OpenCL devices:
  Device #0: CPU. 11th Gen Intel(R) Core(TM) i7-11700K @ 3.60GHz. Intel(R) Corporation. Total memory: 32573 Mb
  Device #1: GPU. Intel(R) UHD Graphics 750. Total memory: 13029 Mb
  Device #2: GPU. NVIDIA GeForce RTX 3070. Total memory: 8191 Mb
Using device #2: GPU. NVIDIA GeForce RTX 3070. Total memory: 8191 Mb
Data generated for M=1024, K=1024, N=1024
CPU: 2.515+-0 s
CPU: 0.795229 GFlops
[naive, ts=4]
    GPU: 0.00766667+-0.000471405 s
    GPU: 260.87 GFlops
    Average difference: 0.0138686%
[naive, ts=8]
    GPU: 0.00266667+-0.000471405 s
    GPU: 750 GFlops
    Average difference: 0.0138686%
[naive, ts=16]
    GPU: 0.0025+-0.0005 s
    GPU: 800 GFlops
    Average difference: 0.0138686%
[local, ts=4]
    GPU: 0.00633333+-0.000471405 s
    GPU: 315.789 GFlops
    Average difference: 0.000196008%
[local, ts=8]
    GPU: 0.002+-0 s
    GPU: 1000 GFlops
    Average difference: 0.000196008%
[local, ts=16]
    GPU: 0.00116667+-0.000372678 s
    GPU: 1714.29 GFlops
    Average difference: 0.000196008%
[local wpt, ts=4, wpt=2]
    GPU: 0.00716667+-0.000372678 s
    GPU: 279.07 GFlops
    Average difference: 0.000196008%
[local wpt, ts=4, wpt=4]
    GPU: 0.0106667+-0.000471405 s
    GPU: 187.5 GFlops
    Average difference: 0.000196008%
[local wpt, ts=8, wpt=2]
    GPU: 0.00133333+-0.000471405 s
    GPU: 1500 GFlops
    Average difference: 0.000196008%
[local wpt, ts=8, wpt=4]
    GPU: 0.002+-0 s
    GPU: 1000 GFlops
    Average difference: 0.000196008%
[local wpt, ts=8, wpt=8]
    GPU: 0.003+-4.1159e-11 s
    GPU: 666.667 GFlops
    Average difference: 0.000196008%
[local wpt, ts=16, wpt=2]
    GPU: 0.001+-0 s
    GPU: 2000 GFlops
    Average difference: 0.000196008%
[local wpt, ts=16, wpt=4]
    GPU: 0.000833333+-0.000372678 s
    GPU: 2400 GFlops
    Average difference: 0.000196008%
[local wpt, ts=16, wpt=8]
    GPU: 0.000666667+-0.000471405 s
    GPU: 3000 GFlops
    Average difference: 0.000196008%
[local wpt, ts=16, wpt=16]
    GPU: 0.001+-0 s
    GPU: 2000 GFlops
    Average difference: 0.000196008%



OpenCL devices:
  Device #0: CPU. 11th Gen Intel(R) Core(TM) i7-11700K @ 3.60GHz. Intel(R) Corporation. Total memory: 32573 Mb
  Device #1: GPU. Intel(R) UHD Graphics 750. Total memory: 13029 Mb
  Device #2: GPU. NVIDIA GeForce RTX 3070. Total memory: 8191 Mb
Using device #2: GPU. NVIDIA GeForce RTX 3070. Total memory: 8191 Mb
Data generated for M=4096, K=4096
[matrix_transpose_naive]
    GPU: 0.00045+-0.000497494 s
    GPU: 37282.7 millions/s
[matrix_transpose_local_bad_banks]
    GPU: 0.00035+-0.00047697 s
    GPU: 47934.9 millions/s
[matrix_transpose_local_good_banks]
    GPU: 0.00035+-0.00047697 s
    GPU: 47934.9 millions/s