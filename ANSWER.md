# 4.1.2

Провел эксперимент на своем ноуте на процессоре и на древней видеокарте Radeon HD 6850.

Процессор:
```
OpenCL devices:
  Device #0: CPU. 12th Gen Intel(R) Core(TM) i7-1260P. Intel(R) Corporation. Total memory: 31716 Mb
Using device #0: CPU. 12th Gen Intel(R) Core(TM) i7-1260P. Intel(R) Corporation. Total memory: 31716 Mb
Data generated for M=4096, K=4096
[matrix_transpose_naive]
    GPU: 0.00643078+-7.0743e-05 s
    GPU: 2608.89 millions/s
[matrix_transpose_local_bad_banks]
    GPU: 0.00673028+-7.79243e-05 s
    GPU: 2492.79 millions/s
[matrix_transpose_local_good_banks]
    GPU: 0.00682775+-0.000100604 s
    GPU: 2457.21 millions/s
```

Логов компиляции не вставил, но все кернелы были векторизованы до 8.

Видеокарта:
```
OpenCL devices:
  Device #0: GPU.  (Barts). Free memory: 509/512 Mb
  Device #1: CPU. Intel(R) Core(TM)2 Quad CPU    Q6600  @ 2.40GHz. GenuineIntel. Free memory: 3162/3952 Mb
Using device #0: GPU.  (Barts). Free memory: 509/512 Mb
Data generated for M=4096, K=4096
[matrix_transpose_naive]
    GPU: 0.0286214+-1.77117e-05 s
    GPU: 586.178 millions/s
[matrix_transpose_local_bad_banks]
    GPU: 0.0292156+-4.02646e-05 s
    GPU: 574.255 millions/s
[matrix_transpose_local_good_banks]
    GPU: 0.0293583+-3.53537e-05 s
    GPU: 571.464 millions/s
```

Как видим, ни одна из наших оптимизаций не дала выигрыша в производительности ни на процессоре, ни на видеокарте.
С процессором это объяснимо, так как наши оптимизации все-таки заточены под видеокарту.
Почему выигрыша нет на видеокарте - вопрос. Возможно это связано с кешами, которые за нас все соптимизировали.

# 4.2.3

Аналогично, прогоним и на процессоре и на видеокарте.

Процессор:
```
OpenCL devices:
  Device #0: CPU. 12th Gen Intel(R) Core(TM) i7-1260P. Intel(R) Corporation. Total memory: 31716 Mb
Using device #0: CPU. 12th Gen Intel(R) Core(TM) i7-1260P. Intel(R) Corporation. Total memory: 31716 Mb
Data generated for M=1024, K=1024, N=1024
CPU: 2.86558+-0 s
CPU: 0.69794 GFlops
[naive, ts=4]
    GPU: 0.132822+-0.00214371 s
    GPU: 15.0577 GFlops
    Average difference: 0.000149043%
[naive, ts=8]
    GPU: 0.122077+-0.00402763 s
    GPU: 16.383 GFlops
    Average difference: 0.000149043%
[naive, ts=16]
    GPU: 0.122976+-0.00296021 s
    GPU: 16.2634 GFlops
    Average difference: 0.000149043%
[local, ts=4]
    GPU: 0.194005+-0.00112202 s
    GPU: 10.309 GFlops
    Average difference: 0.000149043%
[local, ts=8]
    GPU: 0.11476+-0.000601609 s
    GPU: 17.4276 GFlops
    Average difference: 0.000149043%
[local, ts=16]
    GPU: 0.0940113+-0.000430166 s
    GPU: 21.274 GFlops
    Average difference: 0.000149043%
[local wpt, ts=4, wpt=2]
    GPU: 0.159422+-0.00145472 s
    GPU: 12.5453 GFlops
    Average difference: 0.000149043%
[local wpt, ts=4, wpt=4]
    GPU: 0.129037+-0.000798055 s
    GPU: 15.4994 GFlops
    Average difference: 0.000149043%
[local wpt, ts=8, wpt=2]
    GPU: 0.101135+-0.00340062 s
    GPU: 19.7755 GFlops
    Average difference: 0.000149043%
[local wpt, ts=8, wpt=4]
    GPU: 0.0864793+-0.000784055 s
    GPU: 23.1269 GFlops
    Average difference: 0.000149043%
[local wpt, ts=8, wpt=8]
    GPU: 0.075373+-0.00124269 s
    GPU: 26.5347 GFlops
    Average difference: 0.000149043%
[local wpt, ts=16, wpt=2]
    GPU: 0.0714007+-0.000784316 s
    GPU: 28.0109 GFlops
    Average difference: 0.000149043%
[local wpt, ts=16, wpt=4]
    GPU: 0.0584528+-0.000802348 s
    GPU: 34.2156 GFlops
    Average difference: 0.000149043%
[local wpt, ts=16, wpt=8]
    GPU: 0.054619+-0.000709091 s
    GPU: 36.6173 GFlops
    Average difference: 0.000149043%
[local wpt, ts=16, wpt=16]
    GPU: 0.0768372+-0.000169198 s
    GPU: 26.0291 GFlops
    Average difference: 0.000149043%
```

Видеокарта:
```
OpenCL devices:
  Device #0: GPU.  (Barts). Free memory: 509/512 Mb
  Device #1: CPU. Intel(R) Core(TM)2 Quad CPU    Q6600  @ 2.40GHz. GenuineIntel. Free memory: 3162/3952 Mb
Using device #0: GPU.  (Barts). Free memory: 509/512 Mb
Data generated for M=1024, K=1024, N=1024
CPU: 14.0569+-0 s
CPU: 0.142279 GFlops
[naive, ts=4]
    GPU: 0.49714+-0.000653084 s
    GPU: 4.02301 GFlops
    Average difference: 0%
[naive, ts=8]
    GPU: 0.147692+-0.000215901 s
    GPU: 13.5417 GFlops
    Average difference: 0%
[naive, ts=16]
    GPU: 0.146456+-0.000133221 s
    GPU: 13.656 GFlops
    Average difference: 0%
[local, ts=4]
    GPU: 0.326729+-0.000409533 s
    GPU: 6.12127 GFlops
    Average difference: 0%
[local, ts=8]
    GPU: 0.052017+-7.72442e-06 s
    GPU: 38.449 GFlops
    Average difference: 0%
[local, ts=16]
    GPU: 0.0475108+-5.58582e-05 s
    GPU: 42.0957 GFlops
    Average difference: 0%
[local wpt, ts=4, wpt=2]
    GPU: 0.516788+-5.04414e-05 s
    GPU: 3.87006 GFlops
    Average difference: 0%
[local wpt, ts=4, wpt=4]
    GPU: 0.909569+-0.000151292 s
    GPU: 2.19884 GFlops
    Average difference: 0%
[local wpt, ts=8, wpt=2]
    GPU: 0.0829727+-4.53235e-05 s
    GPU: 24.1043 GFlops
    Average difference: 0%
[local wpt, ts=8, wpt=4]
    GPU: 0.146014+-4.12408e-05 s
    GPU: 13.6973 GFlops
    Average difference: 0%
[local wpt, ts=8, wpt=8]
    GPU: 0.274839+-9.0533e-05 s
    GPU: 7.27698 GFlops
    Average difference: 0%
[local wpt, ts=16, wpt=2]
    GPU: 0.0308165+-4.81932e-05 s
    GPU: 64.9003 GFlops
    Average difference: 0%
[local wpt, ts=16, wpt=4]
    GPU: 0.0306032+-9.55762e-05 s
    GPU: 65.3527 GFlops
    Average difference: 0%
[local wpt, ts=16, wpt=8]
    GPU: 0.0512328+-0.000158381 s
    GPU: 39.0375 GFlops
    Average difference: 0%
[local wpt, ts=16, wpt=16]
    GPU: 0.163355+-0.000186296 s
    GPU: 12.2433 GFlops
    Average difference: 0%
```

## Процессор

Реализация `local` дала несущественный прирост производительности. Возможно это связано с тем, что наши обращения стали лучше попадать в кеш.

Реализация `wpt` дала существенный прирост производительности по сравнению с остальными. Это наверняка связано с тем, что мы просто делаем меньше загрузок из памяти в цикле.

## Видеокарта

Реализация `local` с `ts=8` дала существенный прирост производительности по сравнению с наивной реализацией, потому что мы начали обращаться к глобальной памяти coalesced и, наверное, потому что мы попали в размер wavefront-а.

Реализация `wpt` обогнала все предыдущие реализации (и даже опередила современный процессор!), потому что мы стали делать меньше загрузок из глобальной памяти.
