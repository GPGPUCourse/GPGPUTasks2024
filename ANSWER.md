# 3.2.6

На ноутбуке:
```
CPU:     0.118145+-0.00132885 s
CPU:     846.42 millions/s
CPU OMP: 0.035889+-0.000932792 s
CPU OMP: 2786.37 millions/s
OpenCL devices:
  Device #0: CPU. 12th Gen Intel(R) Core(TM) i7-1260P. Intel(R) Corporation. Total memory: 31716 Mb
Using device #0: CPU. 12th Gen Intel(R) Core(TM) i7-1260P. Intel(R) Corporation. Total memory: 31716 Mb
Building kernels for 12th Gen Intel(R) Core(TM) i7-1260P... 
Kernels compilation done in 0.08118 seconds
Device 1
	Program build log:
Compilation started
Compilation done
Linking started
Linking done
Device build started
Device build done
Kernel <sum_1> was successfully vectorized (8)
Kernel <sum_2> was successfully vectorized (8)
Kernel <sum_3> was successfully vectorized (8)
Kernel <sum_4> was successfully vectorized (8)
Kernel <sum_5> was successfully vectorized (8)
Done.

GPU sum_1: 2.83219+-0.0209721 s
GPU sum_1: 35.3084 millions/s
Building kernels for 12th Gen Intel(R) Core(TM) i7-1260P... 
Kernels compilation done in 0.149637 seconds
Device 1
	Program build log:
Compilation started
Compilation done
Linking started
Linking done
Device build started
Device build done
Kernel <sum_1> was successfully vectorized (8)
Kernel <sum_2> was not vectorized
Kernel <sum_3> was successfully vectorized (8)
Kernel <sum_4> was successfully vectorized (8)
Kernel <sum_5> was successfully vectorized (8)
Done.

GPU sum_2: 0.0729218+-0.00202966 s
GPU sum_2: 1371.33 millions/s
Building kernels for 12th Gen Intel(R) Core(TM) i7-1260P... 
Kernels compilation done in 0.122842 seconds
Device 1
	Program build log:
Compilation started
Compilation done
Linking started
Linking done
Device build started
Device build done
Kernel <sum_1> was successfully vectorized (8)
Kernel <sum_2> was not vectorized
Kernel <sum_3> was successfully vectorized (8)
Kernel <sum_4> was successfully vectorized (8)
Kernel <sum_5> was successfully vectorized (8)
Done.

GPU sum_3: 0.048421+-0.00115248 s
GPU sum_3: 2065.22 millions/s
Building kernels for 12th Gen Intel(R) Core(TM) i7-1260P... 
Kernels compilation done in 0.071804 seconds
Device 1
	Program build log:
Compilation started
Compilation done
Linking started
Linking done
Device build started
Device build done
Kernel <sum_1> was successfully vectorized (8)
Kernel <sum_2> was successfully vectorized (8)
Kernel <sum_3> was successfully vectorized (8)
Kernel <sum_4> was successfully vectorized (8)
Kernel <sum_5> was successfully vectorized (8)
Done.

GPU sum_4: 0.133334+-0.000164259 s
GPU sum_4: 749.995 millions/s
Building kernels for 12th Gen Intel(R) Core(TM) i7-1260P... 
Kernels compilation done in 0.091224 seconds
Device 1
	Program build log:
Compilation started
Compilation done
Linking started
Linking done
Device build started
Device build done
Kernel <sum_1> was successfully vectorized (8)
Kernel <sum_2> was successfully vectorized (8)
Kernel <sum_3> was successfully vectorized (8)
Kernel <sum_4> was successfully vectorized (8)
Kernel <sum_5> was successfully vectorized (8)
Done.

GPU sum_5: 0.178815+-0.00155412 s
GPU sum_5: 559.237 millions/s
```

В CI:
```
CPU:     0.0324283+-0.000159266 s
CPU:     3083.72 millions/s
CPU OMP: 0.0177983+-5.16484e-05 s
CPU OMP: 5618.5 millions/s
OpenCL devices:
  Device #0: CPU. AMD EPYC 7763 64-Core Processor                . Intel(R) Corporation. Total memory: 15991 Mb
Using device #0: CPU. AMD EPYC 7763 64-Core Processor                . Intel(R) Corporation. Total memory: 15991 Mb
Building kernels for AMD EPYC 7763 64-Core Processor                ... 
Kernels compilation done in 0.108124 seconds
Device 1
	Program build log:
Compilation started
Compilation done
Linking started
Linking done
Device build started
Device build done
Kernel <sum_1> was successfully vectorized (8)
Kernel <sum_2> was successfully vectorized (8)
Kernel <sum_3> was successfully vectorized (8)
Kernel <sum_4> was successfully vectorized (8)
Kernel <sum_5> was successfully vectorized (8)
Done.

GPU sum_1: 1.52233+-0.00544297 s
GPU sum_1: 65.6889 millions/s
Building kernels for AMD EPYC 7763 64-Core Processor                ... 
Kernels compilation done in 0.175792 seconds
Device 1
	Program build log:
Compilation started
Compilation done
Linking started
Linking done
Device build started
Device build done
Kernel <sum_1> was successfully vectorized (8)
Kernel <sum_2> was not vectorized
Kernel <sum_3> was successfully vectorized (8)
Kernel <sum_4> was successfully vectorized (8)
Kernel <sum_5> was successfully vectorized (8)
Done.

GPU sum_2: 0.0299032+-4.79528e-05 s
GPU sum_2: 3344.13 millions/s
Building kernels for AMD EPYC 7763 64-Core Processor                ... 
Kernels compilation done in 0.175784 seconds
Device 1
	Program build log:
Compilation started
Compilation done
Linking started
Linking done
Device build started
Device build done
Kernel <sum_1> was successfully vectorized (8)
Kernel <sum_2> was not vectorized
Kernel <sum_3> was successfully vectorized (8)
Kernel <sum_4> was successfully vectorized (8)
Kernel <sum_5> was successfully vectorized (8)
Done.

GPU sum_3: 0.0238452+-2.22067e-05 s
GPU sum_3: 4193.72 millions/s
Building kernels for AMD EPYC 7763 64-Core Processor                ... 
Kernels compilation done in 0.109795 seconds
Device 1
	Program build log:
Compilation started
Compilation done
Linking started
Linking done
Device build started
Device build done
Kernel <sum_1> was successfully vectorized (8)
Kernel <sum_2> was successfully vectorized (8)
Kernel <sum_3> was successfully vectorized (8)
Kernel <sum_4> was successfully vectorized (8)
Kernel <sum_5> was successfully vectorized (8)
Done.

GPU sum_4: 0.0619408+-0.000149131 s
GPU sum_4: 1614.44 millions/s
Building kernels for AMD EPYC 7763 64-Core Processor                ... 
Kernels compilation done in 0.110114 seconds
Device 1
	Program build log:
Compilation started
Compilation done
Linking started
Linking done
Device build started
Device build done
Kernel <sum_1> was successfully vectorized (8)
Kernel <sum_2> was successfully vectorized (8)
Kernel <sum_3> was successfully vectorized (8)
Kernel <sum_4> was successfully vectorized (8)
Kernel <sum_5> was successfully vectorized (8)
Done.

GPU sum_5: 0.267096+-0.000605363 s
GPU sum_5: 374.397 millions/s
```

## sum_1

Здесь самые плохие результаты у всех, так как большая часть работы такой программы это атомарная запись в одну и ту же ячейку памяти.
И это ожидаемо.

## sum_2

С этим кодом все обстоит лучше, на CI он даже справляется почти также хорошо как OpenMP.

Однако интересно, что во всех окружениях (ноутбук и CI) код не векторизовался.
Возникает вопрос, что стало причиной столь хорошей производительности?
Возможно рантайм под процессор умеет распараллеливать выполнение не только на уровне инструкций, но и на уровне потоков.
Эта гипотеза подтверждается тем, что у меня на ноутбуке рантайм породил 16 потоков (как раз по количеству ядер) при исполнении кернела.

## sum_3

И на моем ноубуке, и на CI производительность резко возросла по сравнению с sum_2, но все равно не дотянула до OpenMP.

Из логов компиляции можно сделать вывод, что coelesced доступ к памяти позволил рантайму таки векторизовать кернел.
Из-за этого и рост в производительности.

До OpenMP не дотянули видимо потому что есть атомарная запись в одну и ту же ячейку памяти.

## sum_4

Во всех окружениях здесь производительность хуже даже чем в sum_2.

Лично мне не очень понятно, что такое локальное память в случае CPU рантайма для OpenCL.
Вероятно рантайм и для локальной и для глобальной памяти использует RAM (потому что а что еще?).
В связи с этим мне кажется, что мы здесь потратили время на перекачивание данных из RAM в нее же.

## sum_5

Данный кернел справился еще хуже, чем sum_4 во всех окружениях.

Частично его производительности плоха видимо по той же причине, что и в sum_4.
Вопрос, почему он хуже sum_4?
Возможно дело в количестве барьеров.
В случае CPU они теоретически могут значительно повлиять на производительность, так как у нас нет варпов по 32 потока, у нас есть только векторные инструкции по 8 интов (а их тоже надо как-то синхронизировать).

