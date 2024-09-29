# Логи 1

```

CPU:     0.220333+-0.000942809 s
CPU:     453.858 millions/s
CPU OMP: 0.032+-0.000816497 s
CPU OMP: 3125 millions/s
OpenCL devices:
  Device #0: CPU. AMD Ryzen 7 5800X 8-Core Processor             . Intel(R) Corporation. Total memory: 32670 Mb
  Device #1: GPU. NVIDIA GeForce RTX 4080. Total memory: 16375 Mb
Using device #1: GPU. NVIDIA GeForce RTX 4080. Total memory: 16375 Mb
GPU (sum_baseline): 0.00166667 +-0.000471405 s
GPU (sum_baseline): 60000 millions/s
GPU (sum_cycle): 0.001 +-0 s
GPU (sum_cycle): 100000 millions/s
GPU (sum_cycle_coalesced): 0.000666667 +-0.000471405 s
GPU (sum_cycle_coalesced): 150000 millions/s
GPU (sum_local_memory): 0.000833333 +-0.000372678 s
GPU (sum_local_memory): 120000 millions/s
GPU (sum_tree): 0.001 +-0 s
GPU (sum_tree): 100000 millions/s

```

# Логи 2

```

CPU:     0.222833+-0.000897527 s
CPU:     448.766 millions/s
CPU OMP: 0.032+-0 s
CPU OMP: 3125 millions/s
OpenCL devices:
  Device #0: CPU. AMD Ryzen 7 5800X 8-Core Processor             . Intel(R) Corporation. Total memory: 32670 Mb
  Device #1: GPU. NVIDIA GeForce RTX 4080. Total memory: 16375 Mb
Using device #1: GPU. NVIDIA GeForce RTX 4080. Total memory: 16375 Mb
GPU (sum_baseline): 0.00166667 +-0.000471405 s
GPU (sum_baseline): 60000 millions/s
GPU (sum_cycle): 0.001 +-0 s
GPU (sum_cycle): 100000 millions/s
GPU (sum_cycle_coalesced): 0.000833333 +-0.000372678 s
GPU (sum_cycle_coalesced): 120000 millions/s
GPU (sum_local_memory): 0.000666667 +-0.000471405 s
GPU (sum_local_memory): 150000 millions/s
GPU (sum_tree): 0.001 +-0 s
GPU (sum_tree): 100000 millions/s

```

# Размышления 

Хочу отметить - у меня получалось два различных варианта.

В первом варианте в порядке увеличения производительности список методов выглядел так:

1. sum_baseline
2. sum_cycle и sum_tree (делят вдвоём)
3. sum_local_memory
4. sum_cycle_coalesced

Во втором же вот так:

1. sum_baseline
2. sum_cycle и sum_tree (делят вдвоём)
3. sum_cycle_coalesced
4. sum_local_memory

Видно, что в обоих случаях ожидаемо самый плохой метод - через atomic над глобальной памятью. Вторые места совпадают - их делят суммирование через цикл и дерево. Это соответствует ожиданиям, потому что в случае суммирования через цикл нас может замедлять плохой паттерн обращения к памяти, а в случае сложения через дерево - дополнительные барьеры над локальной памятью, которые хоть и быстрее, чем глобальные, но всё равно влияют на производительность.

Самое интересное, как мне кажется, это "борьба" методов с локальной памятью и цикла с coalesced доступом к памяти. Глобально эти методы похожи в том смысле, что один поток суммирует сразу большой кусок массива и записывает его через один атомик. sum_cycle_coalesced может выигрывать по той причине, что в нем отсутствуют дополнительные синхронизации на чтение в локальную память, да и к тому же нет простаивания части потоков. Но при этом все обращения идут к глобальной памяти - и как раз этот факт может быть причиной того, что иногда выигрывает sum_local_memory. У меня на видеокарте очень большой L1 кэш - 128KB per SM. Разом туда влезает треть всего массива. Это значит, что теоретически может быть такое, что мы сможем просуммировать всё за три больших coalesced копирований из глобальной памяти (для этого, однако, понадобиться увеличить число копирований в локальную память каждым потоком, но, возможно, это уже делается драйвером). Работа с локальной памятью в данном случае перекроет падение производительности от того, что большая часть потоков в рабочей группе не будет ничего делать, кроме копирования. 

Я протестировал это предположение, уменьшив массив в 3 раза, чтобы он точно помещался в кэш. В таком случае, опять же не всегда, но с такой же частотой выигрывал sum_local_memory, что частично подтверждает эту теорию. 

```

CPU:     0.0823333+-0.00406885 s
CPU:     400.81 millions/s
CPU OMP: 0.0141667+-0.00313138 s
CPU OMP: 2329.41 millions/s
OpenCL devices:
  Device #0: CPU. AMD Ryzen 7 5800X 8-Core Processor             . Intel(R) Corporation. Total memory: 32670 Mb
  Device #1: GPU. NVIDIA GeForce RTX 4080. Total memory: 16375 Mb
Using device #1: GPU. NVIDIA GeForce RTX 4080. Total memory: 16375 Mb
GPU (sum_baseline): 0.0005 +-0.0005 s
GPU (sum_baseline): 66000 millions/s
GPU (sum_cycle): 0.000333333 +-0.000471405 s
GPU (sum_cycle): 99000 millions/s
GPU (sum_cycle_coalesced): 0.000166667 +-0.000372678 s
GPU (sum_cycle_coalesced): 198000 millions/s
GPU (sum_local_memory): 0 +-0 s
GPU (sum_local_memory): inf millions/s
GPU (sum_tree): 0.000333333 +-0.000471405 s
GPU (sum_tree): 99000 millions/s

```

Но остается вопрос - почему случается так, что иногда быстрее sum_cycle_coalesced, а иногда sum_local_memory? Мне кажется, что это можно объяснить тем, что на одном SM может работать несколько кернелов, а L1 кэш у них один. То есть, возможно, SM уже частично занят другими вычислениями, которые используют большую часть кэша.