В этом репозитории предложены задания для курса по вычислениям на видеокартах 2024

[Остальные задания](https://github.com/GPGPUCourse/GPGPUTasks2024/).

# Задание 10. Просто космос






https://github.com/GPGPUCourse/GPGPUTasks2023/assets/22657075/f14a5a41-b2cb-4d1c-896f-50e1e61567f1






[![Build Status](https://github.com/GPGPUCourse/GPGPUTasks2024/actions/workflows/cmake.yml/badge.svg?branch=task10&event=push)](https://github.com/GPGPUCourse/GPGPUTasks2024/actions/workflows/cmake.yml)

0. Сделать fork проекта
1. Выполнить задания 10.0, 10.1, 10.2, 10.3
2. Отправить **Pull-request** с названием ```Task10 <Имя> <Фамилия> <Аффиляция>``` (указав вывод каждой программы при исполнении на вашем компьютере - в тройных кавычках для сохранения форматирования)

**Дедлайн**: 23:59 15 декабря.


Задание 10.0. CPU N-body
=========

Запустите и проверьте, что работает тест `(LBVH, Nbody)`, если закомментировать все варианты реализации кроме первого. 
Так же можно поэкспериментировать и освоиться с настройками, перечисленными в начале файла, позапускав тест `(LBVH, Nbody_meditation)` с наивной CPU реализацией и включенным GUI.


Задание 10.1. GPU N-body 
=========

Реализуйте кернел `nbody_calculate_force_global` и запустите тест `(LBVH, Nbody)` без последних двух вариантов.


Задание 10.2. CPU LBVH
=========

Реализуйте TODO в файле ```src/main_lbvh.cpp```, чтобы начал проходить тест `(LBVH, CPU)` и тест `(LBVH, Nbody)` без последнего варианта.


Задание 10.3. GPU LBVH
=========

Реализуйте оставшиеся TODO в файлах ```src/main_lbvh.cpp``` и ```src/cl/lbvh.cl```, чтобы начал проходить тест `(LBVH, GPU)` и тест `(LBVH, Nbody)` полностью.




https://github.com/GPGPUCourse/GPGPUTasks2023/assets/22657075/d693d49b-009b-4bae-baf4-07b1fdebdcfb

(оверлей структуры LBVH поверх симуляции)

