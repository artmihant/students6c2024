# Практикум по вычмеху для 6 курса мехмата МГУ
## Темы занятий
1. Методы Эйлера и Рунге-Кутты на примере расчета движения математического маятника
2. Расчет двойного маятника. Физическая неусточивость
3. Метод стрельбы для решения нелинейных краевых задач. 
4. Метод Ньютона для решения нелинейных задач.
5. Методы ускорения кода
6. Lattice Boltzmann Method
7. Уравнения акустики в 1D случае, явная схема

## Домашние задания
1. Написать программу, рисующую n = 5 маятников с разной частотой.
2. Написать программу, рисующую пространство эволюции лагранджиана в зависимости от двух параметров маятников. В качестве параметров выбрать: 
    а) массы
    б) длины

3. Написать программу, расчитывающую формы конечной (не обязательно малой) деформации линейно-упруго консольно закрепленной балки методом Ньютона. 
    Расчет той же формы методом стрельбы предоставлен в папке урока и может быть использован для верификации результата.
    Данное задание обладает повышенной сложностью и оценивается в два балла

4. Написать программу, рисующую фрактал Мандельброта. Проверить, как отличается рисунок фрактала, если взять в качестве стартового значения итерационного процесса не 0 а какое-либо другое число.

5. На репозиторий загружена программа: /lesson6/fluid/LBMfluidD2Q9.py . Она моделирует (методом решеток Больцмана) потоки вокруг движущегося сквозь воздушную среду тела. Требуется ускорить данную программу средствами numba. В зависимости от железа, возможно ускорить её выполнение в несколько раз без существенных усилий (не пребегая к GPU).

6. На семинаре был разобран метод решетчатых уравнений Больцмана (Lattice Boltzmann Method). 
    На репозиторий загружена программа: /lesson6/fluid/LBMfluidD2Q9.py реализующая моделирование потоков газа вокруг равномерно движущегося круглого тела методом LBM. Взяв за основу её код или написав свой, необходимо сделать одну из следующих модификаций:
    
    a. Реализовать трехмерную схему D3Q19 (с^2 = 1/3, w0 = 1/3, w[1-6]=1/18, w[7-18]=1/36) или D3Q27 (с^2 = 1/3, все коэфициенты - результат внешнего возведения в куб вектора весов для W_C1Q3 = [1/6, 2/3, 1/6])
    b. Реализовать шестиугольную схему D2Q7 (с^2 = 1/4, w0 = 1/2, w[1-6]=1/12)
    с. Реализовать неравномерное движение тела, заданное в коде программы функционально
    d. Реализовать внутри системы "силу тяжести" по одной из осей
    f. Реализовать ускорение расчета задачи средствами CUDA/GPU

7. На занятии была решена одномерная задача акустики, эта программа лежит в репозитории: /lesson7/acoustics_1D.py . Также на занятии была описана постановка одномерной динамической задачи упругости. Необходимо внести в программу изменения, чтобы она стала решать задачу о распространении упругой волны вместо акустической.

