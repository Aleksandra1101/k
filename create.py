import numpy as np
import matplotlib.pyplot as plt

def f(x_point):
    x1 = x_point[0]  # х координата
    x2 = x_point[1]  # у координата
    print('x1 ->', x1)
    print('x2 ->', x2)
    # функция Розенброка:
    fx = 100.0 * (x1 ** 2.0 - x2) ** 2.0 + (x1 - 1) ** 2.0
    print('f(x) =', fx, '\n')
    return fx

x0 = np.array([-1.2, 0], float)  # начальная точка
eps = 0.000000001  # задаем точность вычислений
alpha = 3  # при удачном шаге
beta = -0.5  # при неудачном шаге

d1 = np.array([1, 0], float)  # вектор (1, 0)
d2 = np.array([0, 1], float)  # вектор (0, 1)
delta1 = 0.000001  # дельты принятые на первом шаге
delta2 = 0.000001  # дельты принятые на первом шаге
N = 3  # ??? максимальное число неудачных серий шагов
n = 1  # максимальное число неудачных серий шагов
l = 0  # выполнено серий неудачных шагов

a = np.array([[0, 0]], float)

y = np.array([x0])  # массив точек у
x = np.array([x0])  # массив точек х
lamb = np.array([[0, 0]], float)  # массив лямбд для шага 4

k = 0  # ???
i = 0  # текущая итреация 2-го шага
delta = [delta1, delta2]  # создаем массив из выше написанных дельт
d = np.array([d1, d2])  # создаем массив из выше написанных д

STOP = False  # условие окончания

z = 0
while not STOP:
    z += 1
    print('---------------------------------- z :', z)

    # Шаг 2
    print('\n\n-------- Шаг 2 --------')
    if f(y[-1] + delta[i] * d[i]) >= f(y[-1]):
        print('# шаг неудачен')
        y = np.append(y, [y[-1]], axis=0)
        delta[i] = delta[i] * beta
    else:
        print('# шаг удачен')
        y = np.append(y, [y[-1] + delta[i] * d[i]], axis=0)
        delta[i] = delta[i] * alpha
    # Конец Шаг 2

    # Шаг 3
    print('\n\n-------- Шаг 3 --------')
    if i < n:  # перейти к Шаг 2
        i = i + 1
    else:
        l = l + 1
        if f(y[-1]) < f(y[0]):  # перейти к Шаг 2
            print('# шаг неудачен')
            i = 0
            y = np.array([y[-1]])
        else:
            print('# шаг удачен')
            if f(y[-1]) < f(x[-1]):  # перейти к Шаг 4
                print('')

                # Шаг 4
                print('\n\n-------- Шаг 4 --------')
                x = np.append(x, [y[-1]], axis=0)
                if np.sqrt(np.sum(np.square(x[-1] - x[-2]))) > eps:
                    point = np.array([x[-1][0] - x[-2][0], x[-1][1] - x[-2][1]])
                    left_side = np.array([[d[0][0], d[1][0]], [d[0][1], d[1][1]]], float)
                    right_side = np.array([point[0], point[1]], float)
                    res_for_lambda = np.linalg.solve(left_side, right_side)
                    lamb[0] = np.array(res_for_lambda)
                    print(lamb[0])
                    # print(lamb_search(lamb[0]))
                    a1 = lamb[0][0] * d[0] + lamb[0][1] * d[1]
                    b1 = np.copy(a1)
                    a2 = lamb[0][1] * d[1]
                    d1_new = b1 / (np.sqrt(np.sum(np.square(b1))))
                    b2 = a2.T - (a2 @ d1_new.T) * d1_new.T
                    d2_new = b2 / (np.sqrt(np.sum(np.square(b2))))
                    i = 0
                    k = k + 1
                    d = np.array([d1_new, d2_new.T])
                    print('d1 => ', d[0])
                    print('d2 => ', d[1])
                    delta = [delta1, delta2]
                    y = np.array([x[-1]])
                # Конец Шаг 4

            else:
                if abs(np.sort(delta)[0]) < eps and abs(np.sort(delta)[-1]) < eps:  # возможно ответ
                    print('Результат: ', y[-1])
                    STOP = True
                else:  # перейти к Шаг 2
                    i = 0
                    successful_step = 0
                    y = np.array([y[-1]])
                    # Конец Шаг 3


plt.plot(x.T[0], x.T[1])
plt.plot(x.T[0], x.T[1], 'ro')
plt.xlabel('x1', size=14)
plt.ylabel('x2', size=14)

plt.grid()

plt.show()
print('Точок було побудовано:', len(x))

 
ДОДАТОК Б
Лістинг програми 2
import numpy as np
import matplotlib.pyplot as plt

def f(x_point):
    x1 = x_point[0]  # х координата
    x2 = x_point[1]  # у координата
    print('x1 ->', x1)
    print('x2 ->', x2)
    # функция, пока еще НЕ Розенброка:
    # fx = 100.0 * (x1 ** 2.0 - x2) ** 2.0 + (x1 - 1) ** 2.0
    if (x1 ** 2 + x2 ** 2 - 4) >= 0:
        fx = 100.0 * (x1 ** 2.0 - x2) ** 2.0 + (x1 - 1) ** 2.0
    else:
        fx = 100.0 * (x1 ** 2.0 - x2) ** 2.0 + (x1 - 1) ** 2.0 + 1 / (x1 ** 2 + x2 ** 2 - 1)
    print('f(x) =', fx, '\n')
    return fx

def rosen_check(x_point):
    x1 = x_point[0]  # х координата
    x2 = x_point[1]  # у координата
    res = 100.0 * (x1 ** 2.0 - x2) ** 2.0 + (x1 - 1) ** 2.0
    return res

x0 = np.array([-1.2, 0], float)  # начальная точка

eps = 0.000000001  # задаем точность вычислений
alpha = 3  # при удачном шаге
beta = -0.5  # при неудачном шаге

d1 = np.array([1, 0], float)  # вектор (1, 0)
d2 = np.array([0, 1], float)  # вектор (0, 1)
delta1 = 0.000001  # дельты принятые на первом шаге
delta2 = 0.000001  # дельты принятые на первом шаге
N = 3  # ??? максимальное число неудачных серий шагов
n = 1  # максимальное число неудачных серий шагов
l = 0  # выполнено серий неудачных шагов

a = np.array([[0, 0]], float)

y = np.array([x0])  # массив точек у
x = np.array([x0])  # массив точек х
lamb = np.array([[0, 0]], float)  # массив лямбд для шага 4

k = 0  # ???
i = 0  # текущая итреация 2-го шага
delta = [delta1, delta2]  # создаем массив из выше написанных дельт
d = np.array([d1, d2])  # создаем массив из выше написанных д

STOP = False  # условие окончания

z = 0
while not STOP:
    z += 1
    print('---------------------------------- z :', z)

    # Шаг 2
    print('\n\n-------- Шаг 2 --------')
    if f(y[-1] + delta[i] * d[i]) >= f(y[-1]):
        print('# шаг неудачен')
        y = np.append(y, [y[-1]], axis=0)
        delta[i] = delta[i] * beta
    else:
        print('# шаг удачен')
        y = np.append(y, [y[-1] + delta[i] * d[i]], axis=0)
        delta[i] = delta[i] * alpha
    # Конец Шаг 2

    # Шаг 3
    print('\n\n-------- Шаг 3 --------')
    if i < n:  # перейти к Шаг 2
        i = i + 1
    else:
        l = l + 1
        if f(y[-1]) < f(y[0]):  # перейти к Шаг 2
            print('# шаг неудачен')
            i = 0
            y = np.array([y[-1]])
        else:
            print('# шаг удачен')
            if f(y[-1]) < f(x[-1]):  # перейти к Шаг 4
                print('')

                # Шаг 4
                print('\n\n-------- Шаг 4 --------')
                x = np.append(x, [y[-1]], axis=0)
                if np.sqrt(np.sum(np.square(x[-1] - x[-2]))) > eps:
                    point = np.array([x[-1][0] - x[-2][0], x[-1][1] - x[-2][1]])
                    left_side = np.array([[d[0][0], d[1][0]], [d[0][1], d[1][1]]], float)
                    right_side = np.array([point[0], point[1]], float)
                    res_for_lambda = np.linalg.solve(left_side, right_side)
                    lamb[0] = np.array(res_for_lambda)
                    print(lamb[0])
                    # print(lamb_search(lamb[0]))
                    a1 = lamb[0][0] * d[0] + lamb[0][1] * d[1]
                    b1 = np.copy(a1)
                    a2 = lamb[0][1] * d[1]
                    d1_new = b1 / (np.sqrt(np.sum(np.square(b1))))
                    b2 = a2.T - (a2 @ d1_new.T) * d1_new.T
                    d2_new = b2 / (np.sqrt(np.sum(np.square(b2))))
                    i = 0
                    k = k + 1
                    d = np.array([d1_new, d2_new.T])
                    print('d1 => ', d[0])
                    print('d2 => ', d[1])
                    delta = [delta1, delta2]
                    y = np.array([x[-1]])
                # Конец Шаг 4

            else:
                if abs(np.sort(delta)[0]) < eps and abs(np.sort(delta)[-1]) < eps:  # возможно ответ
                    # print('Результат: ', y[-1])
                    STOP = True
                else:  # перейти к Шаг 2
                    i = 0
                    successful_step = 0
                    y = np.array([y[-1]])
                    # Конец Шаг 3

# circle2 = plt.Circle((0, 0), 0.5, color='b', linewidth=2, fill=False)
circle2 = plt.Circle((0, 0), 1, color='g', linewidth=2, fill=False)
# circle3 = plt.Circle((0, 0), 2, color='r', linewidth=2, fill=False)
ax = plt.gca()
plt.plot(x.T[0], x.T[1])
plt.plot(x.T[0], x.T[1], 'ro')
plt.xlabel('x1', size=14)
plt.ylabel('x2', size=14)

ax.grid()

ax.add_artist(circle2)
# ax.add_artist(circle3)
ax.set_xlim((-2.7, 2.7))
ax.set_ylim((-2, 2))
plt.show()
print('Точок було побудовано:', len(x))

print('Результат: ', rosen_check(y[-1]))
