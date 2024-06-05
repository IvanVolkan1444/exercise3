import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt

class FieldDisplay:
    def __init__(self, size_m, dx, ymin, ymax, probePos, sourcePos):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line = self.ax.plot(np.arange(0, size_m, dx), [0]*int(size_m/dx))[0]
        self.ax.plot(probePos*dx, 0, 'xr')  # Позиция датчика
        self.ax.plot(sourcePos*dx, 0, 'ok')  # Позиция источника
        self.ax.set_xlim(0, size_m)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_xlabel('x, м')
        self.ax.set_ylabel('Ez, В/м')
        self.probePos = probePos
        self.sourcePos = sourcePos

    def update(self, data):
        self.line.set_ydata(data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class Probe:
    def __init__(self, probePos, maxT, dt):
        self.maxT = maxT
        self.dt = dt
        self.probePos = probePos
        self.t = 0
        self.E = np.zeros(self.maxT)
    def add(self, data):
        self.E[self.t] = data[self.probePos]
        self.t += 1

    def showSpectrum(self):
        sp = fftshift(np.abs(fft(self.E)))  # Выполнение Фурье-преобразования и центрирование спектра
        df = 1 / (self.maxT * self.dt)
        freq = np.arange(-self.maxT * df / 2, self.maxT * df / 2, df)
        fig, ax = plt.subplots()
        ax.plot(freq, sp / max(sp))
        ax.set_xlabel('f, Гц')  # Ось частоты в Гц
        ax.set_ylabel('|S|/|S|max')
        ax.set_xlim(0, 2.5e9)  # Ограничение частотного диапазона до 2.5 ГГц
    
    def showSignal(self):
        fig, ax = plt.subplots()
        ax.plot(np.arange(0, self.maxT * self.dt, self.dt), self.E)
        ax.set_xlabel('t, c')
        ax.set_ylabel('Ez, В/м')
        ax.set_xlim(0, self.maxT * self.dt)

# Параметры моделирования
eps = 8.0  # Диэлектрическая проницаемость
W0 = 120 * np.pi  # Коэффициент пространственной дискретизации
Sc = 1  # Коэффициент устойчивости
maxT = 4000  # Количество временных шагов
size_m = 3.5  # Размер области моделирования
dx = size_m / 3000  # Пространственный шаг
maxSize = int(size_m / dx) + 1  # Максимальный размер массива (+1 для учета граничного элемента)
probePos = int(size_m / 4 / dx)  # Позиция датчика
sourcePos = int(size_m / 2 / dx)  # Позиция источника
dt = dx * np.sqrt(eps) * Sc / 3e8  # Временной шаг

probe = Probe(probePos, maxT, dt)
display = FieldDisplay(size_m, dx, -1, 1, probePos, sourcePos)
Ez = np.zeros(maxSize)
Hy = np.zeros(maxSize)  # Увеличиваем размер Hy на 1, чтобы сделать его равным размеру Ez
Sc1 = Sc / np.sqrt(eps)
A_max = 10
Ez_old = 0

# Параметры для дифференцированного гауссова импульса
F_min = 0.1e9  # Минимальная частота сигнала в Герцах
F_max = 2.2e9  # Максимальная частота сигнала в Герцах

# Вычисляем f0 и df
f0 = (F_max + F_min) / 2
df = F_max - F_min

# Вычисляем w_g и d_g на основе f0 и df
w_g = np.sqrt(np.log(5.5 * A_max)) / (np.pi * f0) / dt
d_g = w_g * np.sqrt(np.log(2.5 * A_max * np.sqrt(np.log(2.5 * A_max))))

# Моделирование распространения волны
for q in range(1, maxT):
    Hy[:-1] = Hy[:-1] + (Ez[1:] - Ez[:-1]) * Sc / W0
    Ez[1:] = Ez[1:] + (Hy[1:] - Hy[:-1]) * Sc * W0 / eps
    # Используем w_g и d_g для расчета дифференцированного гауссова импульса
    Ez[sourcePos] += -2 * ((q - d_g) / w_g) * np.exp(-((q - d_g) / w_g) ** 2)
    
    # Граничное условие ABC первой степени на левой границе
    Ez[0] = Ez_old + (Sc - 1) / (Sc + 1) * (Ez[1] - Ez[0])
    Ez_old = Ez[1]

    probe.add(Ez)
    if q % 10 == 0:
        display.update(Ez)

plt.ioff()
probe.showSignal()
probe.showSpectrum()
plt.show()
