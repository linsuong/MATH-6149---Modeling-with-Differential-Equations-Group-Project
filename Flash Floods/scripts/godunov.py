import numpy as np
from matplotlib import pyplot as plt

# Constants
g = 9.81
f = 0.1
sigma = 0.5
delta = 5
V = 10
A_L = 1

shape = 'Rectangle'

def l(A):
    if shape == 'Rectangle':
        w = 10
        return w + (2 * A) / w
    if shape == 'Wedge':
        theta = np.pi / 6
        return np.sqrt((8 * A) / (np.sin(theta)))
    if shape == 'Semi':
        theta = np.pi / 3
        return np.sqrt((2 * A) / (theta - np.sin(theta))) * theta

def u_bar(A):
    if shape == 'Rectangle':
        alpha = np.arctan(0.02)  # Correctly left in radians
    if shape == 'Wedge':
        alpha = np.arctan(0.08)  # Correctly left in radians
    return np.sqrt((g * np.sin(alpha) * A) / (f * l(A)))

def int_cond(s):
    b = 0
    norm = np.sqrt(2 * np.pi * (sigma ** 2))
    return (V / norm) * np.exp(-((s - b) ** 2) / (sigma ** 2)) + A_L

def Q(A):
    return A * u_bar(A)

# Discretization
N = 100
L = 5
dx = L / N
x = np.linspace(-5 * L, 5 * L, N)
A = int_cond(x)

# CFL condition
max_speed = np.max(u_bar(A))
CFL = 0.5
dt = CFL * dx / max_speed
t_end = 10
t = 0

# Godunov method
while t < t_end:
    Q_flux = np.zeros(N + 1)
    
    # Upwind flux calculation using Godunov's method
    for i in range(1, N):
        if u_bar(A[i - 1]) > 0:
            Q_flux[i] = Q(A[i - 1])
        else:
            Q_flux[i] = Q(A[i])
    
    # Update cell averages
    A_new = np.zeros(N)
    for i in range(1, N - 1):
        A_new[i] = A[i] - (dt / dx) * (Q_flux[i + 1] - Q_flux[i])
    
    # Apply boundary conditions (e.g., zero flux at boundaries)
    A_new[0] = A_new[1]
    A_new[-1] = A_new[-2]

    # Update time and solution
    t += dt
    A = A_new
    
    # Plot every 1 second
    if t % 1.0 < dt:
        plt.plot(x, A, label=f't={t:.1f}s')
        #print(A)

plt.xlabel('x')
plt.ylabel('A')
plt.legend()
plt.show()
