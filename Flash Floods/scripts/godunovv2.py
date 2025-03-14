import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81
f = 0.1
sigma = 0.5
delta = 5
V = 10
A_L = 1
shape = 'Rectangle'

# Functions
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
        alpha = np.arctan(0.02)
    if shape == 'Wedge':
        alpha = np.arctan(0.08)
    return np.sqrt((g * np.sin(alpha) * A) / (f * l(A)))

def int_cond(s):
    b = 0
    norm = np.sqrt(2 * np.pi * (sigma ** 2))
    return (V / norm) * np.exp(-((s - b) ** 2) / (sigma ** 2)) + A_L

def Q(A):
    return A * u_bar(A)

def wave_speed(A):
    if shape == 'Rectangle':
        alpha = np.arctan(0.02)
        w = 10
        return (3/2) * np.sqrt(g * w * np.sin(alpha)/f) * ((3/2)* np.sqrt(A/(2*A + w**2)) - ((2*A + w**2) ** (-3/2)))
    if shape == 'Wedge':
        alpha = np.arctan(0.08)
        theta = np.pi/6
        return (5/4) * np.sqrt((g * np.sin(alpha)/f) * np.sqrt(np.sin(theta)/8)) * (A ** (1/4))

# Discretization
N = 100
L = 50
dx = L / N
x = np.linspace(1, L, N)
A = int_cond(x)

# CFL condition
max_speed = np.max(u_bar(A))
CFL = 0.5
dt = CFL * dx / max_speed
t_end = 20
t = 0

# Prepare for plotting characteristics
T = np.linspace(0, t_end, 500)
lines = []

# Godunov method
plt.figure(figsize=(14, 8))
while t < t_end:
    Q_flux = np.zeros(N + 1)
    
    for i in range(1, N):
        if u_bar(A[i - 1]) > 0:
            Q_flux[i] = Q(A[i - 1])
        else:
            Q_flux[i] = Q(A[i])
    
    A_new = np.zeros(N)
    for i in range(1, N - 1):
        A_new[i] = A[i] - (dt / dx) * (Q_flux[i + 1] - Q_flux[i])
    
    A_new[0] = A_new[1]
    A_new[-1] = A_new[-2]
    
    t += dt
    A = A_new
    
    # Plot solution every second
    if t % 1.0 < dt:
        plt.plot(x, A, label=f't={t:.1f}s', alpha=0.7)

        # Plot characteristics
        line = []
        for s0 in x:
            line.append(s0 + wave_speed(int_cond(s0)) * t)
        lines.append(line)

# Find shock intersections
intersections = []
for i in range(len(lines) - 1):
    for j in range(len(lines[i]) - 1):
        if np.isclose(lines[i][j], lines[i + 1][j], rtol=1e-5):
            intersections.append((lines[i][j], T[j]))

# Plot characteristics
for line in lines:
    plt.plot(line, T, 'k-', lw=0.8, alpha=0.5)

# Mark shock intersections
for (x_shock, t_shock) in intersections:
    plt.plot(x_shock, t_shock, 'ro', markersize=5)

plt.xlabel('Distance, $x$ (m)')
plt.ylabel('Time, $t$ (s)')
plt.title('Godunov Method with Characteristics and Shocks')
plt.legend()
plt.show()
