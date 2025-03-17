import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as interp

# Define constants
shape = 'Rectangular'
g = 9.81
f = 0.05 
A_max = 5000
L = 100  # Standard deviation
theta = np.pi/4

def wave_speed(A):
    if shape == 'Rectangular':
        alpha = np.arctan(0.02)  
        w = 10.0  # Width of the channel (m)

        return (3/2) * np.sqrt(g * w * np.sin(alpha)/f) * ((3/2)* np.sqrt(A/(2*A + w**2)) - A**(3/2)*((2*A + w**2) ** (-3/2)))
        
    if shape == 'Wedge':
        alpha = np.arctan(0.08) 
        theta = np.pi/6
        return (5/4) * np.sqrt((g * np.sin(alpha)/f) * np.sqrt(np.sin(theta)/8)) * (A ** (1/4))

def A0(s):
    b = 0  
    sigma = L   
    norm = np.sqrt(2 * np.pi * (sigma ** 2))  
    AL = 5  
    return AL + (A_max/norm) * np.exp(-((s - b)**2) / (sigma**2))

# Characteristic equation
def characteristic(s0, t):
    return s0 + wave_speed(A0(s0)) * t  

# Increase time resolution
t_shock = L / wave_speed(A_max)
t_max = 2 * t_shock * 10  
t = np.linspace(0, t_max, 5000)  # High resolution

# Increase s resolution
s0_values = np.linspace(-10 * L, 10 * L, 200)  
characteristics = []

plt.figure(figsize=(10, 6))

for s0 in s0_values:
    char_curve = characteristic(s0, t)
    characteristics.append(char_curve)
    plt.plot(char_curve, t, 'b-', color='black', lw=1, alpha=0.5)

# Convert to array for interpolation
characteristics = np.array(characteristics)

# Find intersections
intersections = []
for i in range(len(characteristics) - 1):
    for j in range(len(t) - 1):
        if np.isclose(characteristics[i][j], characteristics[i + 1][j], rtol=1e-5):
            intersections.append((characteristics[i][j], t[j]))

# Sort intersections by time (so first appears first)
intersections.sort(key=lambda x: x[1])

# Print the first intersection
if intersections:
    first_intersection = intersections[0]
    print(f"First intersection occurs at s = {first_intersection[0]:.2f}, t = {first_intersection[1]:.2f}")
else:
    print("No intersections found.")

# Ask for user input
a = float(input("Enter value for a (near shock region): "))
b = float(input("Enter value for b (near shock region): "))
t_0 = float(input("Enter value for t_0: "))

# Interpolation function to find s(t_0)
def interpolate_s_for_t(s_lines, t_lines, t_target):
    """ Interpolates s values for a given time t_target """
    s_at_t = []
    for i in range(len(s_lines)):
        f_interp = interp.interp1d(t_lines, s_lines[i], kind='linear', bounds_error=False, fill_value=np.nan)
        s_at_t.append(f_interp(t_target))
    return np.array(s_at_t)

# Get all interpolated s values at t_0
s_at_t0 = interpolate_s_for_t(characteristics, t, t_0)

# Find closest points to a and b
idx_a = np.argmin(np.abs(s_at_t0 - a))
idx_b = np.argmin(np.abs(s_at_t0 - b))

# Compute gradients (ds/dt) using finite difference
def finite_difference_gradient(s_vals, t_vals, idx):
    """ Computes gradient (ds/dt) at a given index using finite differences """
    if idx == 0:
        return (s_vals[idx + 1] - s_vals[idx]) / (t_vals[idx + 1] - t_vals[idx])
    elif idx == len(s_vals) - 1:
        return (s_vals[idx] - s_vals[idx - 1]) / (t_vals[idx] - t_vals[idx - 1])
    else:
        return (s_vals[idx + 1] - s_vals[idx - 1]) / (t_vals[idx + 1] - t_vals[idx - 1])

# Compute gradients at a and b
gradient_a = finite_difference_gradient(characteristics[idx_a], t, np.searchsorted(t, t_0))
gradient_b = finite_difference_gradient(characteristics[idx_b], t, np.searchsorted(t, t_0))

# Print results
print(f"Gradient at (a={a}, t_0={t_0}): {gradient_a:.4f}")
print(f"Gradient at (b={b}, t_0={t_0}): {gradient_b:.4f}")

# Mark points on the plot
plt.scatter([s_at_t0[idx_a]], [t_0], color='red', label=f'Point (a={a}, t_0={t_0})')
plt.scatter([s_at_t0[idx_b]], [t_0], color='blue', label=f'Point (b={b}, t_0={t_0})')

# Mark first intersection
if intersections:
    plt.scatter([first_intersection[0]], [first_intersection[1]], color='green', marker='x', s=100, label='First Intersection')

# Plot settings
plt.xlabel('Distance along river, $s$ (m)')
plt.ylabel('Time, $t$ (seconds)')
plt.title(f'Characteristic Diagram for {shape} Channel (High Resolution)')
plt.xlim(-10 * L, 10 * L)  
plt.legend()
plt.grid(alpha=0.3)
plt.show()