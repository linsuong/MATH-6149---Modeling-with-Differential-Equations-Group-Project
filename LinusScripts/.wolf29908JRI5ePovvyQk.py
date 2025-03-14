import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
g = 9.81  # Gravity (m/s^2)
r_max = 2  # Max length (squatting)
r_min = 1  # Min length (standing)
cycles = 50  # Number of oscillations
t_step = 100  # Simulation time per phase

# Initial maximum angular velocity (bottom to top)
theta_dot_plus = 2 * np.sqrt(g * r_max) / r_max

# Pendulum equations
def pendulum_system(t, state, r):
    theta, theta_dot = state
    theta_ddot = - (g / r) * np.sin(theta)
    return [theta_dot, theta_ddot]

# Time settings
t_eval = np.linspace(-t_step, 0, 2000)

# Phase space storage
theta_values = []
theta_dot_values = []

for _ in range(cycles):
    # r_max integration (squatting)
    sol_max = solve_ivp(pendulum_system, (-t_step, 0), [0, theta_dot_plus], t_eval=t_eval, args=(r_max,))
    theta_values.extend((sol_max.y[0] + np.pi) % (2 * np.pi) - np.pi)
    theta_dot_values.extend(sol_max.y[1])
    
    # Velocity jump at switching (r_max to r_min)
    theta_min = (sol_max.y[0][-1] + np.pi) % (2 * np.pi) - np.pi
    theta_dot_minus = (r_max**2 / r_min**2) * sol_max.y[1][-1]
    
    # r_min integration (standing)
    sol_min = solve_ivp(pendulum_system, (-t_step, 0), [theta_min, theta_dot_minus], t_eval=t_eval, args=(r_min,))
    theta_values.extend((sol_min.y[0] + np.pi) % (2 * np.pi) - np.pi)
    theta_dot_values.extend(sol_min.y[1])
    
    # Velocity jump at switching (r_min to r_max)
    theta_dot_plus = (r_min**2 / r_max**2) * sol_min.y[1][-1]

# Phase plot
plt.figure(figsize=(10, 6))
plt.scatter(theta_values, theta_dot_values, s=0.5, color="b", alpha=0.7, label="Phase Trajectory")
plt.xlabel("Theta (rad)")
plt.ylabel("Theta dot (rad/s)")
plt.title("Phase Portrait of Kiiking Swing (θ = 0 Bottom, θ = π Top)")
plt.axvline(0, color='gray', linestyle='--', alpha=0.5, label="Bottom of Swing (0)")
plt.axvline(np.pi, color='red', linestyle='--', alpha=0.5, label="Top of Swing (π)")
plt.axvline(-np.pi, color='red', linestyle='--', alpha=0.5)
plt.legend()
plt.grid(True)
plt.show()
