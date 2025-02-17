import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.ticker as ticker

# Constants
g = 9.81  
r_max = 7  # Squatting
r_min = 6  # Standing
cycles = 20  

# Equations of motion
def pendulum_stand(t, state):
    theta, theta_dot = state
    theta_ddot = - (g / r_min) * np.sin(theta)
    return [theta_dot, theta_ddot]

def pendulum_sit(t, state):
    theta, theta_dot = state
    theta_ddot = - (g / r_max) * np.sin(theta)
    return [theta_dot, theta_ddot]

def theta_dot_plus(theta_minus):
    return ((r_min/r_max) ** 2) * theta_minus

# Initial angular velocity sequence
theta_dot_max = (2 * np.sqrt(g * r_max) / r_max) * ((r_max/r_min) ** 2)
theta_dots = [theta_dot_max]

for _ in range(cycles - 1):
    theta_dots.append(theta_dot_plus(theta_dots[-1]))

# Lists to store time and theta values
total_time = []
total_theta = []

current_time = 0  # Keeps track of accumulated time

# Iterate over cycles
for i in range(cycles - 1):
    y_0 = [0, theta_dots[i + 1]]
    
    if i % 2 == 1:
        sol = solve_ivp(pendulum_sit, t_span=[0, 100], y0=y_0, method='RK45', t_eval=np.linspace(0, 10, 10000))
        
    else:
        sol = solve_ivp(pendulum_stand, t_span=[0, 100], y0=y_0, method='RK45', t_eval=np.linspace(0, 10, 10000))

    # Adjust time so it continues from the last phase
    adjusted_time = sol.t + current_time
    current_time = adjusted_time[-1]  # Update the current time for the next phase

    # Store values
    total_time.extend(adjusted_time)
    total_theta.extend(sol.y[0])

# Plot theta vs. time
total_time = np.flip(total_time)

plt.figsize = (15, 15)
plt.plot(total_time, total_theta, color='blue')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel(r'$\theta$ (radians)', fontsize=12)
plt.title(r'$\theta$ vs Time for Kiiking Swing', fontsize=16)
plt.grid()

# Set y-axis ticks at multiples of π/2
yticks = np.arange(-np.pi, np.pi + np.pi/2, np.pi/2)
plt.yticks(yticks)
plt.gca().yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda val, pos: 
                         r'$0$' if np.isclose(val, 0, atol=1e-10) else 
                         rf'${int(val/np.pi*2)}/2\pi$' if val != np.pi and val != -np.pi else 
                         rf'${int(val/np.pi)}\pi$')
)

# Increase tick font sizes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig('Figures/thetatimeplot.pdf', bbox_inches="tight")
plt.show()
plt.close('all')