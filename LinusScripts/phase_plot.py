import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import solve_ivp

# Constants
g = 9.81  # Gravity (m/s^2)
r_max = 7 # Max length (squatting)
r_min = 6  # Min length (standing)
cycles = 20 # Number of oscillations

#height = 2m
# Pendulum equations
def pendulum_stand(t, state_stand):
    theta, theta_dot = state_stand
    theta_ddot = - (g / r_min) * np.sin(theta)
    return [theta_dot, theta_ddot]

def pendulum_sit(t, state_sit):
    theta, theta_dot = state_sit
    theta_ddot = - (g / r_max) * np.sin(theta)
    return [theta_dot, theta_ddot]

def theta_dot_plus(theta_minus):
    return ((r_min/r_max) ** 2) * theta_minus

def event_stand(t, state_stand):
    return state_stand[1]
event_stand.terminal = True

def event_sit(t, state_sit):
    return state_sit[1]
event_sit.terminal = True

theta_dot_max = (2 * np.sqrt(g * r_max) / r_max) * ((r_max/r_min) ** 2)
theta_dots = []
theta_new = theta_dot_max

for _ in range(cycles):
    theta_dots.append(theta_new)
    
    theta_new = theta_dot_plus(theta_new)
    
#print(theta_dots)
#plt.scatter(np.zeros_like(theta_dots), theta_dots)
#plt.show()
evals = cycles - 1
times = [0]
total_time = []
total_theta = []
max_time_sit = 0
max_time_stand = 0

for i in range(evals):
    y_0 = [0, theta_dots[i + 1]]
    
    if i % 2 == 1:
        sol_sit = solve_ivp(pendulum_sit, t_span = [0, 100], y0 = y_0, method = 'RK45', t_eval = np.linspace(0, 10, 10000), events = event_sit)
        
        #rmax
        plt.plot(-sol_sit.y[1], -sol_sit.y[0], label = 'sit', color = 'blue')
        plt.plot(-sol_sit.y[1], sol_sit.y[0], label = 'sit', color = 'blue')
        
        max_time_sit = max(sol_sit.t)
        times.append(max_time_sit)
        
        total_time.append(sol_sit.t + times[i])
        #total_time.append(sol_sit.t + max_time_sit)
        
        total_theta.append(sol_sit.y[1])
        #total_theta.append(np.flip(-sol_sit.y[1]))
    
    else:
        sol_stand = solve_ivp(pendulum_stand, t_span = [0, 100], y0 = y_0, method = 'RK45', t_eval = np.linspace(0, 10, 10000), events = event_stand)
        #print(sol_stand)
        #rmin
        plt.plot(sol_stand.y[1], sol_stand.y[0], label = 'stand', color = 'blue')
        
        if i != 0:
            plt.plot(sol_stand.y[1], -sol_stand.y[0], label = 'stand', color = 'blue') 
            
        max_time_stand = max(sol_stand.t)
        times.append(max_time_stand) 
        
        #total_time.append(sol_stand.t + max_time_stand)
        #total_time.append(np.flip(-sol_stand.t) + max_time_stand)
        
        #total_theta.append(sol_stand.y[1])
        #total_theta.append(np.flip(-sol_stand.y[1]))
    
#print(times)
#print(np.sum(times))

#plt.legend()
plt.grid()
plt.axhline(0,color='black')
plt.axvline(0,color='black')
plt.xlim(-0.75 * np.pi - 0.2, 0.75 * np.pi + 0.2)
plt.ylim(-2.1, 2.52)
plt.scatter(0, 2 * np.sqrt(g * r_max) / r_max, c = 'red', marker = 'x')
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.5))
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(np.pi/4))
plt.gca().xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda val, pos: 
                         r'$0$' if np.isclose(val, 0, atol=1e-10) else 
                         rf'${val/np.pi:.2g}\pi$')
                                    )
plt.text(0.1, (2 * np.sqrt(g * r_max) / r_max) - 0.2, s = '$\\theta_{max}$', color = 'red')
plt.title('Phase Portrait of Kikker')
plt.xlabel(r'$\theta$ (radians)')
plt.ylabel(r'$\dot{\theta}$ (rad/s)')
plt.savefig('Figures/phase_plot.pdf') 
plt.show()
plt.close('all')

flat_time = np.array(np.concatenate(total_time), dtype=np.float64)
flat_theta = np.array(np.concatenate(total_theta), dtype=np.float64)

plt.plot(flat_time, flat_theta, label='Total Time vs Theta')
plt.xlabel('Time (s)')
plt.ylabel(r'$\theta$ (radians)')
plt.title('Combined Theta vs Time')
plt.grid(True)
plt.legend()
plt.show()