import numpy as np
import matplotlib.pyplot as plt

shape = 'Rectangle'

g = 9.81
f = 0.1 
A_max = 5.0 
L = 10.0 

if shape == 'Rectangle':
    alpha = np.radians(1/np.pi)
    w = 10.0  # width of the channel (m)

    def wave_speed(A):
        l =  w + (2 * A/w)
        #l_prime = 
        return (3/2) * np.sqrt(g * np.sin(alpha) / l) * np.sqrt(A)
    
if shape == 'Wedge':
    alpha = np.radians(5/np.pi)
    theta = np.pi/6
    
    def wave_speed(A): 
        l =  np.sqrt(8 * A / np.sin(theta))
        return (3/2) * np.sqrt(g/f) * np.sqrt(np.sin(alpha) / l) * np.sqrt(A)

def A0(s):
    b = 0
    sigma = L/5
    return A_max * np.exp(-((s - b)**2) / (2 * sigma**2))
    #return np.where(np.abs(s) <= L, A_max * (1 - np.abs(s)/L), 0)

def characteristic(s0, t):
    return s0 + wave_speed(A0(s0)) * t #x(t;P) = v(c)t + P

t_shock = L / wave_speed(A_max)
t_max = 2 * t_shock
t = np.linspace(0, t_max, 500)
s = np.linspace(- 5*L, 5*L, 1000)

plt.figure(figsize=(10, 6))
for s0 in np.linspace(-L, 5*L, 100):
    plt.plot(characteristic(s0, t), t, 'b-', lw=1, alpha=0.5)

plt.xlabel('Distance along river, $s$ (m)')
plt.ylabel('Time, $t$ (s)')
plt.title(f'Characteristic Diagram for {shape} Channel')
plt.legend()
plt.grid(alpha=0.3)
plt.show()