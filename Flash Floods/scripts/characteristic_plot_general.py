import numpy as np
import matplotlib.pyplot as plt

shape = 'Rectangle'

g = 9.81
f = 0.1 
A_max = 5
L = 5 #std dev

if shape == 'Rectangle':
    alpha = np.arctan(0.02) * np.pi /180
    w = 10.0  # width of the channel (m)

    def wave_speed(A):
        #l =  w + (2 * A/w)
        return (3/2) * np.sqrt(g * w * np.sin(alpha)/f) * ((3/2)* np.sqrt(A/(2*A + w**2)) - ((2*A + w**2) ** (-3/2)))
    
if shape == 'Wedge':
    alpha = np.arctan(0.08) * np.pi /180
    theta = np.pi/6
    
    def wave_speed(A): 
        #l =  np.sqrt(8 * A / np.sin(theta))
        return (5/4) * np.sqrt((g * np.sin(alpha)/f) * np.sqrt(np.sin(theta)/8)) * (A ** (1/4))

def A0(s):
    b = 0
    sigma = L/5
    return (1/A_max) * np.exp(-((s - b)**2) / (sigma**2))
    #return np.where(np.abs(s) <= L, A_max * (1 - np.abs(s)/L), 0)

def characteristic(s0, t):
    return s0 + wave_speed(A0(s0)) * t #x(t;P) = v(c)t + P

t_shock = L / wave_speed(A_max)
t_max = 2 * t_shock
t = np.linspace(0, t_max, 500)
s = np.linspace(5 * L, 5 * L, 1000)

plt.figure(figsize=(10, 6))
for s0 in np.linspace(-L, L, 100):
    plt.plot(characteristic(s0, t), t, 'b-', lw=1, alpha=0.5)

plt.xlabel('Distance along river, $s$ (m)')
plt.ylabel('Time, $t$ (Hours)')
plt.title(f'Characteristic Diagram for {shape} Channel')
plt.legend()
plt.grid(alpha=0.3)
plt.show()