import scipy.integrate as int
import numpy as np
import matplotlib.pyplot as plt

#boundary conditions:
def A(t):
    # Example boundary value at left end, change as needed
    return 0

def B(t):
    # Example boundary value at right end, change as needed
    return 0


N = 10
x = np.linspace(-1, 1, N)

u = np.zeros(N)
s = np.zeros(N)

array = -2 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)
print(array)

boundaryConditions = np.zeros(N)

def lines_1D(t, u):
    boundaryConditions[0] = A(t)
    boundaryConditions[-1] = B(t)
    diffusion = (np.matmul(array, u) + boundaryConditions) / (x[1] - x[0])**2
    source = np.exp(u)
    return diffusion + source

# Initial condition: all zeros (or you can modify this)
u0 = np.zeros(N)

# Time span for the simulation
t_span = (0, 1)  # From t=0 to t=1
t_eval = np.linspace(t_span[0], t_span[1], 100)  # Points where solution is computed

# Solve the system using solve_ivp
sol = int.solve_ivp(lines_1D, t_span, u0, t_eval=t_eval, method='RK45')

# Plot the solution at final time
plt.plot(x, sol.y[:, -1])
plt.xlabel('x')
plt.ylabel('Temperature T')
plt.title('Temperature profile at final time')
plt.grid(True)
plt.show()
