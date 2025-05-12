import scipy.integrate as int
import numpy as np
import matplotlib.pyplot as plt

N = 10
x = np.linspace(0, 1, N)
dx = x[1] - x[0]

#matrix
main_diag = -2 * np.ones(N)
off_diag = np.ones(N - 1)
A_matrix = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

def A(t):
    return 0

def B(t):
    return 0

def lines_1D(t, u):
    #u[0] = A(t)
    u[0] = u[1] - A(t) * dx
    #(u[1] - u[0])/dx = A(t)
    u[-1] = B(t)


    b = np.zeros(N)
    b[0] = A(t)
    b[-1] = B(t)

    #eqn 2 in method of lines
    first_term = (A_matrix @ u + b) / dx**2
    source = np.exp(u)
    return first_term + source

# initial condition
u0 = np.zeros(N)
u0[0] = A(0)
u0[-1] = B(0)

# time span and evaluation points
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 100)

sol = int.solve_ivp(lines_1D, t_span, u0, t_eval=t_eval, method='RK45')

# plot the solution at final time
plt.plot(x, sol.y[:, -1])
plt.show()

plt.pcolormesh(x, sol.t, sol.y.T, shading='auto', cmap='hot')
#plt.plot(x, sol.y[:, -1])
plt.xlabel('x', fontsize = 10)
plt.ylabel('Time (s)', fontsize = 10)
plt.grid(True)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Heatmap of [ ]', fontsize = 20)
cbar = plt.colorbar()
cbar.set_label(label='Temperature $T$', fontsize=15)
cbar.ax.tick_params(labelsize=15)


plt.savefig('Combustion/plots/plot.pdf', bbox_inches = 'tight')

plt.show()