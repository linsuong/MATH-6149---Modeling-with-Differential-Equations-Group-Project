import scipy.integrate as int
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 10
x = np.linspace(-1, 1, N)
dx = x[1] - x[0]

# Tridiagonal matrix for second derivative
main_diag = -2 * np.ones(N)
off_diag = np.ones(N - 1)
A_matrix = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

# Boundary condition functions
def A(t):
    return 0  # Left boundary value

def B(t):
    return 0  # Right boundary value

# ODE system from Method of Lines
def lines_1D(t, u):
    # Enforce boundary conditions
    u[0] = A(t)
    u[-1] = B(t)

    # Construct boundary vector
    b = np.zeros(N)
    b[0] = A(t)
    b[-1] = B(t)

    # Matrix-vector formulation
    laplacian = (A_matrix @ u + b) / dx**2
    source = np.exp(u)
    return laplacian + source

# Initial condition
u0 = np.zeros(N)
u0[0] = A(0)
u0[-1] = B(0)

# Time span and evaluation points
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 100)

# Solve the system
sol = int.solve_ivp(lines_1D, t_span, u0, t_eval=t_eval, method='RK45')

# Plot the solution at final time
plt.plot(x, sol.y[:, -1])
plt.xlabel('x')
plt.ylabel('Temperature T')
plt.title('Temperature profile at final time')
plt.grid(True)
plt.show()
