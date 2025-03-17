import numpy as np
import scipy.optimize as opt

# Constants
g = 9.81
f = 0.05 
shape = 'Rectangular'

# Function for wave speed
def wave_speed(A):
    if shape == 'Rectangular':
        alpha = np.arctan(0.02)  
        w = 10.0  # Width of the channel (m)
        
        return (3/2) * np.sqrt(g * w * np.sin(alpha) / f) * ((3/2) * np.sqrt(A / (2*A + w**2)) - A**(3/2) * ((2*A + w**2) ** (-3/2)))

# Function to find A given a target wave speed
def find_A_for_wave_speed(target_v, A_guess=10):
    def equation(A):
        return wave_speed(A) - target_v  # Solve for A where wave_speed(A) = target_v

    # Define bounds for A (must be positive)
    A_min, A_max = 0.1, 5000  # Avoid A=0 for numerical stability

    # Find all possible solutions using Brent's method
    roots = []
    try:
        root1 = opt.brentq(equation, A_min, A_max)
        roots.append(root1)
    except ValueError:
        pass  # No solution in this range

    # If multiple solutions exist, check other ranges
    A_range = np.linspace(A_min, A_max, 100)
    for i in range(len(A_range) - 1):
        try:
            root = opt.brentq(equation, A_range[i], A_range[i + 1])
            if not np.isclose(root, roots).any():  # Avoid duplicates
                roots.append(root)
        except ValueError:
            continue  # No root in this small interval

    return roots

# Example usage
target_v = float(input("Enter wave speed (m/s): "))
A_solutions = find_A_for_wave_speed(target_v)

if A_solutions:
    print(f"Possible values of A for wave speed {target_v} m/s: {A_solutions}")
else:
    print("No valid A found for the given wave speed.")