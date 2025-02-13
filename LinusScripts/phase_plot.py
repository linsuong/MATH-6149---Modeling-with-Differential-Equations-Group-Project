from matplotlib import pyplot as plt
import math

r_max = 2
r_min = 1

theta_dot_plus =  3.14 #max value of theta (working from top to bottom)

theta_dot_values = []

#theta_dot_minus = ((r_max ** 2)) /(r_min ** 2 ) * theta_dot_plus

# Recursive iteration
while True:
    theta_dot_minus = ((r_max**2) / (r_min**2)) * theta_dot_plus
    theta_dot_values.append(theta_dot_minus)
    print(theta_dot_minus)
    
    '''
    theta_dot_plus = ((r_min**2) / (r_max**2)) * theta_dot_minus
    theta_dot_values.append(theta_dot_plus)
    print(theta_dot_plus)
    '''
    
    # Stopping condition (can be modified based on physics constraints)
    if abs(theta_dot_minus - theta_dot_plus) < 1e-10 or math.isinf(theta_dot_minus):  # Convergence check
        break
    
    theta_dot_plus = theta_dot_minus  # Update for next iteration

# Print results
print("Theta dot values:", theta_dot_values)


