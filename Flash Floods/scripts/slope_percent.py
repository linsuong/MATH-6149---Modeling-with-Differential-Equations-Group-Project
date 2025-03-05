import numpy as np
from matplotlib import pyplot as plt

percent = np.linspace(0, 100, 1000)

angle = np.arctan(percent/100) * np.pi /180

for perc in percent:
    plt.plot(angle, percent)
    
plt.xlabel('angle(radians)')
plt.ylabel('percentage')
plt.show()