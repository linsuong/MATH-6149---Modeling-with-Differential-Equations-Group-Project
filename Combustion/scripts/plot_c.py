import matplotlib.pyplot as plt
import numpy as np

T_0 = np.linspace(0, 5, 100)


Da_range = np.linspace(0, 1, 10)
for Da in Da_range:
    c = 2 * (np.e ** T_0) * Da
    lhs = np.sqrt(c)
    rhs = np.log(np.abs((np.sqrt(c - 2 * Da) - np.sqrt(c))/-(np.sqrt(c - 2 * Da) - np.sqrt(c))))

    plt.plot(T_0, lhs, c = 'red', label = f'Da = {Da}')
    plt.plot(T_0, rhs, c = 'blue', label = f'Da = {Da}')

plt.label('show')
plt.show()