import numpy as np
import csv

with open('Flash Floods/data/Rectangle_intersections.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

data_array = np.array(data, dtype = 'float')
#print(np.shape(data_array))
#print(len(data_array[:]))

for i in range(len(data_array[:]) - 1):
    grad = np.gradient(data_array[i + 1])
    print(f'gradient for line {i + 1} is = {grad[0]}')