import numpy as np


q = np.zeros((5, 5, 2, 2, 4))
num_boxes = 2

q[0, 0, 0, 0, 0] = 10
q[0, 0, 1, 0, 0] = 20

print(q[0, 0][1][0])


