import csv
import numpy as np
import math
import numpy.linalg as linalg

import matplotlib.pyplot as plt

gyroData = []
data = []

with open("sample_data.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
    for row in reader:  # each row is a list
        data.append(row)

# initial orientation
initialData = [data[0][0]], [data[0][1]], [data[0][2]]
v2 = np.array(initialData)

# declare local Y
localY = np.array([[0,1,0]])

# get new magnitude of "m" vector
v2Mag = np.linalg.norm(v2)
newMag = (-1 * np.matmul(localY, v2))/v2Mag

# m has same direction as v2 
newDirection = v2/v2Mag
newVec = newMag * newDirection
# v1 = y^T - m
v1 = np.add(np.transpose(localY), newVec)

# normalize the vectors
v1 = v1/np.linalg.norm(v1)
v2 = v2/np.linalg.norm(v2)

# solve for rotation matrices
# make determinant first
det = []
det.append(v1[1][0]*v2[2][0] - v1[2][0]*v2[1][0])
det.append(v1[0][0]*v2[2][0] - v1[2][0]*v2[0][0])
det.append(v1[0][0]*v2[1][0] - v1[1][0]*v2[0][0])
# solve
a = np.array([[v1[0][0], v1[1][0], v1[2][0]], [v2[0][0], v2[1][0], v2[2][0]], [det[0], -1 * det[1], det[2]]])
b = np.array([0, 0, 1])
x = np.linalg.solve(a, b)

R = np.array([[x[0], x[1], x[2]], [v1[0][0], v1[1][0], v1[2][0]], [v2[0][0], v2[1][0], v2[2][0]]])

print(R)
# PART 2
for k in range(1, len(data)):
    gyroData.append(data[k])



f = open("result.txt", "w+")
f.write(str(x[0]))
f.write(" ")
f.write(str(v1[0][0]))
f.write(" ")
f.write(str(v2[0][0]))
f.write("\n")
f.close()


