import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

def meme(acc_path, gyro_path, filter_height, step_lonk, color, output_path):
    path = []
    path.append([0,0,0])
    R_array = []
    acc_data = pd.read_csv(acc_path)
    acc_data = acc_data.to_numpy()

    gyro_data = pd.read_csv(gyro_path)
    gyro_data = gyro_data.to_numpy()
    macc_data = [linalg.norm(i[1:]) for i in acc_data]

    step_size = step_lonk

    steps = signal.find_peaks(macc_data, height=filter_height, distance= 40, prominence=10)[0]

    # initial orientation
    initialData = [acc_data[0][1]], [acc_data[0][2]], [acc_data[0][3]]
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
    R_array.append(R)

    pt = acc_data[0][0]
    for ct,wx,wy,wz in gyro_data:
        if ct != np.nan:
            # current time - previous time
            dt = (ct - pt)/1000
            # find l and delta_theta
            l = np.array([wx,wy,wz])
            dtheta = np.sqrt(np.square(wx)+ np.square(wy)+np.square(wz))*dt
            # find projection of l on global axis and normalize
            l_proj = np.matmul(R, l) / np.linalg.norm(np.matmul(R,l))
            ux = l_proj[0]
            uy = l_proj[1]
            uz = l_proj[2]
            ux2 = np.square(l_proj[0])
            uy2 = np.square(l_proj[1])
            uz2 = np.square(l_proj[2])
            # use angle and normalized axis to find new rotation matrix
            newR = np.array([[np.cos(dtheta) + ux2*(1-np.cos(dtheta)), ux*uy*(1-np.cos(dtheta)) - uz*np.sin(dtheta), ux*uz*(1-np.cos(dtheta)) + uy*np.sin(dtheta)],
                             [uy*ux*(1-np.cos(dtheta)) + uz*np.sin(dtheta), np.cos(dtheta)+uy2*(1-np.cos(dtheta)), uy * uz *(1-np.cos(dtheta)) - ux*np.sin(dtheta)],
                             [uz*ux*(1-np.cos(dtheta)) - uy * np.sin(dtheta), uz * uy * (1-np.cos(dtheta)) + ux * np.sin(dtheta), np.cos(dtheta) + uz2 *(1-np.cos(dtheta))]])
            # Next R is this newR times the previous R
            R = newR @ R
            pt = ct
            R_array.append(newR)

    # keep track of the previous step so we know where we are integrating from
    prev_step = steps[0]
    for step in steps[1:]:
        # This is where we start integrating
        dR = R_array[prev_step]
        # Integrate until the midpoint (first step)
        for i in R_array[prev_step+1:int((step+prev_step+1)/2)]:
            dR = i @ dR

        # Calculate the axis of rotation from u = [h-f,c-g,d-b]
        axis = np.array([dR[2][1] - dR[1][2], dR[0][2]-dR[2][0], dR[1][0] - dR[0][1]])
        # axis = np.array([dR[2][1] - dR[1][2], dR[0][2]-dR[2][0]])
        # Normalize and add vector to previous position
        walking = axis / linalg.norm(axis)
        path.append(path[-1] + walking*step_size)

        # This is where we start integrating
        dR = R_array[int((step+prev_step+1)/2)]
        # Integrate from midpoint to next peak (second peak)
        for i in R_array[int((step+prev_step+1)/2)+1:step]:
            dR = i @ dR

        # Calculate the axis of rotation from u = [h-f,c-g,d-b]
        axis = np.array([dR[2][1] - dR[1][2], dR[0][2]-dR[2][0], dR[1][0] - dR[0][1]])
        # axis = np.array([dR[2][1] - dR[1][2], dR[0][2]-dR[2][0]])
        # Flip vector around and add to previous position
        walking = -1 * axis / linalg.norm(axis)
        path.append(path[-1] + walking*step_size)
        prev_step = step

    path = np.array(path)
    # View accelerometer data / step detection
    # plt.plot(macc_data)
    # step_y = [macc_data[i] for i in steps]
    # plt.scatter(steps, step_y, c = 'red')
    # View path
    plt.plot(path[:, 0], path[:, 1], color, label=output_path[0])
    f = open(output_path, "w+")
    for i in path:
        f.write(str(i[0]))
        f.write(" ")
        f.write(str(i[1]))
        f.write("\n")
    f.close()

meme("1/Accelerometer.csv", "1/Gyroscope.csv", 21, 1.1, 'b-', '1.txt')
meme("2/Accelerometer.csv", "2/Gyroscope.csv", 24, 1.1, 'r-', '2.txt')
meme("3/Accelerometer.csv", "3/Gyroscope.csv", 29, 1.1, 'g-', '3.txt')
meme("4/Accelerometer.csv", "4/Gyroscope.csv", 25, 1.1, 'y-', '4.txt')
meme("5/Accelerometer.csv", "5/Gyroscope.csv", 25, 1.1, 'p-', '5.txt')
meme("6/Accelerometer.csv", "6/Gyroscope.csv", 23, 1.1, 'o-', '6.txt')

plt.grid()
plt.legend()
plt.show()