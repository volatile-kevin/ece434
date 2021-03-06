import csv
import numpy as np
import math
import numpy.linalg as linalg

from scipy import signal
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt

import pandas as pd

# import csv here
data = []

# these will hold magnitudes of acceleration
m_data = []

# replace the string with ("data.csv") before submission
# with open("../finalProject/data_imu_loc/route1/Accelerometer.csv") as csvfile:
#     reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
#     for row in reader:  # each row is a list
#         data.append(row)
acc_data = pd.read_csv("../finalProject/data_imu_loc/route1/Accelerometer.csv")
acc_data = acc_data.to_numpy()
# for x in range(len(data)):
#     m_data.append(math.sqrt((data[x][0]) ** 2 + (data[x][1]) ** 2 + (data[x][2]) ** 2))
m_data = [linalg.norm(i[1:]) for i in acc_data]

# Filter requirements.
order = 4
cutoff = 4.5  # desired cutoff frequency of the filter, Hz
b, a = butter(order, cutoff / 50, btype='low', analog=False)
f_data = lfilter(b, a, m_data)


steps = len(signal.find_peaks(f_data, prominence=2.8)[0])
print(steps)

f = open("result.txt", "w+")
f.write(str(int(steps)))
f.close()
plt.subplot(2, 1, 2)
# plt.plot(m_data, 'b-', label='data')
plt.plot(f_data, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()
plt.show()
