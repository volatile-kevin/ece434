import csv
import numpy as np
import math
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


# import csv here
data = []
# these will hold magnitudes of acceleration
m_data = []

# replace the string with ("data.csv") before submission
with open("sample_data/swing_38steps.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
    for row in reader:  # each row is a list
        data.append(row)

for x in range(len(data)):
    m_data.append(math.sqrt((data[x][0]) ** 2 + (data[x][1]) ** 2 + (data[x][2]) ** 2))

# low pass filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Filter requirements.
order = 6
fs = 100.0       # sample rate, Hz
cutoff = 4  # desired cutoff frequency of the filter, Hz
f_data = butter_lowpass_filter(m_data, cutoff, fs, order)


up = True
if f_data[0] < f_data[1]:
    up = True
else:
    up = False

# if the peak/valley is too small ignore it
min_dif = 3
prev_peak = f_data[0]
prev_bottom = f_data[0]
i = 1
steps = 0
while i < len(f_data):
    if up and f_data[i] < f_data[i - 1]:
        if np.abs(i-prev_bottom) > min_dif:
            up = False
            prev_peak = f_data[i]
            steps += 1

    elif not up and f_data[i] > f_data[i - 1] and np.abs(i-prev_peak) > min_dif:
        prev_bottom = f_data[i]
        up = True
    i += 1
print(steps)


plt.subplot(2, 1, 2)
plt.plot(m_data, 'b-', label='data')
plt.plot(f_data, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()
plt.show()
