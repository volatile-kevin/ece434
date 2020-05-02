import numpy as np
import math

from scipy import signal
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt

f = 100000
c = 299792458
myLamdba = c/f


# recieved
N = 10
theta = 90
t = 5
phi = 2* np.pi * theta / myLamdba

x_t = np.exp(1j*2*np.pi*f*t)

steering = np.transpose([np.exp(1j * phi * k) for k in range(N)])

steering_c = np.transpose([(2* np.pi * k / myLamdba) for k in range(1,361)])

y_t = steering * x_t
constructed = steering_c * x_t

spectrum = np.correlate(constructed, y_t)

# plt.plot(range(360), spectrum)
# plt.show()