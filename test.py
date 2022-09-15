from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from skimage.filters import gaussian

# Load the data
data = loadmat('phantom.mat')
phantom = data["phantom"]


# simulate mri imaging system
class MRSystem:
    def __init__(self, signal, distance=180, base_resolution=90):
        self.signal = signal
        self.base_resolution = base_resolution
        self.distance = distance
        self.voxel_size = distance / base_resolution

        # off resonance
        self.off_resonance = np.zeros(signal.shape)
        self.off_resonance[10:15, 40:50] = -1e-2
        self.off_resonance = gaussian(self.off_resonance, sigma=(6, 12))

        # get signal shape
        self.shape = signal.shape
        assert len(self.shape) == 2
        assert self.shape[0] == self.shape[1]
        self.real_resolution = self.shape[0]
        self.real_size = distance / self.real_resolution

        # compute the real resolution to base resolution ratio
        self.resolution_ratio = self.real_resolution / self.base_resolution

        # generate position grid
        self.y, self.x = np.mgrid[0:self.shape[0], 0:self.shape[1]]
        self.x = self.x.astype(float) / self.shape[0]
        self.y = self.y.astype(float) / self.shape[1]

        # set larmor frequency
        self.larmor_frequency = 42.577 / (2 * np.pi)  # MHz / T
        self.gamma = self.larmor_frequency

        # set gradient strengths
        self.Gx = 1 / self.larmor_frequency
        self.Gy = 1 / self.larmor_frequency

    def kalpha(self, gradient, duration):
        return self.larmor_frequency * gradient * duration

    def kx(self, t):
        m = np.round(t // self.base_resolution)
        n = np.round(t % self.base_resolution)
        if m % 2 == 0:
            t = n
        else:
            t = self.base_resolution - n - 1
        return self.kalpha(self.Gx, t)

    def ky(self, t):
        return self.kalpha(self.Gy, np.round(np.floor(t / self.base_resolution)))

    def get_sample(self, t):
        return np.sum(self.signal * np.exp(-1j * 2 * np.pi * (self.off_resonance * t + self.kx(t) * self.x + self.ky(t) * self.y)))


sys = MRSystem(phantom, base_resolution=90)
even = np.zeros(phantom.shape).astype(np.complex64)
odd = np.zeros(phantom.shape).astype(np.complex64)
for t in range(sys.base_resolution ** 2):
    m = np.round(t // sys.base_resolution)
    n = np.round(t % sys.base_resolution)
    if m % 2 == 0:
        even[n, m] = sys.get_sample(t)
    else:
        odd[sys.base_resolution - n - 1, m] = sys.get_sample(t)        
recon = np.flipud(np.fliplr(np.fft.fft2(even + odd).T))

f = plt.figure()
f.add_subplot(2, 2, 1)
plt.imshow(np.abs(recon), cmap="gray")
f.add_subplot(2, 2, 2)
plt.imshow(np.angle(recon), cmap="gray")
f.add_subplot(2, 2, 3)
plt.imshow(np.abs(phantom), cmap="gray")
f.add_subplot(2, 2, 4)
plt.imshow(sys.off_resonance)
plt.show()

