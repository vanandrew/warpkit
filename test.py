from scipy.io import loadmat
from scipy.sparse import lil_array
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian


# simulate mri imaging system
class MRSystem:
    def __init__(self, signal, distance=100, base_resolution=100, off_resonance: bool = False):
        # store signal
        self.signal = signal

        # get signal shape
        self.shape = signal.shape
        assert len(self.shape) == 2
        assert self.shape[0] == self.shape[1]

        # store distance and base resolution
        self.distance = distance
        self.base_resolution = base_resolution

        # get voxel and physical size
        self.voxel_size = distance / base_resolution
        self.physical_size = distance / signal.shape[0]

        # setup voxel basis
        # get the size of the pixel relative to the original signal size
        # this is the number of elements inside a voxel
        self.basis_size = int(self.voxel_size / self.physical_size)
        # check if voxel is integer multiple of physical size
        assert np.isclose(self.basis_size, self.voxel_size / self.physical_size)
        # check that basis size is integer multiple of signal
        assert self.shape[0] % self.basis_size == 0

        # get number of basis
        num_basis = int(self.shape[0] / self.basis_size)

        # generate the basis
        row_index = np.zeros(((self.basis_size ** 2) * (num_basis ** 2)))
        col_index = np.zeros(((self.basis_size ** 2) * (num_basis ** 2)))
        data = np.ones(((self.basis_size ** 2) * (num_basis ** 2))) * (1 / (self.basis_size ** 2))
        basis_set = lil_array((self.shape[0] * self.shape[1], num_basis * num_basis))
        n = 0
        resampled = np.zeros((num_basis, num_basis))
        for j in range(num_basis):
            for i in range(num_basis):
                # start stop block
                start_i = i * self.basis_size
                end_i = (i + 1) * self.basis_size
                start_j = j * self.basis_size
                end_j = (j + 1) * self.basis_size

                resampled[i, j] = signal[start_i:end_i, start_j:end_j].mean()

                # # loop through each line of block
                # for nj in range(start_j, end_j):
                #     lin_start = start_i + nj * num_basis * self.basis_size
                #     lin_end = end_i + nj * num_basis * self.basis_size
                #     basis_set[lin_start:lin_end, n] = 1
                # n += 1
        self.signal = resampled
        # self.basis = csc_array((data, (row_index, col_index)), shape=(self.shape[0], self.shape[1]))

        # generate position grid
        self.x = np.linspace(0, distance - self.voxel_size, self.signal.shape[0])
        self.y = np.linspace(0, distance - self.voxel_size, self.signal.shape[1])

        # make grid
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing="xy")

        # setup off resonance artifact
        self.off_resonance = np.zeros(self.signal.shape)
        if off_resonance:
            mid_point = self.signal.shape[0] // 2
            self.off_resonance[-40 + mid_point - 10 : -40 + mid_point + 10, mid_point - 10 : mid_point + 10] = 2.5e-3
            self.off_resonance = gaussian(self.off_resonance, sigma=(10, 20))

        # set larmor frequency
        self.larmor_frequency = 42.577 / (2 * np.pi)  # MHz / T

        # setup gradient timings
        self.Gx = np.zeros((self.base_resolution**2))
        self.Gy = np.zeros((self.base_resolution**2))
        for i in range(self.base_resolution):
            if i % 2 == 0:
                self.Gx[i * self.base_resolution : (i + 1) * self.base_resolution] = 1
            else:
                self.Gx[i * self.base_resolution : (i + 1) * self.base_resolution] = -1
            self.Gx[i * self.base_resolution] = 0
            if i != self.base_resolution - 1:
                self.Gy[(i + 1) * self.base_resolution] = 1

        # set gradient strengths
        self.Gx /= self.larmor_frequency
        self.Gy /= self.larmor_frequency

    def kalpha(self, grad_integral):
        return self.larmor_frequency * grad_integral

    def kx(self, t):
        grad_integral = np.sum(self.Gx[0 : t + 1])
        return self.kalpha(grad_integral) / self.base_resolution

    def ky(self, t):
        grad_integral = np.sum(self.Gy[0 : t + 1])
        return self.kalpha(grad_integral) / self.base_resolution

    def get_sample(self, t):
        kx = self.kx(t)
        ky = self.ky(t)
        print("                                     ", end="\r")
        print(f"kx: {kx:.3f}, ky: {ky:.3f}", end="\r")
        return np.sum(
            self.signal
            * np.exp(-1j * 2 * np.pi * (self.off_resonance * t + kx * self.xx + ky * self.yy))
        )


def cut_kspace(k_space, cut):
    q1 = k_space[0:cut, 0:cut]
    q2 = k_space[0:cut, -cut:]
    q3 = k_space[-cut:, 0:cut]
    q4 = k_space[-cut:, -cut:]
    h1 = np.concatenate((q1, q2), axis=1)
    h2 = np.concatenate((q3, q4), axis=1)
    return np.concatenate((h1, h2), axis=0)


if __name__ == "__main__":
    # Load the data
    data = loadmat("phantom.mat")
    phantom = data["phantom"]

    sys = MRSystem(phantom, distance=100, base_resolution=100, off_resonance=False)
    even = np.zeros(sys.signal.shape).astype(np.complex64)
    odd = np.zeros(sys.signal.shape).astype(np.complex64)
    for t in range(sys.base_resolution**2):
        m = np.round(t // sys.base_resolution)
        n = np.round(t % sys.base_resolution)
        if m % 2 == 0:
            even[m, n] = sys.get_sample(t)
        else:
            odd[m, sys.base_resolution - n - 1] = sys.get_sample(t)
    print()
    kspace = even + odd
    recon = np.fft.ifft2(kspace)
    orig_kspace = np.fft.fft2(phantom)

    f = plt.figure()
    f.add_subplot(2, 2, 1)
    plt.imshow(np.log(np.abs(kspace)))
    f.add_subplot(2, 2, 2)
    plt.imshow(np.log(np.abs(orig_kspace)))
    f.add_subplot(2, 2, 3)
    plt.plot(sys.Gx)
    plt.plot(sys.Gy)
    f.add_subplot(2, 2, 4)
    plt.plot(np.add.accumulate(sys.Gx))
    plt.plot(np.add.accumulate(sys.Gy))

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
