from scipy.io import loadmat
from scipy.sparse import dia_array
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian, window


# simulate mri imaging system
class MRSystem:
    def __init__(self, signal, base_resolution=100, off_resonance: bool = False):
        # store signal
        self.signal = signal

        # get signal shape
        self.shape = signal.shape
        assert len(self.shape) == 2
        assert self.shape[0] == self.shape[1]

        # store base resolution
        self.base_resolution = base_resolution

        # generate the basis
        # self.basis_set = np.zeros((self.shape[0] ** 2, self.shape[0] ** 2))
        # self.system_matrix = np.zeros((self.shape[0] ** 2, self.shape[0] ** 2), dtype=np.complex64)
        # for i in range(self.shape[0]):
        #     for j in range(self.shape[1]):
        #         self.basis_set[i * self.shape[0] + j, i * self.shape[0] + j] = 1
        #         basis = np.zeros(self.shape)
        #         basis[i, j] = 1
        #         self.system_matrix[:, i * self.shape[0] + j] = np.fft.fft2(basis).ravel()

        # generate position grid
        self.x = np.arange(self.shape[1])
        self.y = np.arange(self.shape[0])

        # make grid
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing="xy")

        # setup off resonance artifact
        self.off_resonance = np.zeros(self.signal.shape)
        if off_resonance:
            mid_point = self.shape[0] // 2
            self.off_resonance[5:30, mid_point - 30 : mid_point + 30] = 1e-3
            self.off_resonance = gaussian(self.off_resonance, sigma=(10, 20))

        # set larmor frequency
        self.larmor_frequency = 42.577 / (2 * np.pi)  # MHz / T

        # setup gradient timings
        self.Gx = np.zeros((self.shape[1] ** 2))
        self.Gy = np.zeros((self.shape[0] ** 2))
        for i in range(self.shape[1]):
            if i % 2 == 0:
                self.Gx[i * self.shape[1] : (i + 1) * self.shape[1]] = 1
            else:
                self.Gx[i * self.shape[1] : (i + 1) * self.shape[1]] = -1
            self.Gx[i * self.shape[1]] = 0
            if i != self.shape[1] - 1:
                self.Gy[(i + 1) * self.shape[1]] = 1

        # set gradient strengths
        self.Gx /= self.larmor_frequency
        self.Gy /= self.larmor_frequency

        self.or_term = self.off_resonance.copy()

    def kalpha(self, grad_integral):
        return self.larmor_frequency * grad_integral

    def kx(self, t):
        grad_integral = np.sum(self.Gx[0 : t + 1])
        return self.kalpha(grad_integral) / self.shape[1]

    def ky(self, t):
        grad_integral = np.sum(self.Gy[0 : t + 1])
        return self.kalpha(grad_integral) / self.shape[0]


    def get_sample(self, t):
        kx = self.kx(t)
        ky = self.ky(t)
        print("                                     ", end="\r")
        print(f"kx: {kx:.3f}, ky: {ky:.3f}", end="\r")
        res = 1 / self.shape[0]
        self.or_term[np.round(ky / res).astype(int), np.round(kx / res).astype(int)] *= t
        return np.sum(self.signal * np.exp(-1j * 2 * np.pi * (self.off_resonance * t + kx * self.xx + ky * self.yy)))


def subsample_kspace(k_space, cut):
    q1 = k_space[0:cut, 0:cut]
    q2 = k_space[0:cut, -cut:]
    q3 = k_space[-cut:, 0:cut]
    q4 = k_space[-cut:, -cut:]
    h1 = np.concatenate((q1, q2), axis=1)
    h2 = np.concatenate((q3, q4), axis=1)
    return np.concatenate((h1, h2), axis=0)


def apply_window(data):
    win = window("hamming", data.shape)
    return np.fft.ifftshift(np.fft.fftshift(data) * win)


if __name__ == "__main__":
    # Load the data
    data = loadmat("phantom.mat")
    phantom = data["phantom"]

    sys = MRSystem(phantom, base_resolution=100, off_resonance=True)
    even = np.zeros(sys.shape).astype(np.complex64)
    odd = np.zeros(sys.shape).astype(np.complex64)
    for t in range(np.multiply.reduce(sys.shape)):
        m = np.round(t // sys.shape[0])
        n = np.round(t % sys.shape[1])
        if m % 2 == 0:
            even[m, n] = sys.get_sample(t)
        else:
            odd[m, sys.shape[1] - n - 1] = sys.get_sample(t)
    print()
    kspace = even + odd
    orig_kspace = subsample_kspace(np.fft.fft2(phantom), 5)

    # apply hamming window
    kspace_win = apply_window(kspace)
    orig_kspace_win = apply_window(orig_kspace)
    recon = np.fft.ifft2(kspace_win)
    orig_recon = np.fft.ifft2(orig_kspace_win)

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
    plt.imshow(np.abs(orig_recon), cmap="gray")
    f.add_subplot(2, 2, 4)
    plt.imshow(sys.off_resonance)
    plt.show()
