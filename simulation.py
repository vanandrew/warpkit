import matplotlib.pyplot as plt
import numpy as np
<<<<<<< HEAD
import scipy as sp
from scipy.io import loadmat
=======
from scipy.io import loadmat
from numpy.linalg import lstsq
>>>>>>> c7e2916ec3795c9abfcf1f90ca577bf7db7cef53
from skimage.filters import gaussian


if __name__ == "__main__":
    # load phantom data
<<<<<<< HEAD
    phantom_data = loadmat("phantom.mat")
=======
    phantom_data = loadmat("phantom3.mat")
>>>>>>> c7e2916ec3795c9abfcf1f90ca577bf7db7cef53

    # get the phantom
    phantom = phantom_data["phantom"]

<<<<<<< HEAD
    # create timing information for off-resonance term
    t = np.arange(phantom.shape[0] * phantom.shape[1])
    t = t.reshape(phantom.shape)

    # EPI readout needs to be flipped for even and odd lines
    odd_indices = np.s_[1 : phantom.shape[1] : 2]
    t[odd_indices, :] = t[odd_indices, ::-1]

    # setup off resonance artifact
    off_resonance = np.zeros(phantom.shape)
    mid_point = phantom.shape[0] // 2
    off_resonance[5:30, mid_point - 30 : mid_point + 30] = 2e-3
    off_resonance = gaussian(off_resonance, sigma=(10, 20))

    # for each y line, do an fft
    data = np.zeros(phantom.shape, dtype=np.complex64)
=======
    # get phantom resolution
    phantom_resolution = phantom.shape[0]

    # set base resolution
    base_resolution = 100
    size_factor = phantom_resolution // base_resolution

    # create timing information for off-resonance term
    t = np.arange(base_resolution * base_resolution)
    t = t.reshape((base_resolution, base_resolution))

    # EPI readout needs to be flipped for even and odd lines
    odd_indices = np.s_[1:base_resolution:2]
    t[odd_indices, :] = t[odd_indices, ::-1]

    # create T2* map for phantom
    T2_star = np.ones((base_resolution, base_resolution)) * 100000
    for value in np.unique(phantom):
        break
        if value >= 0.95:
            T2_star[np.isclose(phantom, value)] = 10000
        elif value >= 1e-2:
            T2_star[np.isclose(phantom, value)] = 10000 * value

    # setup off resonance artifact
    off_resonance = np.zeros(phantom.shape)
    mid_point = phantom_resolution // 2
    off_resonance[5 : mid_point // 4, mid_point - mid_point // 4 : mid_point + mid_point // 4] = 5e-4
    off_resonance = gaussian(off_resonance, sigma=(40, 40))

    # for each y line, do an fft
    data = np.zeros((base_resolution, base_resolution), dtype=np.complex64)
    data_no = np.zeros((base_resolution, base_resolution), dtype=np.complex64)
>>>>>>> c7e2916ec3795c9abfcf1f90ca577bf7db7cef53

    # get position and k-space arrays
    pos = np.arange(phantom.shape[1])
    xx, yy = np.meshgrid(pos, pos, indexing="xy")
    k = np.arange(phantom.shape[1]) / phantom.shape[1]

<<<<<<< HEAD
    # for each x line, simulate the data collection with off-resonance term
    for x in range(phantom.shape[1]):
        for y in range(phantom.shape[0]):
            print(f"Sampling Simulated K-space: {k[x]:.3f}, {k[y]:.3f}      ", end="\r")
            data[y, x] = np.sum(phantom * np.exp(-1j * 2 * np.pi * (off_resonance * t[y, x] + k[x] * xx + k[y] * yy)))
    print("\n")

    # get image
    img = np.fft.ifft2(data)

    # plot the data
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(np.log(np.abs(data)), cmap="gray")
    f.add_subplot(1, 2, 2)
    plt.imshow(np.abs(img), cmap="gray")
=======
    # sub select k space samples that based on base resolution
    sub_range = base_resolution // 2
    selection = np.r_[0:sub_range, phantom_resolution - sub_range : phantom_resolution]
    k = k[selection]
    kxx, kyy = np.meshgrid(k, k, indexing="xy")

    # create arrays for system matrix
    system_matrix_d = np.zeros((base_resolution**2, base_resolution**2), dtype=np.complex64)
    system_matrix_c = np.zeros((base_resolution**2, phantom_resolution**2), dtype=np.complex64)

    # for each x line, simulate the data collection with off-resonance term
    for y in range(base_resolution):
        for x in range(base_resolution):
            print(f"Sampling Simulated K-space: {k[x]:.4f}, {k[y]:.4f}        ", end="\r")
            data[y, x] = np.sum(
                phantom
                * np.exp(-1j * 2 * np.pi * (off_resonance * t[y, x] + k[x] * xx + k[y] * yy))
                # * np.exp(-t[y, x] / T2_star)
            )
            data_no[y, x] = np.sum(
                phantom
                * np.exp(-1j * 2 * np.pi * (k[x] * xx + k[y] * yy))
                # * np.exp(-t[y, x] / T2_star)
            )
            system_matrix_d[y * base_resolution + x, :] = np.exp(
                -1j
                * 2
                * np.pi
                * (
                    off_resonance[::size_factor, ::size_factor] * t[y, x]
                    + k[x] * xx[::size_factor, ::size_factor]
                    + k[y] * yy[::size_factor, ::size_factor]
                )
            ).ravel()
            system_matrix_c[y * base_resolution + x, :] = np.exp(
                -1j * 2 * np.pi * (off_resonance * t[y, x] + k[x] * xx + k[y] * yy)
            ).ravel()
    print("\n")

    # naive reconstruction
    img_naive = np.fft.ifft2(data)

    print("Reconstructing image accounting for off-resonance using least squares...")
    print("Discrete Sampling")
    img = lstsq(system_matrix_d, data.ravel(), rcond=None)[0].reshape(base_resolution, base_resolution)
    print("Continuous Sampling")
    img_c = lstsq(system_matrix_c, data.ravel(), rcond=None)[0].reshape(phantom_resolution, phantom_resolution)

    # no off-resonance reconstruction
    img_no = np.fft.ifft2(data_no)

    # plot the dataprint
    f = plt.figure()
    f.add_subplot(2, 3, 1)
    plt.imshow(np.abs(phantom), cmap="gray")
    plt.title("Original Object")
    f.add_subplot(2, 3, 4)
    plt.imshow(np.abs(img_naive), cmap="gray")
    plt.title("Image w/ off-resonance/Inverse FFT")
    f.add_subplot(2, 3, 2)
    plt.imshow(np.abs(img), cmap="gray")
    plt.title("Image w/ off-resonance/least squares (discrete sampling)")
    f.add_subplot(2, 3, 5)
    plt.imshow(np.abs(img_c), cmap="gray")
    plt.title("Image w/ off-resonance/least squares (continuous sampling)")
    f.add_subplot(2, 3, 3)
    plt.imshow(np.abs(img_no), cmap="gray")
    plt.title("Image w/o off-resonance")
    f.add_subplot(2, 3, 6)
    plt.imshow(np.abs(off_resonance), cmap="gray")
    plt.title("Off-resonance map")
>>>>>>> c7e2916ec3795c9abfcf1f90ca577bf7db7cef53
    plt.show()
