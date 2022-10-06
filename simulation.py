import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.io import loadmat
from skimage.filters import gaussian


if __name__ == "__main__":
    # load phantom data
    phantom_data = loadmat("phantom.mat")

    # get the phantom
    phantom = phantom_data["phantom"]

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

    # get position and k-space arrays
    pos = np.arange(phantom.shape[1])
    xx, yy = np.meshgrid(pos, pos, indexing="xy")
    k = np.arange(phantom.shape[1]) / phantom.shape[1]

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
    plt.show()
