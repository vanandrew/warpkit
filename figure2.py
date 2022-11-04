import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

sns.set(font="Lato", style="white")

# romeo fmap
romeo_fmap = nib.load('test2/fieldmap.nii.gz')

# brain mask
mask = nib.load('test2/ref_bet_mask.nii.gz')

# mask data
romeo = romeo_fmap.get_fdata() * mask.get_fdata()[:, :, :, np.newaxis]

f = plt.figure(figsize=(9, 5), layout="tight")

axm = f.add_subplot(1, 1, 1)
axm.set_autoscale_on(True)
axm.set_visible(False)

# slice num
slice_num = 24
frames_to_show = [0, 20, 40, 60, 80, 100, 120, 140]

for n in range(len(frames_to_show)):
    frame = frames_to_show[n]
    ax = f.add_subplot(2, 4, n + 1)
    ax.imshow(romeo[:, :, slice_num, frame].T, origin='lower', cmap='gray', vmin=-100, vmax=200)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel(f"Frame {frame}")
plt.show()
