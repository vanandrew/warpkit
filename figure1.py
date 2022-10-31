import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

sns.set(font="Lato", style="white")

# romeo fmap
romeo_fmap = nib.load('test2/fmap.nii.gz')

# topup fmap
topup_fmap = nib.load('test2/fieldmap_d.nii.gz')

# brain mask
mask = nib.load('test2/ref_bet_mask.nii.gz')

# mask data
romeo = romeo_fmap.get_data() * mask.get_data()
topup = topup_fmap.get_data() * mask.get_data()

# compute absolute difference
diff = np.abs(romeo - topup)

f = plt.figure(figsize=(9, 5), layout="tight")

axm = f.add_subplot(1, 1, 1)
axm.set_autoscale_on(True)
axm.set_visible(False)

# slice num
slice_num = 24

ax1 = f.add_subplot(3, 3, 1)
ax1.imshow(romeo[..., slice_num].T, origin='lower', cmap='gray')
ax1.set_xticklabels([])
ax1.set_yticklabels([])

ax2 = f.add_subplot(3, 3, 2)
ax2.imshow(topup[..., slice_num].T, origin='lower', cmap='gray')
ax2.set_xticklabels([])
ax2.set_yticklabels([])

ax3 = f.add_subplot(3, 3, 3)
ax3.imshow(diff[..., slice_num].T, origin='lower')
ax3.set_xticklabels([])
ax3.set_yticklabels([])

# slice num
slice_num = 55

ax4 = f.add_subplot(3, 3, 4)
im4 = ax4.imshow(romeo[slice_num, :, :].T, origin='lower', cmap='gray')
ax4.set_xticklabels([])
ax4.set_yticklabels([])

ax5 = f.add_subplot(3, 3, 5)
ax5.imshow(topup[slice_num, :, :].T, origin='lower', cmap='gray')
ax5.set_xticklabels([])
ax5.set_yticklabels([])

ax6 = f.add_subplot(3, 3, 6)
im6 = ax6.imshow(diff[slice_num, :, :].T, origin='lower')
ax6.set_xticklabels([])
ax6.set_yticklabels([])

# slice num
slice_num = 55

ax7 = f.add_subplot(3, 3, 7)
ax7.imshow(romeo[:, slice_num, :].T, origin='lower', cmap='gray')
ax7.set_xticklabels([])
ax7.set_yticklabels([])
ax7.set_xlabel('ME Fieldmap')

ax8 = f.add_subplot(3, 3, 8)
ax8.imshow(topup[:, slice_num, :].T, origin='lower', cmap='gray')
ax8.set_xticklabels([])
ax8.set_yticklabels([])
ax8.set_xlabel('RPE Fieldmap')

ax9 = f.add_subplot(3, 3, 9)
ax9.imshow(diff[:, slice_num, :].T, origin='lower')
ax9.set_xticklabels([])
ax9.set_yticklabels([])
ax9.set_xlabel('Absolute Difference (Hz)')

dividerl = make_axes_locatable(axm)
caxl = dividerl.append_axes("left", size="1%", pad=0.25)
f.colorbar(im4, cax=caxl, orientation='vertical')

dividerr = make_axes_locatable(axm)
caxr = dividerr.append_axes("right", size="1%", pad=0.25)
f.colorbar(im6, cax=caxr, orientation='vertical')
caxr.yaxis.set_ticks_position('left')

plt.show()
