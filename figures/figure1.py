import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib

sns.set(font="Lato", style="dark")
plt.style.use("dark_background")

# romeo fmap
romeo_fmap = nib.load("test2/fmap_smooth.nii.gz")

# topup fmap
topup_fmap = nib.load("test2/fieldmap_d.nii.gz")

# brain mask
mask = nib.load("test2/ref_bet_mask.nii.gz")

# mask data
romeo = romeo_fmap.get_fdata() * mask.get_fdata()
topup = topup_fmap.get_fdata() * mask.get_fdata()

# compute absolute difference
diff = romeo - topup

f = plt.figure(figsize=(9, 8), layout="constrained")

# pixel/Hz
pix_per_hz = 0.000279996 * 110

# slice num
slice_num = 24

v_min = -200
v_max = 200
v_min_px = v_min * pix_per_hz
v_max_px = v_max * pix_per_hz

ax1 = f.add_subplot(3, 3, 1)
ax1.imshow(romeo[..., slice_num].T, origin="lower", cmap="gray", vmin=-100, vmax=200)
ax1.set_xticklabels([])
ax1.set_yticklabels([])

ax1_2 = f.add_subplot(3, 3, 1)
ax1_2.set_visible(False)

ax2 = f.add_subplot(3, 3, 2)
ax2.imshow(topup[..., slice_num].T, origin="lower", cmap="gray", vmin=-100, vmax=200)
ax2.set_xticklabels([])
ax2.set_yticklabels([])

ax3 = f.add_subplot(3, 3, 3)
ax3.imshow(diff[..., slice_num].T, origin="lower", cmap="icefire", vmin=v_min, vmax=v_max)
ax3.set_xticklabels([])
ax3.set_yticklabels([])

ax3_2 = f.add_subplot(3, 3, 3)
ax3_2.set_visible(False)

# slice num
slice_num = 54

ax4 = f.add_subplot(3, 3, 4)
im4 = ax4.imshow(romeo[slice_num, :, :].T, origin="lower", cmap="gray", vmin=-100, vmax=200)
ax4.set_xticklabels([])
ax4.set_yticklabels([])

ax5 = f.add_subplot(3, 3, 5)
ax5.imshow(topup[slice_num, :, :].T, origin="lower", cmap="gray", vmin=-100, vmax=200)
ax5.set_xticklabels([])
ax5.set_yticklabels([])

ax6 = f.add_subplot(3, 3, 6)
im6 = ax6.imshow(diff[slice_num, :, :].T, origin="lower", cmap="icefire", vmin=v_min, vmax=v_max)
ax6.set_xticklabels([])
ax6.set_yticklabels([])

ax6_2 = f.add_subplot(3, 3, 6)
im6_2 = ax6.imshow(pix_per_hz * diff[slice_num, :, :].T, origin="lower", cmap="icefire", vmin=v_min_px, vmax=v_max_px)
ax6_2.set_xticklabels([])
ax6_2.set_yticklabels([])
ax6_2.set_visible(False)

# slice num
slice_num = 54

ax7 = f.add_subplot(3, 3, 7)
ax7.imshow(romeo[:, slice_num, :].T, origin="lower", cmap="gray", vmin=-100, vmax=200)
ax7.set_xticklabels([])
ax7.set_yticklabels([])
ax7.set_xlabel("(A) Field map from\n Multi-echo (ME) phase")

ax8 = f.add_subplot(3, 3, 8)
ax8.imshow(topup[:, slice_num, :].T, origin="lower", cmap="gray", vmin=-100, vmax=200)
ax8.set_xticklabels([])
ax8.set_yticklabels([])
ax8.set_xlabel("(B) Field map from\n Reverse-Phase\n Encode (RPE) method")

ax9 = f.add_subplot(3, 3, 9)
ax9.imshow(diff[:, slice_num, :].T, origin="lower", cmap="icefire", vmin=v_min, vmax=v_max)
ax9.set_xticklabels([])
ax9.set_yticklabels([])
ax9.set_xlabel("(C) Difference (ME - RPE)")

cbar = f.colorbar(im4, ax=[ax1_2, ax4, ax7], aspect=30, pad=0.25, location="left", orientation="vertical")
cbar.ax.yaxis.set_ticks_position("left")
cbar.ax.set_ylabel("Hz", rotation=90)
cax = cbar.ax.twinx()
cax.set_ylim(-3.08, 6.16)
cax.set_ylabel("pixels", rotation=90)

cbar = f.colorbar(im6_2, ax=[ax3_2, ax6, ax9], aspect=30, pad=0.35, location="right", orientation="vertical")
cbar.ax.set_ylabel("pixels", rotation=270)
cax = cbar.ax.twinx()
cax.set_ylim(v_min, v_max)
cax.yaxis.set_ticks_position("left")
cax.set_ylabel("Hz", labelpad=-40, rotation=270)
cbar.ax.yaxis.set_ticks_position("right")

plt.show()
