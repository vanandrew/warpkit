import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

# import simplebrainviewer as sbv

sns.set(font="Lato", style="dark")
plt.style.use("dark_background")

frame_ref = nib.load("/home/vanandrew/Data/test/run01_fmap_masked.nii.gz")
frame_mov = nib.load("/home/vanandrew/Data/test/run04_01space_fmap_masked.nii.gz")

data_ref = frame_ref.get_fdata()
data_mov = frame_mov.get_fdata()
diff = data_mov - data_ref

f = plt.figure(figsize=(9, 6), layout="constrained")

# pixel/Hz
pix_per_hz = 0.000279996 * 110

v_min = -200
v_max = 200
v_min_px = v_min * pix_per_hz
v_max_px = v_max * pix_per_hz

slice_num = 25

ax1 = f.add_subplot(2, 3, 1)
im1 = ax1.imshow(data_ref[:, :, slice_num].T, origin="lower", cmap="gray", vmin=-100, vmax=200)
ax1.set_xticklabels([])
ax1.set_yticklabels([])

ax1_2 = f.add_subplot(2, 3, 1)
ax1_2.set_visible(False)

ax2 = f.add_subplot(2, 3, 2)
im2 = ax2.imshow(data_mov[:, :, slice_num].T, origin="lower", cmap="gray", vmin=-100, vmax=200)
ax2.set_xticklabels([])
ax2.set_yticklabels([])

axA = f.add_subplot(2, 3, 3)
imA = axA.imshow(diff[:, :, slice_num].T, origin="lower", cmap="icefire", vmin=v_min, vmax=v_max)
axA.set_xticklabels([])
axA.set_yticklabels([])

axA_2 = f.add_subplot(2, 3, 3)
axA_2.set_visible(False)

# slice num
slice_num = 55

ax3 = f.add_subplot(2, 3, 4)
im3 = ax3.imshow(data_ref[slice_num, :, :].T, origin="lower", cmap="gray", vmin=-100, vmax=200)
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_xlabel("(A) Frame 1")

ax3_2 = f.add_subplot(2, 3, 4)
ax3_2.set_visible(False)

ax4 = f.add_subplot(2, 3, 5)
im4 = ax4.imshow(data_mov[slice_num, :, :].T, origin="lower", cmap="gray", vmin=-100, vmax=200)
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.set_xlabel("(B) Frame 2")

axB = f.add_subplot(2, 3, 6)
imB = axB.imshow(diff[slice_num, :, :].T, origin="lower", cmap="icefire", vmin=v_min, vmax=v_max)
axB.set_xticklabels([])
axB.set_yticklabels([])
axB.set_xlabel("(C) Difference\n (Frame 1 - Frame 2)")

axB_2 = f.add_subplot(2, 3, 6)
imB_2 = axB_2.imshow(pix_per_hz * diff[slice_num, :, :].T, origin="lower", cmap="icefire", vmin=v_min_px, vmax=v_max_px)
axB_2.set_visible(False)

cbar = f.colorbar(im1, ax=[ax1_2, ax3_2], aspect=30, pad=0.25, location="left", orientation="vertical")
cbar.ax.yaxis.set_ticks_position("left")
cbar.ax.set_ylabel("Hz", rotation=90)
cax = cbar.ax.twinx()
cax.set_ylim(-3.08, 6.16)
cax.set_ylabel("pixels", rotation=90)

cbar = f.colorbar(imB_2, ax=[axA_2, axB_2], aspect=30, pad=0.35, location="right", orientation="vertical")
cbar.ax.set_ylabel("pixels", rotation=270)
cax = cbar.ax.twinx()
cax.set_ylim(v_min, v_max)
cax.yaxis.set_ticks_position("left")
cax.set_ylabel("Hz", labelpad=-40, rotation=270)
cbar.ax.yaxis.set_ticks_position("right")

plt.show()
# sbv.plot_brain(diff, limits=(-200, 200))
