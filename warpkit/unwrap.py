import logging
from types import SimpleNamespace
from typing import cast, List, Tuple, Optional, Union
import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy.stats import mode
from skimage.filters import threshold_otsu  # type: ignore
from scipy.ndimage import (
    generate_binary_structure,
    binary_erosion,
    binary_dilation,
    binary_fill_holes,
    gaussian_filter,
)
from .model import weighted_regression
from .utilities import (
    rescale_phase,
    corr2_coeff,
    create_brain_mask,
    get_largest_connected_component,
)
from .julia import JuliaContext
from .concurrency import run_executor


FMAP_PROPORTION_HEURISTIC = 0.25
FMAP_AMBIGUIOUS_HEURISTIC = 0.5


def reject_outliers(data, m=2.0):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else np.zeros(len(d))
    return data[s < m]


def get_dual_echo_fieldmap(phases, TEs, mags, mask):
    JULIA = JuliaContext()
    # unwrap the phases
    unwrapped_phases = JULIA.romeo_unwrap4D(  # type: ignore
        phase=phases,
        TEs=TEs,
        weights="romeo",
        mag=mags,
        mask=mask,
        correct_global=True,
        maxseeds=1,
        merge_regions=False,
        correct_regions=False,
    )

    phase_diff = unwrapped_phases[..., 1] - unwrapped_phases[..., 0]
    fieldmap = (1000 / (2 * np.pi)) * phase_diff / (TEs[1] - TEs[0])
    return fieldmap, unwrapped_phases


def mcpc_3d_s(
    mag0: npt.NDArray[np.float32],
    mag1: npt.NDArray[np.float32],
    phase0: npt.NDArray[np.float32],
    phase1: npt.NDArray[np.float32],
    TE0: npt.NDArray[np.float32],
    TE1: npt.NDArray[np.float32],
    mask: npt.NDArray[np.bool_],
    ref: Optional[npt.NDArray[np.float32]] = None,
    ref_mask: Optional[npt.NDArray[np.bool_]] = None,
    wrap_limit: bool = False,
):
    JULIA = JuliaContext()
    signal_diff = mag0 * mag1 * np.exp(1j * (phase1 - phase0))
    mag_diff = np.abs(signal_diff)
    phase_diff = np.angle(signal_diff)
    unwrapped_diff = JULIA.romeo_unwrap3D(  # type: ignore
        phase=phase_diff,
        weights="romeo",
        mag=mag_diff,
        mask=mask,
        correct_global=True,
    )
    voxel_mask = create_brain_mask(mag0, -2)
    # if ref is not None and ref_mask is not None:
    #     best_offset = None
    #     best_error = None
    #     for offset in range(-2, 3):
    #         udiff = unwrapped_diff.copy()
    #         udiff[ref_mask] += offset * 2 * np.pi
    #         error = ((ref[ref_mask] - udiff[ref_mask]) ** 2).sum()
    #         if best_offset is None or error < best_error:
    #             best_error = error
    #             best_offset = offset
    #     unwrapped_diff[mask] += cast(int, best_offset) * 2 * np.pi
    # else:
    #     # check if we are neg or pos domain (move to pos domain)
    #     print(unwrapped_diff[mask].mean())
    #     if unwrapped_diff[mask].mean() < 0:
    #         unwrapped_diff[mask] +=  2 * np.pi
    # brain_mask = create_brain_mask(mag0, -2)
    # unwrapped_diff -= (
    #     mode(
    #         np.round(reject_outliers(unwrapped_diff[brain_mask]) / (2 * np.pi)).astype(int),
    #         axis=0,
    #         keepdims=False,
    #     ).mode
    #     * 2
    #     * np.pi
    # )
    # second_echo_brain_mask = create_brain_mask(mag1, -10)
    # voxel_mean = reject_outliers(unwrapped_diff[second_echo_brain_mask]).mean()
    # voxel_prop = (
    #     np.count_nonzero(unwrapped_diff[second_echo_brain_mask] > 0) / unwrapped_diff[second_echo_brain_mask].shape[0]
    # )
    # global DEBUG
    # if DEBUG:
    #     print(f"voxel mean: {voxel_mean}")
    #     print(f"voxel_prop: {voxel_prop}")
    # # if the mean for diff is negative, and the
    # # proportion of negative voxels is < FMAP_PROPORTION_HEURISTIC, then add 2pi
    # if voxel_mean < 0 and voxel_prop < FMAP_PROPORTION_HEURISTIC:
    #     unwrapped_diff += 2 * np.pi

    # # if the voxel prop is in ambiguous range, first look at the mean of the field map
    # # if it's really negative (< -10 Hz), then add 2pi
    # # do a second check looking at the phase offset
    # # if it is < -1, tne add 2pi
    # if voxel_mean < 0 and voxel_prop >= FMAP_PROPORTION_HEURISTIC and voxel_prop < FMAP_AMBIGUIOUS_HEURISTIC:
    #     # compute initial field map
    #     initial_fieldmap = (1000 / (2 * np.pi)) * unwrapped_diff / (TE1 - TE0)
    #     fmap_mask = create_brain_mask(mag1, -1)
    #     mean_voxel_diff = initial_fieldmap[fmap_mask].mean()
    #     print(f"mean_voxel_diff: {mean_voxel_diff}")
    #     if mean_voxel_diff < -10:
    #         unwrapped_diff += 2 * np.pi
    #     else:
    #         # Last resort, requires an extra unwrapping step

    #         # compute initial phase offset
    #         initial_phase_offset = np.angle(np.exp(1j * (phase0 - ((TE0 * unwrapped_diff) / (TE1 - TE0)))))

    #         # subtract from first and second echo phase
    #         new_phase0 = phase0 - initial_phase_offset

    #         new_unwrapped_phase0 = JULIA.romeo_unwrap3D(  # type: ignore
    #             phase=new_phase0,
    #             weights="romeo",
    #             mag=mag0,
    #             mask=mask,
    #             correct_global=True,
    #         )

    #         # get common domain of voxels
    #         common_domain = mode(
    #             np.round(reject_outliers(new_unwrapped_phase0[brain_mask]) / (2 * np.pi)).astype(int),
    #             axis=0,
    #             keepdims=False,
    #         ).mode
    #         breakpoint()
    #         print(f"common_domain: {common_domain}")

    #         # compute mean of phase offset
    #         mean_phase_offset = initial_phase_offset[brain_mask].mean()
    #         print(f"mean_phase_offset: {mean_phase_offset}")

    #         # if less than -1 then add 2pi
    #         if mean_phase_offset < -1:
    #             unwrapped_diff += 2 * np.pi

    #         # do another field map check
    #         initial_fieldmap = (1000 / (2 * np.pi)) * unwrapped_diff / (TE1 - TE0)
    #         fmap_mask = create_brain_mask(mag1, -1)
    #         mean_voxel_diff = initial_fieldmap[fmap_mask].mean()
    #         print(f"mean_voxel_diff: {mean_voxel_diff}")

    # if DEBUG:
    #     voxel_mean = reject_outliers(unwrapped_diff[second_echo_brain_mask]).mean()
    #     voxel_prop = (
    #         np.unwrapped_diff(unwrapped_diff[second_echo_brain_mask] > 0)
    #         / unwrapped_diff[second_echo_brain_mask].shape[0]
    #     )
    #     print(voxel_mean)
    #     print(voxel_prop)

    # unwrapped_diff -= (
    #     mode(
    #         np.round(reject_outliers(unwrapped_diff[voxel_mask]) / (2 * np.pi)).astype(int),
    #         axis=0,
    #         keepdims=False,
    #     ).mode
    #     * 2
    #     * np.pi
    # )
    phases = np.stack([phase0, phase1], axis=-1)
    mags = np.stack([mag0, mag1], axis=-1)
    TEs = np.array([TE0, TE1])
    all_TEs = np.array([0.0, TE0, TE1])
    proposed_offset = np.angle(np.exp(1j * (phase0 - ((TE0 * unwrapped_diff) / (TE1 - TE0)))))

    # get the new phases
    proposed_phases = phases - proposed_offset[..., np.newaxis]

    # compute the fieldmap
    proposed_fieldmap, proposed_unwrapped_phases = get_dual_echo_fieldmap(proposed_phases, TEs, mags, mask)

    # check if the propossed fieldmap is below 10
    # print(f"proposed_fieldmap: {proposed_fieldmap[voxel_mask].mean()}")
    if proposed_fieldmap[voxel_mask].mean() < -10:
        unwrapped_diff += 2 * np.pi
    # check if the propossed fieldmap is between -10 and 0
    elif proposed_fieldmap[voxel_mask].mean() < 0 and not wrap_limit:
        # look at proportion of voxels that are positive
        voxel_prop = np.count_nonzero(proposed_fieldmap[voxel_mask] > 0) / proposed_fieldmap[voxel_mask].shape[0]

        # if the proportion of positive voxels is less than 0.25, then add 2pi
        if voxel_prop < FMAP_PROPORTION_HEURISTIC:
            unwrapped_diff += 2 * np.pi
        elif voxel_prop < FMAP_AMBIGUIOUS_HEURISTIC:
            # compute mean of phase offset
            mean_phase_offset = proposed_offset[voxel_mask].mean()
            # print(f"mean_phase_offset: {mean_phase_offset}")
            # if less than -1 then
            if mean_phase_offset < -1:
                phase_fits = np.concatenate(
                    (np.zeros((*proposed_unwrapped_phases.shape[:-1], 1)), proposed_unwrapped_phases), axis=-1
                )
                _, residuals_1, _, _, _ = np.polyfit(all_TEs, phase_fits[voxel_mask, :].T, 1, full=True)

                # check if adding 2pi makes it better
                new_proposed_offset = np.angle(
                    np.exp(1j * (phase0 - ((TE0 * (unwrapped_diff + 2 * np.pi)) / (TE1 - TE0))))
                )
                new_proposed_phases = phases - new_proposed_offset[..., np.newaxis]
                new_proposed_fieldmap, new_proposed_unwrapped_phases = get_dual_echo_fieldmap(
                    new_proposed_phases, TEs, mags, mask
                )
                new_voxel_prop = (
                    np.count_nonzero(new_proposed_fieldmap[voxel_mask] > 0) / new_proposed_fieldmap[voxel_mask].shape[0]
                )
                # fit linear model to the proposed phases
                new_phase_fits = np.concatenate(
                    (np.zeros((*new_proposed_unwrapped_phases.shape[:-1], 1)), new_proposed_unwrapped_phases), axis=-1
                )
                _, residuals_2, _, _, _ = np.polyfit(all_TEs, new_phase_fits[voxel_mask, :].T, 1, full=True)
                print(f"mean_proposed_fieldmap 1: {proposed_fieldmap[voxel_mask].mean()}")
                print(f"voxel_prop 1 : {voxel_prop}")
                print(f"mean_residuals 1: {residuals_1.mean()}")
                print(f"mean_phase_offset 1: {mean_phase_offset}")
                print(f"proposed_fieldmap 2: {new_proposed_fieldmap[voxel_mask].mean()}")
                print(f"voxel_prop 2: {new_voxel_prop}")
                print(f"mean_residuals 2: {residuals_2.mean()}")
                print(f"mean_phase_offset 2: {new_proposed_offset.mean()}")
                if (
                    np.isclose(residuals_1.mean(), residuals_2.mean(), atol=1e-3, rtol=1e-3)
                    and new_proposed_fieldmap[voxel_mask].mean() > 0
                ):
                    unwrapped_diff += 2 * np.pi
                else:
                    unwrapped_diff -= 2 * np.pi
                    # new_proposed_offset = np.angle(
                    #     np.exp(1j * (phase0 - ((TE0 * (unwrapped_diff - 2 * np.pi)) / (TE1 - TE0))))
                    # )
                    # new_proposed_phases = phases - new_proposed_offset[..., np.newaxis]
                    # new_proposed_fieldmap, new_proposed_unwrapped_phases = get_dual_echo_fieldmap(
                    #     new_proposed_phases, TEs, mags, mask
                    # )
                    # new_voxel_prop = (
                    #     np.count_nonzero(new_proposed_fieldmap[voxel_mask] > 0)
                    #     / new_proposed_fieldmap[voxel_mask].shape[0]
                    # )
                    # # fit linear model to the proposed phases
                    # new_phase_fits = np.concatenate(
                    #     (np.zeros((*new_proposed_unwrapped_phases.shape[:-1], 1)), new_proposed_unwrapped_phases),
                    #     axis=-1,
                    # )
                    # _, residuals_2, _, _, _ = np.polyfit(all_TEs, new_phase_fits[voxel_mask, :].T, 1, full=True)
                    # print(f"proposed_fieldmap 3: {new_proposed_fieldmap[voxel_mask].mean()}")
                    # print(f"voxel_prop 3: {new_voxel_prop}")
                    # print(f"mean_residuals 3: {residuals_2.mean()}")
                    # print(f"mean_phase_offset 3: {new_proposed_offset.mean()}")
                    # if (
                    #     np.isclose(residuals_1.mean(), residuals_2.mean(), atol=1e-3, rtol=1e-3)
                    #     and new_proposed_fieldmap[voxel_mask].mean() > 0
                    # ):
                    #     unwrapped_diff -= 2 * np.pi

                # if new_proposed_fieldmap.mean() > 0 and np.isclose(
                #     residuals_2.mean(), residuals_1.mean(), atol=1e-3, rtol=1e-3
                # ):
                #     # print(f"proposed_fieldmap 2: {new_proposed_fieldmap.mean()}")
                #     # print(f"voxel_prop 2: {new_voxel_prop}")
                #     # print(f"mean_residuals 2: {residuals_2.mean()}")
                #     unwrapped_diff += 2 * np.pi
                # else:
                #     print(f"mean_proposed_fieldmap 1: {proposed_fieldmap[voxel_mask].mean()}")
                #     print(f"voxel_prop 1 : {voxel_prop}")
                #     print(f"mean_residuals 1: {residuals_1.mean()}")
                #     print(f"mean_phase_offset 1: {mean_phase_offset}")
                #     print(f"proposed_fieldmap 2: {new_proposed_fieldmap.mean()}")
                #     print(f"voxel_prop 2: {new_voxel_prop}")
                #     print(f"mean_residuals 2: {residuals_2.mean()}")
                #     print(f"mean_phase_offset 2: {new_proposed_offset.mean()}")
                #     # check if subtracting 2pi makes it better
                #     new_proposed_offset = np.angle(
                #         np.exp(1j * (phase0 - ((TE0 * (unwrapped_diff - 2 * np.pi)) / (TE1 - TE0))))
                #     )

                #     # get the new phases
                #     new_proposed_phases = phases - new_proposed_offset[..., np.newaxis]

                #     # compute the fieldmap
                #     new_proposed_fieldmap, new_proposed_unwrapped_phases = get_dual_echo_fieldmap(
                #         new_proposed_phases, TEs, mags, mask
                #     )

                #     # compute the mean of the fieldmap
                #     new_voxel_prop = (
                #         np.count_nonzero(new_proposed_fieldmap[voxel_mask] > 0)
                #         / new_proposed_fieldmap[voxel_mask].shape[0]
                #     )

                #     # fit linear model to the proposed phases
                #     new_phase_fits = np.concatenate(
                #         (np.zeros((*new_proposed_unwrapped_phases.shape[:-1], 1)), new_proposed_unwrapped_phases),
                #         axis=-1,
                #     )
                #     _, residuals_2, _, _, _ = np.polyfit(all_TEs, new_phase_fits[voxel_mask, :].T, 1, full=True)
                #     print(f"proposed_fieldmap 2: {new_proposed_fieldmap.mean()}")
                #     print(f"voxel_prop 2: {new_voxel_prop}")
                #     print(f"mean_residuals 2: {residuals_2.mean()}")
                # if new_voxel_prop < 0.1:
                #     unwrapped_diff += 4 * np.pi
                #     proposed_new_offset = np.angle(np.exp(1j * (phase0 - ((TE0 * unwrapped_diff) / (TE1 - TE0)))))
                #     proposed_new_phases = phases - proposed_new_offset[..., np.newaxis]
                #     proposed_new_fieldmap = get_dual_echo_fieldmap(proposed_new_phases, TEs, mags, mask)
                #     print(f"current_fieldmap: {proposed_fieldmap[voxel_mask].mean()}")
                #     print(f"proposed_new_fieldmap: {proposed_new_fieldmap[voxel_mask].mean()}")
                #     if proposed_new_fieldmap[voxel_mask].mean() < 0:
                #         pass
                #     unwrapped_diff -= 2 * np.pi
                # elif new_voxel_prop > 0.9:
                #     # unwrapped_diff -= 2 * np.pi
                #     pass

        # unwrapped_diff += 2 * np.pi
        # proposed_new_offset = np.angle(np.exp(1j * (phase0 - ((TE0 * unwrapped_diff) / (TE1 - TE0)))))
        # proposed_new_phases = phases - proposed_new_offset[..., np.newaxis]
        # proposed_new_fieldmap = get_dual_echo_fieldmap(proposed_new_phases, TEs, mags, mask)
        # # if adding an offset made it worse, then don't do it
        # print(f"proposed_new_fieldmap: {proposed_new_fieldmap[voxel_mask].mean()}")
        # if proposed_new_fieldmap[voxel_mask].mean() < 0:
        #     unwrapped_diff -= 2 * np.pi
        # else:
        #     # check field in center of brain for negatives
        #     center_mask = create_brain_mask(mag0, -10)
        #     proposed_new_fieldmap_center = proposed_new_fieldmap[center_mask].mean()
        #     print(f"proposed_new_fieldmap_center: {proposed_new_fieldmap_center}")

    # # now loop over potential phase offsets, choosing the one that has the smallest mean fieldmap
    # smallest_mean = 0
    # smallest_offset = None
    # for idx, i in enumerate([0, 0, -1, 1]):
    #     proposed_unwrapped_diff = unwrapped_diff + i * 2 * np.pi
    #     proposed_offset = np.angle(np.exp(1j * (phase0 - ((TE0 * proposed_unwrapped_diff) / (TE1 - TE0)))))

    #     # get the new phases
    #     proposed_phases = phases - proposed_offset[..., np.newaxis]

    #     # unwrap the phases
    #     proposed_unwrapped_phases = JULIA.romeo_unwrap4D(  # type: ignore
    #         phase=proposed_phases,
    #         TEs=TEs,
    #         weights="romeo",
    #         mag=mags,
    #         mask=mask,
    #         correct_global=True,
    #         maxseeds=1,
    #         merge_regions=False,
    #         correct_regions=False,
    #     )

    #     # get fieldmap
    #     proposed_phase_diff = proposed_unwrapped_phases[..., 1] - proposed_unwrapped_phases[..., 0]
    #     proposed_fieldmap = (1000 / (2 * np.pi)) * proposed_phase_diff / (TE1 - TE0)
    #     mean_proposed_fieldmap = proposed_fieldmap[voxel_mask].mean()

    #     if idx == 0:
    #         if mean_proposed_fieldmap < -5:
    #             unwrapped_diff += 2 * np.pi
    #             continue
    #         else:
    #             continue

    #     # check the mean of the unwrapped phase 0
    #     mean_unwrapped_phase0 = proposed_unwrapped_phases[voxel_mask, 0].mean()
    #     if DEBUG:
    #         print(f"mean_unwrapped_phase0: {mean_unwrapped_phase0}")
    #         print(f"mean_proposed_fieldmap: {mean_proposed_fieldmap}")

    #     # fit linear model to the proposed phases
    #     phase_fits = np.concatenate(
    #         (np.zeros((*proposed_unwrapped_phases.shape[:-1], 1)), proposed_unwrapped_phases), axis=-1
    #     )
    #     _, residuals, _, _, _ = np.polyfit(all_TEs, phase_fits[voxel_mask, :].T, 1, full=True)
    #     if DEBUG:
    #         print(f"mean_residuals: {residuals.mean()}")

    #     if smallest_offset is None:
    #         smallest_mean = residuals.mean()
    #         smallest_offset = i
    #     elif residuals.mean() < smallest_mean:
    #         # if it's close don't do anything
    #         if np.isclose(residuals.mean(), smallest_mean, atol=1e-3, rtol=1e-3):
    #             continue
    #         smallest_mean = residuals.mean()
    #         smallest_offset = i

    # add offset to diff
    # unwrapped_diff += smallest_offset * 2 * np.pi

    # compute the phase offset
    return np.angle(np.exp(1j * (phase0 - ((TE0 * unwrapped_diff) / (TE1 - TE0))))), unwrapped_diff


def unwrap_phase(
    phase_data: npt.NDArray[np.float32],
    mag_data: npt.NDArray[np.float32],
    TEs: npt.NDArray[np.float32],
    mask_data: npt.NDArray[np.bool_],
    automask: bool = True,
    automask_dilation: int = 3,
    idx: Union[int, None] = None,
    wrap_limit: bool = False,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int8]]:
    """Unwraps the phase for a single frame of ME-EPI

    Parameters
    ----------
    phase_data : npt.NDArray[np.float32]
        Single frame of phase data with shape (x, y, z, echo)
    mag_data : npt.NDArray[np.float32]
        Single frame of magnitude data with shape (x, y, z, echo)
    TEs : npt.NDArray[np.float32]
        Echo times associated with each phase
    mask_data : npt.NDArray[np.bool_]
        Mask of voxels to use for unwrapping
    automask : bool, optional
        Automatically compute a mask, by default True
    automask_dilation : int, optional
        Number of extra dilations (or erosions if negative) to perform, by default 3
    idx : int, optional
        Index of the frame being processed for verbosity, by default None

    Returns
    -------
    npt.NDArray[np.float32]
        unwrapped phase in radians
    npt.NDArray[np.int8]
        mask
    """
    # get Julia Context
    JULIA = JuliaContext()

    if idx is not None:
        logging.info(f"Processing frame: {idx}")

    # if automask is True, generate a mask for the frame, instead of using mask_data
    if automask:
        # get the index with the shortest echo time
        echo_idx = np.argmin(TEs)
        mag_shortest = mag_data[..., echo_idx]
        # the theory goes like this, the magnitude/otsu base mask can be too aggressive occasionally
        # and the voxel quality mask can get extra voxels that are not brain, but is noisy
        # so we combine the two masks to get a better mask
        vq = JULIA.romeo_voxelquality(phase_data, TEs, np.ones(shape=mag_data.shape, dtype=np.float32))  # type: ignore
        vq_mask = vq > threshold_otsu(vq)
        strel = generate_binary_structure(3, 2)
        vq_mask = cast(npt.NDArray[np.bool_], binary_fill_holes(vq_mask, strel))
        # get largest connected component
        vq_mask = get_largest_connected_component(vq_mask)

        # combine masks
        echo_idx = np.argmin(TEs)
        mag_shortest = mag_data[..., echo_idx]
        brain_mask = create_brain_mask(mag_shortest)
        combined_mask = brain_mask | vq_mask
        combined_mask = get_largest_connected_component(combined_mask)

        # erode then dilate
        combined_mask = cast(npt.NDArray[np.bool_], binary_erosion(combined_mask, strel, iterations=2))
        combined_mask = get_largest_connected_component(combined_mask)
        combined_mask = cast(npt.NDArray[np.bool_], binary_dilation(combined_mask, strel, iterations=2))

        # get a dilated verision of the mask
        combined_mask_dilated = cast(
            npt.NDArray[np.bool_], binary_dilation(combined_mask, strel, iterations=automask_dilation)
        )

        # get sum of masks (we can select dilated vs original version by indexing)
        mask_data_select = combined_mask.astype(np.int8) + combined_mask_dilated.astype(np.int8)

        # let mask_data be the dilated version
        mask_data = mask_data_select > 0

    # Do MCPC-3D-S algo to compute phase offset
    # first pass is used to get a reference unwrapping to make sure the phase offset is in the correct domain
    # ref_mask = create_brain_mask(mag_data[..., 0], -10).astype(bool)
    # _, unwrapped_phase_ref = mcpc_3d_s(
    #     mag_data[..., 0] * ref_mask,
    #     mag_data[..., 1] * ref_mask,
    #     phase_data[..., 0] * ref_mask,
    #     phase_data[..., 1] * ref_mask,
    #     TEs[0],
    #     TEs[1],
    #     ref_mask,
    # )
    # nib.Nifti1Image(unwrapped_phase_ref, np.eye(4)).to_filename(f"unwrapped_ref{idx}.nii")
    # nib.Nifti1Image(ref_mask.astype('f8'), np.eye(4)).to_filename(f"ref_mask{idx}.nii")
    phase_offset, unwrapped_diff = mcpc_3d_s(
        mag_data[..., 0],
        mag_data[..., 1],
        phase_data[..., 0],
        phase_data[..., 1],
        TEs[0],
        TEs[1],
        mask_data,
        wrap_limit=wrap_limit,
        # ref_mask,
    )
    global DEBUG
    if DEBUG:
        global affine
        global header
        nib.Nifti1Image(phase_offset, affine, header).to_filename(f"phase_offset{idx}.nii")
        nib.Nifti1Image(unwrapped_diff, affine, header).to_filename(f"ud{idx}.nii")
    # remove phase offset from data
    phase_data -= phase_offset[..., np.newaxis]

    # unwrap the phase data
    unwrapped = JULIA.romeo_unwrap4D(  # type: ignore
        phase=phase_data,
        TEs=TEs,
        weights="romeo",
        mag=mag_data,
        mask=mask_data,
        correct_global=True,
        maxseeds=1,
        merge_regions=False,
        correct_regions=False,
    )

    # global mode correction
    # this computes the global mode offset for the first echo then tries to find the offset
    # that minimizes the residuals for each subsequent echo
    # use auto mask to get brain mask
    echo_idx = np.argmin(TEs)
    mag_shortest = mag_data[..., echo_idx]
    brain_mask = create_brain_mask(mag_shortest)
    # eroded_brain_mask = create_brain_mask(mag_shortest, -2)

    # global mode offset the first echo
    # print(mode(np.round(unwrapped[brain_mask, 0] / (2 * np.pi)).astype(int), axis=0, keepdims=False).mode * 2 * np.pi)
    # unwrapped -= (
    #     mode(
    #         np.round(reject_outliers(unwrapped[eroded_brain_mask, 0]) / (2 * np.pi)).astype(int),
    #         axis=0,
    #         keepdims=False,
    #     ).mode
    #     * 2
    #     * np.pi
    # )
    # global mode offset sometimes on wrong side, so if the mean for last echo negative, then add 2pi
    # reestimated_phase_at_1ms = unwrapped[..., 0] / TEs[0]
    # print(reestimated_phase_at_1ms[eroded_brain_mask].mean())

    # for the first echo, find the global offset that's closest to the estimated first echo phase
    # computed from the unwrapped_diff phase
    # estimated_phase_at_echo1 = (TEs[0] * unwrapped_diff) / (TEs[1] - TEs[0])
    # print((estimated_phase_at_echo1[eroded_brain_mask] - unwrapped[eroded_brain_mask, 0]))
    # offset = mode(
    #     np.round((estimated_phase_at_echo1[eroded_brain_mask] - unwrapped[eroded_brain_mask, 0]) / (2 * np.pi)).astype(
    #         int
    #     ),
    #     axis=0,
    #     keepdims=False,
    # ).mode
    # print(offset)
    # unwrapped += offset * 2 * np.pi

    # subsequent echoes are corrected by minimizing fit to linear model from 1st echo

    # for each of these matrices TEs are on rows, voxels are columns
    # get design matrix
    X = TEs[:, np.newaxis]

    # get magnitude weight matrix
    W = mag_data[brain_mask, :].T

    # loop over each index past 1st echo
    for i in range(1, TEs.shape[0]):
        # get matrix with the masked unwrapped data (weighted by magnitude)
        Y = unwrapped[brain_mask, :].T

        # Compute offset through linear regression method
        best_offset = compute_offset(i, W, X, Y)

        # apply the offset
        unwrapped[..., i] += 2 * np.pi * best_offset

    # # examine second echo, look at proportion of voxels that are positive
    # second_echo_brain_mask = create_brain_mask(mag_data[..., 1], -10)
    # voxel_mean = reject_outliers(unwrapped[second_echo_brain_mask, 1]).mean()
    # voxel_prop = (
    #     np.count_nonzero(unwrapped[second_echo_brain_mask, 1] > 0) / unwrapped[second_echo_brain_mask, 1].shape[0]
    # )
    # # print(voxel_mean)
    # # print(voxel_prop)
    # # if in the wrong mean domain for second echo, add 2pi to first echo
    # if voxel_mean < 0 and voxel_prop < FMAP_PROPORTION_HEURISTIC:
    #     # add 2pi to 1st echo
    #     unwrapped[..., 0] += 2 * np.pi
    #     # loop over each index past 1st echo
    #     for i in range(1, TEs.shape[0]):
    #         # get matrix with the masked unwrapped data (weighted by magnitude)
    #         Y = unwrapped[brain_mask, :].T

    #         # Compute offset through linear regression method
    #         best_offset = compute_offset(i, W, X, Y)

    #         # apply the offset
    #         unwrapped[..., i] += 2 * np.pi * best_offset

    # set anything outside of mask_data to 0
    unwrapped[~mask_data] = 0

    # set final mask to return
    if automask:
        final_mask = mask_data_select  # type: ignore
    else:
        final_mask = mask_data.astype(np.int8)

    # return the unwrapped data
    return unwrapped, final_mask


def check_temporal_consistency_corr(
    unwrapped_data: npt.NDArray,
    unwrapped_echo_1: npt.NDArray,
    TEs,
    mag: List[nib.Nifti1Image],
    t,
    frame_idx,
    masks: npt.NDArray,
    threshold: float = 0.98,
):
    """Ensures phase unwrapping solutions are temporally consistent

    This uses correlation as a similarity metric between frames to enforce temporal consistency.

    Parameters
    ----------
    unwrapped_data : npt.NDArray
        unwrapped phase data, where last column is time, and second to last column are the echoes
    TEs : npt.NDArray
        echo times
    mag : List[nib.Nifti1Image]
        magnitude images
    frames : List[int]
        list of frames that are being processed
    threshold : float
        threshold for correlation similarity. By default 0.98
    """

    logging.info(f"Computing temporal consistency check for frame: {t}")

    # generate brain mask (with 1 voxel erosion)
    echo_idx = np.argmin(TEs)
    mag_shortest = mag[echo_idx].dataobj[..., frame_idx]
    brain_mask = create_brain_mask(mag_shortest, -1)

    # get the current frame phase
    current_frame_data = unwrapped_echo_1[brain_mask, t][:, np.newaxis]

    # get the correlation between the current frame and all other frames
    corr = corr2_coeff(current_frame_data, unwrapped_echo_1[brain_mask, :]).ravel()

    # threhold the RD
    tmask = corr > threshold

    # get indices of mask
    indices = np.where(tmask)[0]

    # get mask for frame
    mask = masks[..., t] > 0

    # for each frame compute the mean value along the time axis (masked by indices and mask)
    mean_voxels = np.mean(unwrapped_echo_1[mask][:, indices], axis=-1)

    # for this frame figure out the integer multiple that minimizes the value to the mean voxel
    int_map = np.round((mean_voxels - unwrapped_echo_1[mask, t]) / (2 * np.pi)).astype(int)

    # correct the data using the integer map
    unwrapped_data[mask, 0, t] += 2 * np.pi * int_map

    # format weight matrix
    weights_mat = np.stack([m.dataobj[..., frame_idx] for m in mag], axis=-1)[mask].T

    # form design matrix
    X = TEs[:, np.newaxis]

    # fit subsequent echos to the weighted linear regression from the first echo
    for echo in range(1, unwrapped_data.shape[-2]):
        # form response matrix
        Y = unwrapped_data[mask, :echo, t].T

        # fit model to data
        coefficients, _ = weighted_regression(X[:echo], Y, weights_mat[:echo])

        # get the predicted values for this echo
        Y_pred = coefficients * TEs[echo]

        # compute the difference and get the integer multiple map
        int_map = np.round((Y_pred - unwrapped_data[mask, echo, t]) / (2 * np.pi)).astype(int)

        # correct the data using the integer map
        unwrapped_data[mask, echo, t] += 2 * np.pi * int_map


def compute_field_map(
    unwrapped_mat: npt.NDArray,
    mag: List[nib.Nifti1Image],
    num_echos: int,
    TEs_mat: npt.NDArray,
    frame_num: int,
) -> npt.NDArray:
    """Function for computing field map for a given frame
    def reject_outliers(data, m = 2.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else np.zeros(len(d))
        return data[s<m]
        Parameters
        ----------
        unwrapped_mat: np.ndarray
            Array of unwrapped phase data for a given frame
        mag: List[nib.NiftiImage]
            List of magnitudes
        num_echos: int
            Number of echos
        TEs_mat: npt.NDArray
            Echo times in a 2d matrix
        frame_num: int
            Frame number

        Returns
        -------
        B0: np.ndarray
    """
    logging.info(f"Computing field map for frame: {frame_num}")
    unwrapped_mat = unwrapped_mat.reshape(-1, num_echos).T
    mag_data = np.stack([m.dataobj[..., frame_num] for m in mag], axis=-1).astype(np.float32)
    weights = mag_data.reshape(-1, num_echos).T
    B0 = weighted_regression(TEs_mat, unwrapped_mat, weights)[0].T.reshape(*mag_data.shape[:3])
    B0 *= 1000 / (2 * np.pi)
    return B0


def compute_offset(echo_ind: int, W: npt.NDArray, X: npt.NDArray, Y: npt.NDArray) -> int:
    """Method for computing the global mode offset for echoes > 1

    Parameters
    ----------
    echo_ind: int
        Echo index
    W: npt.NDArray
        Weights
    X: npt.NDArray
        TEs in 2d matrix
    Y: npt.NDArray
        Masked unwrapped data weighted by magnitude

    Returns
    -------
    best_offset: int
    """
    # fit the model to the up to previous echo
    coefficients, _ = weighted_regression(X[:echo_ind], Y[:echo_ind], W[:echo_ind])

    # compute the predicted phase for the current echo
    Y_pred = X[echo_ind] * coefficients

    # compute the difference between the predicted phase and the unwrapped phase
    Y_diff = Y_pred - Y[echo_ind]

    # compute closest multiple of 2pi to the difference
    int_map = np.round(Y_diff / (2 * np.pi)).astype(int)

    # compute the most often occuring multiple
    best_offset = mode(int_map, axis=0, keepdims=False).mode
    best_offset = cast(int, best_offset)

    return best_offset


def svd_filtering(
    field_maps: npt.NDArray,
    new_masks: npt.NDArray,
    voxel_size: float,
    n_frames: int,
    border_filt: Tuple[int, int],
    svd_filt: int,
):
    if new_masks.max() == 2 and n_frames >= np.max(border_filt):
        logging.info("Performing spatial/temporal filtering of border voxels...")
        smoothed_field_maps = np.zeros(field_maps.shape, dtype=np.float32)
        # smooth by 4 mm kernel
        sigma = (4 / voxel_size) / 2.355
        for i in range(field_maps.shape[-1]):
            smoothed_field_maps[..., i] = gaussian_filter(field_maps[..., i], sigma=sigma)
        # compute the union of all the masks
        union_mask = np.sum(new_masks, axis=-1) > 0
        # do temporal filtering of border voxels with SVD
        U, S, VT = np.linalg.svd(smoothed_field_maps[union_mask], full_matrices=False)
        # first pass of SVD filtering
        recon = np.dot(U[:, : border_filt[0]] * S[: border_filt[0]], VT[: border_filt[0], :])
        recon_img = np.zeros(field_maps.shape, dtype=np.float32)
        recon_img[union_mask] = recon
        # set the border voxels in the field map to the recon values
        for i in range(field_maps.shape[-1]):
            field_maps[new_masks[..., i] == 1, i] = recon_img[new_masks[..., i] == 1, i]
        # do second SVD filtering pass
        U, S, VT = np.linalg.svd(field_maps[union_mask], full_matrices=False)
        # second pass of SVD filtering
        recon = np.dot(U[:, : border_filt[1]] * S[: border_filt[1]], VT[: border_filt[1], :])
        recon_img = np.zeros(field_maps.shape, dtype=np.float32)
        recon_img[union_mask] = recon
        # set the border voxels in the field map to the recon values
        for i in range(field_maps.shape[-1]):
            field_maps[new_masks[..., i] == 1, i] = recon_img[new_masks[..., i] == 1, i]

    # use svd filter to denoise the field maps
    if n_frames >= svd_filt:
        logging.info("Denoising field maps with SVD...")
        logging.info(f"Keeping {svd_filt} components...")
        # compute the union of all the masks
        union_mask = np.sum(new_masks, axis=-1) > 0
        # compute SVD
        U, S, VT = np.linalg.svd(field_maps[union_mask], full_matrices=False)
        # only keep the first n_components components
        recon = np.dot(U[:, :svd_filt] * S[:svd_filt], VT[:svd_filt, :])
        recon_img = np.zeros(field_maps.shape, dtype=np.float32)
        recon_img[union_mask] = recon
        # set the voxel values in the mask to the recon values
        for i in range(field_maps.shape[-1]):
            field_maps[new_masks[..., i] > 0, i] = recon_img[new_masks[..., i] > 0, i]


def unwrap_and_compute_field_maps(
    phase: List[nib.Nifti1Image],
    mag: List[nib.Nifti1Image],
    TEs: Union[List[float], Tuple[float], npt.NDArray[np.float32]],
    mask: Union[nib.Nifti1Image, SimpleNamespace, None] = None,
    automask: bool = True,
    border_size: int = 5,
    border_filt: Tuple[int, int] = (1, 5),
    svd_filt: int = 10,
    frames: Union[List[int], None] = None,
    n_cpus: int = 4,
    debug: bool = False,
    wrap_limit: bool = False,
) -> nib.Nifti1Image:
    """Unwrap phase of data weighted by magnitude data and compute field maps. This makes a call
    to the ROMEO phase unwrapping algorithm for each frame. To learn more about ROMEO, see this paper:

    Dymerska, B., Eckstein, K., Bachrata, B., Siow, B., Trattnig, S., Shmueli, K., Robinson, S.D., 2020.
    Phase Unwrapping with a Rapid Opensource Minimum Spanning TreE AlgOrithm (ROMEO).
    Magnetic Resonance in Medicine. https://doi.org/10.1002/mrm.28563

    Parameters
    ----------
    phase : List[nib.Nifti1Image]
        Phases to unwrap
    mag : List[nib.Nifti1Image]
        Magnitudes associated with each phase
    TEs : Union[List[float], Tuple[float], npt.NDArray[np.float32]]
        Echo times associated with each phase (in ms)
    mask : nib.Nifti1Image, optional
        Boolean mask, by default None
    automask : bool, optional
        Automatically generate a mask (ignore mask option), by default True
    border_size : int, optional
        Size of border in automask, by default 5
    border_filt : Tuple[int, int], optional
        Number of SVD components for each step of border filtering, by default (1, 5)
    svd_filt : int, optional
        Number of SVD components to use for filtering of field maps, by default 30
    frames : List[int], optional
        Only process these frame indices, by default None (which means all frames)
    n_cpus : int, optional
        Number of CPUs to use, by default 4
    debug : bool, optional
        Debug mode, by default False

    Returns
    -------
    nib.Nifti1Image
        Field maps in Hz
    """
    # check TEs if < 0.1, tell user they probably need to convert to ms
    if np.min(TEs) < 0.1:
        logging.warning(
            "WARNING: TEs are unusually small. Your inputs may be incorrect. Did you forget to convert to ms?"
        )

    # convert TEs to np array
    TEs = cast(npt.NDArray[np.float32], np.array(TEs))

    # make sure affines/shapes are all correct
    for p1, m1 in zip(phase, mag):
        for p2, m2 in zip(phase, mag):
            if not (
                np.allclose(p1.affine, p2.affine, rtol=1e-3, atol=1e-3)
                and np.allclose(p1.shape, p2.shape, rtol=1e-3, atol=1e-3)
                and np.allclose(m1.affine, m2.affine, rtol=1e-3, atol=1e-3)
                and np.allclose(m1.shape, m2.shape, rtol=1e-3, atol=1e-3)
                and np.allclose(p1.affine, m1.affine, rtol=1e-3, atol=1e-3)
                and np.allclose(p1.shape, m1.shape, rtol=1e-3, atol=1e-3)
                and np.allclose(p2.affine, m2.affine, rtol=1e-3, atol=1e-3)
                and np.allclose(p2.shape, m2.shape, rtol=1e-3, atol=1e-3)
            ):
                raise ValueError("Affines/Shapes of images do not all match.")

    # check if data is 4D or 3D
    if len(phase[0].shape) == 3:
        # set total number of frames to 1
        n_frames = 1
        # convert data to 4D
        phase = [nib.Nifti1Image(p.get_fdata()[..., np.newaxis], p.affine, p.header) for p in phase]
        mag = [nib.Nifti1Image(m.get_fdata()[..., np.newaxis], m.affine, m.header) for m in mag]
    elif len(phase[0].shape) == 4:
        # if frames is None, set it to all frames
        if frames is None:
            frames = list(range(phase[0].shape[-1]))
        # get the total number of frames
        n_frames = len(frames)
    else:
        raise ValueError("Data must be 3D or 4D.")
    # frames should be a list at this point
    frames = cast(List[int], frames)

    # check echo times = number of mag and phase images
    if len(TEs) != len(phase) or len(TEs) != len(mag):
        raise ValueError("Number of echo times must equal number of mag and phase images.")

    # allocate space for field maps and unwrapped
    field_maps = np.zeros((*phase[0].shape[:3], n_frames), dtype=np.float32)
    unwrapped = np.zeros((*phase[0].shape[:3], len(TEs), n_frames), dtype=np.float32)
    # array for storing auto-generated masks
    new_masks = np.zeros((*mag[0].shape[:3], len(frames)), dtype=np.int8)

    # FOR DEBUGGING
    global DEBUG
    DEBUG = False
    if debug:
        global affine
        affine = phase[0].affine
        global header
        header = phase[0].header
        DEBUG = True

    # allocate mask if needed
    if not mask:
        mask = SimpleNamespace()
        if automask:  # if we are automasking use a fake array that plays nice with the logic
            mask.dataobj = np.ones((1, 1, 1, phase[0].shape[-1]))
        else:
            mask.dataobj = np.ones(phase[0].shape)

    # write a function to iterate over each frame for phase unwrapping
    def phase_iterator(phase, mag, TEs, mask, frames, automask, automask_dilation):
        # note that I separate out idx and frame_idx for the case when the user wants to process a subset of
        # non-contiguous frames
        # this will always reindex the frames to be contiguous however
        # e.g. if the user passes in frames [3, 5, 13] it will be reindexed to [0, 1, 2]

        # estimate the min and max phase values, we assume the phase ranges from -pi to pi
        min_phases = []
        max_phases = []
        for phase_echo in phase:
            # only look at the first frame
            phase_data = phase_echo.dataobj[..., frames[0]]
            # get the min and max phase value
            min_phases.append(phase_data.min())
            max_phases.append(phase_data.max())
        min_phase = mode(min_phases, keepdims=False).mode
        max_phase = mode(max_phases, keepdims=False).mode
        logging.info("Estimated min phase: %f", min_phase)
        logging.info("Estimated max phase: %f", max_phase)

        for idx, frame_idx in enumerate(frames):
            # get the phase and magnitude data from each echo
            phase_data: npt.NDArray[np.float32] = rescale_phase(
                np.stack([p.dataobj[..., frame_idx] for p in phase], axis=-1),
                min=min_phase,
                max=max_phase,
            ).astype(np.float32)
            mag_data: npt.NDArray[np.float32] = np.stack([m.dataobj[..., frame_idx] for m in mag], axis=-1).astype(
                np.float32
            )
            mask_data = cast(npt.NDArray[np.bool_], mask.dataobj[..., frame_idx].astype(bool))
            TEs = TEs.astype(np.float32)
            yield (phase_data, mag_data, TEs, mask_data, automask, automask_dilation, idx, wrap_limit)

    def save_unwrapped_and_mask(idx, result):
        # get the unwrapped image
        logging.info(f"Collecting frame: {idx}")
        # store in the captured unwrapped and new_mask_data arrays
        unwrapped[..., idx], new_masks[..., idx] = result

    # unwrap the phase of each frame
    run_executor(
        ncpus=n_cpus,
        type="process",
        fn=unwrap_phase,
        iterator=phase_iterator(phase, mag, TEs, mask, frames, automask, border_size),
        post_fn=save_unwrapped_and_mask,
    )

    def temporal_consistency_iterator(unwrapped, TEs, mag, frames, masks):
        # Make a copy of the first echo data
        unwrapped_echo_1 = unwrapped[..., 0, :].copy()

        logging.info("Computing temporal consistency...")
        for t, frame_idx in enumerate(frames):
            yield (unwrapped, unwrapped_echo_1, TEs, mag, t, frame_idx, cast(npt.NDArray, masks))

    def post_temporal_consistency_check(idx, result):
        logging.info(f"Temporal consistency check for frame {idx} complete.")

    if not debug:
        run_executor(
            ncpus=n_cpus,
            type="thread",
            fn=check_temporal_consistency_corr,
            iterator=temporal_consistency_iterator(unwrapped, TEs, mag, frames, new_masks),
            post_fn=post_temporal_consistency_check,
        )

    # Save out unwrapped phase for debugging
    if debug:
        logging.info("Saving unwrapped phase images...")
        for i in range(unwrapped.shape[-2]):
            nib.Nifti1Image(unwrapped[:, :, :, i, :], phase[0].affine, phase[0].header).to_filename(f"phase{i}.nii")
        logging.info("Saving masks..")
        nib.Nifti1Image(new_masks, phase[0].affine, phase[0].header).to_filename("masks.nii")

    # compute field maps on temporally consistent unwrapped phase
    def field_map_iterator(field_maps, unwrapped, mag, TEs):
        logging.info(f"Running field map computation...")
        # convert TEs to a matrix
        TEs_mat = TEs[:, np.newaxis]
        for frame_num in range(unwrapped.shape[-1]):
            yield (unwrapped[..., frame_num], mag, TEs.shape[0], TEs_mat, frame_num)

    def post_field_map(idx, result):
        logging.info(f"Field map computation for frame {idx} complete.")
        field_maps[..., idx] = result

    run_executor(
        ncpus=n_cpus,
        type="thread",
        fn=compute_field_map,
        iterator=field_map_iterator(field_maps, unwrapped, mag, TEs),
        post_fn=post_field_map,
    )

    # if border voxels defined, use that information to stabilize border regions using SVD filtering
    # this will probably kill any respiration signals in these voxels but improve the
    # temporal stability of the field maps in these regions (and could we really resolve
    # respiration in those voxels any way? probably not...)
    svd_filtering(
        field_maps,
        new_masks,
        phase[0].header.get_zooms()[0],  # type: ignore
        n_frames,
        border_filt,
        svd_filt,
    )

    # return the field map as a nifti image
    return nib.Nifti1Image(field_maps[..., frames], phase[0].affine, phase[0].header)
