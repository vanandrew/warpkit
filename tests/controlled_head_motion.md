# Background

Some results of the MEDIC algorithm being applied to a controlled head motion dataset.

## Experimental Setup

Participant was asked to remain still in various head poitions as followed:

- Head in neutral position
- Head rotation in +z
- Head rotation in -z
- Head rotation in +x
- Head rotation in -x
- Head rotation in +y
- Head rotation in -y
- Translation in -Z

The scanner shim settings were fixed during the neutral position run to eliminate the influence of shimming on the
field. Subsequently, all rotations + translation runs were performed with the same shim settings as the neutral
position run.

In a second set of runs, the participant was asked to remain in the neutral position, then perform a movement mid
scan in the specified directions:

- Head rotation from neutral to +z
- Head rotation from neutral to -z
- Head rotation from neutral to +x
- Head rotation from neutral to -x
- Head rotation from neutral to +y
- Head rotation from neutral to -y
- Head translation from neutral to -z

For each run, a SE-EPI field map (3 AP/PA frames, TR: 8 s, TE: 66 ms, 72 Slices, FOV: 110x110,
Voxel Size: 2.0mm) and a ME-EPI BOLD run (TR: 1.761 s, TEs: 14.2, 38.93, 63.66, 88.39, 113.12 ms, 72 Slices,
FOV: 110x110, Voxel Size: 2.0 mm, Multi-Band: 6) were acquired.

All data was aligned to a single reference frame (neutral position) for comparison.

# Results

## Data quality/alignment checks

These figures were generated to check acquisition and alignment quality:

### ME-EPI BOLD data:

<video controls loop>
  <source src="test_outputs/alignment_check_bold.mp4" type="video/mp4">
</video>

https://user-images.githubusercontent.com/3641187/208332969-2dfdc78d-cf22-4ebf-86e2-2cea852d33e6.mp4

### SE-EPI field map data:

<video controls loop>
  <source src="test_outputs/alignment_check_topup.mp4" type="video/mp4">
</video>

https://user-images.githubusercontent.com/3641187/208333190-2194ca71-0bf2-42c9-86d6-39391af2826e.mp4

### Alignment between SE-EPI field map and ME-EPI BOLD data (1st run: neutral position):

<video controls loop>
  <source src="test_outputs/alignment_check_ref.mp4" type="video/mp4">
</video>

- There seems to be a slight distortion difference (red circle) between the SE-EPI image and the ME-EPI bold data) that
cannot be explained by motion (there was insignificant motion between SE-EPI and ME-EPI runs):

## MEDIC results

### MEDIC field maps

