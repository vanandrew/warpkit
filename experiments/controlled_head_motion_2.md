# Previous Notes

See here: https://gist.github.com/vanandrew/99eb7f9f5069bb257e7bc1d2c4f49418

> :warning: :warning: :warning: **For each video, right click and select "Loop" before
> playing.** :warning: :warning: :warning:

# Distortion differences between Spin-Echo EPI field maps and ME BOLD data

One observation I saw in the previous field maps was parts of the SE-EPI
field map being distorted differently from the ME BOLD data.

<video controls loop>
  <source src="test_outputs/alignment_check_ref.mp4" type="video/mp4">
</video>

https://user-images.githubusercontent.com/3641187/208333859-1a0c27d4-a9c3-4087-ac96-5cb7434ca15f.mp4

The current theory is that these are due to eddy current effects. Will be 
collecting more data to test this.

# Corrections using MEDIC

One particular question that I raised in the previous notes was whether you
could apply the framewise field maps in a framewise manner? or if averaging
across some frames was necessary? Results below answer this question:

## Framewise corrections

Naive framewise corrections (simply correcting distortion using the field 
map computed at each frame) seems to introduce extra variance to the data
(particularly near areas adjacent to signal dropout/low SNR).

Speed Up (10x)

<video controls loop>
  <source src="test_outputs/framewise_correction.mp4" type="video/mp4">
</video>

## Averaging over field maps

### MEDIC correction with averaged field maps

Corrections on 1st frame using averaged field maps (100 frames averaged). Compared against uncorrected, TOPUP
correction, T2w anatomicals (rigid body aligned to reference functional).

#### Neutral Position

Uncorrected, MEDIC, TOPUP

<video controls loop>
  <source src="test_outputs/correction_compare_run-00_three.mp4" type="video/mp4">
</video>

MEDIC vs. T2w

<video controls loop>
  <source src="test_outputs/correction_compare_run-00_medic.mp4" type="video/mp4">
</video>

TOPUP vs. T2w

<video controls loop>
  <source src="test_outputs/correction_compare_run-00_topup.mp4" type="video/mp4">
</video>

#### +z rotation

Uncorrected, MEDIC, TOPUP

<video controls loop>
  <source src="test_outputs/correction_compare_run-01_three.mp4" type="video/mp4">
</video>

MEDIC vs. T2w

<video controls loop>
  <source src="test_outputs/correction_compare_run-01_medic.mp4" type="video/mp4">
</video>

TOPUP vs. T2w

<video controls loop>
  <source src="test_outputs/correction_compare_run-01_topup.mp4" type="video/mp4">
</video>

#### -z rotation

Uncorrected, MEDIC, TOPUP

<video controls loop>
  <source src="test_outputs/correction_compare_run-02_three.mp4" type="video/mp4">
</video>

MEDIC vs. T2w

<video controls loop>
  <source src="test_outputs/correction_compare_run-02_medic.mp4" type="video/mp4">
</video>

TOPUP vs. T2w

<video controls loop>
  <source src="test_outputs/correction_compare_run-02_topup.mp4" type="video/mp4">
</video>

#### +x rotation

Uncorrected, MEDIC, TOPUP

<video controls loop>
  <source src="test_outputs/correction_compare_run-03_three.mp4" type="video/mp4">
</video>

MEDIC vs. T2w

<video controls loop>
  <source src="test_outputs/correction_compare_run-03_medic.mp4" type="video/mp4">
</video>

TOPUP vs. T2w

<video controls loop>
  <source src="test_outputs/correction_compare_run-03_topup.mp4" type="video/mp4">
</video>

#### -x rotation

Uncorrected, MEDIC, TOPUP

<video controls loop>
  <source src="test_outputs/correction_compare_run-04_three.mp4" type="video/mp4">
</video>

MEDIC vs. T2w

<video controls loop>
  <source src="test_outputs/correction_compare_run-04_medic.mp4" type="video/mp4">
</video>

TOPUP vs. T2w

<video controls loop>
  <source src="test_outputs/correction_compare_run-04_topup.mp4" type="video/mp4">
</video>

#### +y rotation

Uncorrected, MEDIC, TOPUP

<video controls loop>
  <source src="test_outputs/correction_compare_run-05_three.mp4" type="video/mp4">
</video>

MEDIC vs. T2w

<video controls loop>
  <source src="test_outputs/correction_compare_run-05_medic.mp4" type="video/mp4">
</video>

TOPUP vs. T2w

<video controls loop>
  <source src="test_outputs/correction_compare_run-05_topup.mp4" type="video/mp4">
</video>

#### -y rotation

Uncorrected, MEDIC, TOPUP

<video controls loop>
  <source src="test_outputs/correction_compare_run-06_three.mp4" type="video/mp4">
</video>

MEDIC vs. T2w

<video controls loop>
  <source src="test_outputs/correction_compare_run-06_medic.mp4" type="video/mp4">
</video>

TOPUP vs. T2w

<video controls loop>
  <source src="test_outputs/correction_compare_run-06_topup.mp4" type="video/mp4">
</video>

#### -z translation

Uncorrected, MEDIC, TOPUP

<video controls loop>
  <source src="test_outputs/correction_compare_run-07_three.mp4" type="video/mp4">
</video>

MEDIC vs. T2w

<video controls loop>
  <source src="test_outputs/correction_compare_run-07_medic.mp4" type="video/mp4">
</video>

TOPUP vs. T2w

<video controls loop>
  <source src="test_outputs/correction_compare_run-07_topup.mp4" type="video/mp4">
</video>

TOPUP seems to correct areas of signal dropout better than MEDIC (e.g. orbitofrontal and inferotemporal). MEDIC seems to do better in cortex (possibly due to eddy current artifact in Spin-Echo field maps?).

Averaging over the field map frames seems to be necessary to get a good estimate of the field when using MEDIC. But this raises two new questions:

1. How close in head position do the field maps need to be in order for effective averaging? Another way of thinking about this question is in a weighted averaging scheme, how should we weight each field map's motion parameters when conducting a weighted average of the field maps?

2. What is the minimum number of field maps needed for an effective average field map?

### Checking the impulse response

To get at the first question, we can look at the impulse response by
computing a regression model of the field map at each frame against the 
six motion parameters.

TODO (My local machine ran out-of-memory when trying to run this... Will update later once I run this on our servers...)

### Examining number of field maps averaged

To answer question 2, I wanted to see how averaging over a subset of field map frames correlated against averaging over all field map frames for a single scan.

The figure below varys the sample size of the field map for a 440 frame scan with minimal motion (each sample was randomly selected from the 440 frames). Each sample was correlated against the full 440 frame averaged field map.

![](sample_size.png)

Not a huge fan of this figure... Is there a better way to do this analysis?

# Other interesting observations

Was showing this to Dylan the other day and he thought it would be good for me to share.

Below are some other interesting observations that I made while looking at this data (not related to MEDIC):

## Large head motion effect

Looking at the magnitude data, after a large head rotation (10x speed up):

<video controls loop>
  <source src="test_outputs/mag_run-07.mp4" type="video/mp4">
</video>

<video controls loop>
  <source src="test_outputs/mag_run-08.mp4" type="video/mp4">
</video>

<video controls loop>
  <source src="test_outputs/mag_run-09.mp4" type="video/mp4">
</video>

<video controls loop>
  <source src="test_outputs/mag_run-10.mp4" type="video/mp4">
</video>

<video controls loop>
  <source src="test_outputs/mag_run-11.mp4" type="video/mp4">
</video>

<video controls loop>
  <source src="test_outputs/mag_run-12.mp4" type="video/mp4">
</video>

<video controls loop>
  <source src="test_outputs/mag_run-13.mp4" type="video/mp4">
</video>

- Note the signal corruption (that looks like pathology) after a large head motion event (
  particularly during large x and y rotations).
