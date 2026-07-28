#set document(
  title: "Global 2π branch selection in MCPC-3D-S",
  author: "warpkit",
)
#set page(
  paper: "us-letter",
  margin: (x: 1.1in, y: 1in),
  numbering: "1",
)
#set text(font: ("New Computer Modern", "Times New Roman"), size: 10.5pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#show heading.where(level: 1): it => block(above: 1.4em, below: 0.8em)[
  #set text(size: 13pt, weight: "bold")
  #it
]
#show heading.where(level: 2): it => block(above: 1.1em, below: 0.6em)[
  #set text(size: 11pt, weight: "bold")
  #it
]
#show raw.where(block: false): it => box(
  fill: luma(240), inset: (x: 3pt, y: 0pt), outset: (y: 3pt), radius: 2pt, it,
)

#let dte = $Delta "TE"$
#let phiuw = $Delta phi_"uw"$

#let key(body) = block(
  width: 100%, fill: luma(245), stroke: (left: 2pt + luma(140)),
  inset: 9pt, radius: 2pt, body,
)
#let defn(body) = block(
  width: 100%, stroke: 0.5pt + luma(160), inset: 9pt, radius: 2pt, body,
)
#let proof(body) = block(inset: (left: 12pt), text(size: 10pt, body))

#align(center)[
  #text(size: 17pt, weight: "bold")[
    Global $2 pi$ branch selection in MCPC-3D-S
  ]
  #v(0.3em)
  #text(size: 11pt)[Theory and derivation]
  #v(0.2em)
  #text(size: 9.5pt, style: "italic")[warpkit — phase offset estimation for MEDIC]
]

#v(1em)

#outline(indent: auto, depth: 2)

#v(1em)

= The problem

Multi-echo distortion correction needs the receive-coil phase offset $theta$ —
the phase each voxel carries at $"TE" = 0$, independent of the field. MCPC-3D-S
recovers it from the first two echoes. The recovered offset is pinned only up to
a whole number of turns of the unwrapped phase difference, and choosing the
wrong turn corrupts $theta$ by a fixed constant. This note derives what that
constant is, when it is detectable, and what decides the choice when it is not.

Nothing here is specific to a dataset; every quantity is a function of the echo
times or of the signal model.

= Notation and signal model

Let $W(x) = arg(e^(i x))$ denote wrapping to $(-pi, pi]$. Echo times are
evenly spaced,
$
  t_e = "TE"_0 + e dot #dte, quad e = 0, 1, ..., E-1,
$
and we write the dimensionless echo-time ratio
$
  k = "TE"_0 / (#dte).
$
A voxel's measured phase at echo $e$ is
$
  phi_e = W(theta + 2 pi f t_e),
$ <eq:model>
with $theta$ the coil offset and $f$ the local field in Hz. Two derived scales
recur throughout:

#defn[
- *Wrap spacing* $1 slash #dte$ — the field difference between adjacent
  branches, in Hz.
- *Half-wrap* $1 slash (2 #dte)$ — the decision boundary of any
  smallest-$abs(f)$ prior.
]

== MCPC-3D-S

The estimator forms the phase difference of the first two echoes, spatially
unwraps it to #phiuw, and back-extrapolates to $"TE" = 0$:
$
  hat(theta) = W(phi_0 - k dot #phiuw).
$ <eq:est>
Spatial unwrapping fixes #phiuw only up to a global additive $2 pi M$, $M in ZZ$
— it recovers relative phase, not absolute. Selecting $M$ is the subject of this
note.

= Two ambiguities

Two distinct things can be meant by "the branch is wrong". They behave
differently and conflating them is the main source of confusion.

== Ambiguity A: the true alias

#defn[
*Proposition 1.* For evenly spaced echoes and any $N in ZZ$, define
$
  theta' = theta - W(2 pi k N), quad quad f' = f + N / (#dte).
$
Then $(theta', f')$ reproduces @eq:model *identically*: $phi'_e = phi_e$ for
every echo.
] <prop:alias>

#proof[
*Proof.* Using $W(2 pi k N) equiv 2 pi k N mod 2 pi$,
$
  theta' + 2 pi f' t_e
    &equiv theta - 2 pi k N + 2 pi (f + N slash #dte) t_e \
    &= theta + 2 pi f t_e + 2 pi N (t_e slash #dte - k).
$
For evenly spaced echoes $t_e slash #dte = k + e$, so the bracket is $e in ZZ$
and vanishes under $W$. #h(1fr) $qed$
]

Here $theta$ and $f$ move *together*. The wrapped data is bit-identical, so no
statistic computed from the phase can distinguish the alternatives. Only a prior
can.

#key[
*Extra echoes do not help.* #link(<prop:alias>)[Prop. 1] assumes even spacing, and every echo
pair then aliases at the same period $1 slash #dte$: a five-echo acquisition
carries exactly as much branch information as a two-echo one. Unequal spacing
would break it — two pairs aliasing at incommensurate periods are simultaneously
satisfied only by the true $f$ — but even spacing is what an EPI echo train
naturally produces.
]

== Ambiguity B: offset–branch mismatch

#defn[
*Proposition 2.* If #phiuw carries a branch error of $M$ wraps, @eq:est returns
$hat(theta)_M = W(theta - 2 pi k M)$, and the offset-removed phases satisfy
$
  psi_e = phi_e - hat(theta)_M equiv 2 pi f t_e + c mod 2 pi,
  quad quad c = W(2 pi k M),
$
with $c$ *identical on every echo*.
] <prop:offset>

#proof[
*Proof.* Substituting $#phiuw -> 2 pi f #dte + 2 pi M$ into @eq:est gives
$hat(theta)_M = W(theta - 2 pi k M)$. Then
$psi_e equiv (theta + 2 pi f t_e) - (theta - 2 pi k M) = 2 pi f t_e + 2 pi k M$,
and $2 pi k M equiv c mod 2 pi$. #h(1fr) $qed$
]

Here $hat(theta)$ moves *alone*: the field has already been fixed upstream, so
offset and field are now mutually inconsistent, and that inconsistency is a
measurable constant.

#key[
*The distinction.* Ambiguity A changes $theta$ and $f$ together and is
invisible. Ambiguity B changes $theta$ while $f$ stays put, and shows up as a
constant lift off the origin. Only B is detectable.
]

= Detecting ambiguity B

By #link(<prop:offset>)[Prop. 2] the signature of a branch error is a constant added to every
echo. A line through the offset-removed echoes therefore fails to pass through
the origin, and its intercept *is* that constant:
$
  psi_e approx s dot t_e + c, quad quad s = 2 pi f, quad c = W(2 pi k M).
$
So the score for candidate $M$ is $abs(c)$, estimated over a mask of reliable
voxels. A correct branch extrapolates through zero; a wrong one does not.

== The rejection scale is analytic

Candidate intercepts sit on a lattice spaced by
$
  Delta c = abs(W(2 pi k)),
$ <eq:step>
which depends only on the echo times — no data, no fitting, no tuned threshold.
Classifying against $Delta c slash 2$ is nearest-neighbour assignment on that
lattice, not a chosen cutoff.

== Degenerate cases

#defn[
*Integer $k$.* If $k in ZZ$ then $c = W(2 pi k M) = 0$ for all $M$, so the
intercept carries no signal. But by #link(<prop:offset>)[Prop. 2] $hat(theta)_M = W(theta)$ is
*also unchanged*, so no branch can affect the output. The degeneracy is benign
and @eq:step returns zero, which is the correct signal to do nothing.
]

More generally the surviving intercept obeys a two-regime law in $k$, peaking at
$k = 2 slash 3$ and falling to zero at integer $k$. Below that peak the
achievable separation shrinks, and for small $k$ several branches can be exactly
through-origin at once. Since an EPI echo train has $"TE"_0$ shorter than the
spacing between echoes, $k < 1$ always, and low-$k$ protocols are precisely
those where the intercept test loses power.

= The decision rule

#defn[
1. *Consistency.* Score each candidate by its intercept. A branch carrying a
   leftover intercept does not explain the data and is discarded. If exactly one
   candidate survives, it is the answer regardless of its field.
2. *Prior, only on a tie.* If several survive, the alias of #link(<prop:alias>)[Prop. 1] is
   genuinely reachable and the phase cannot rank them. Take the candidate whose
   global field level is closest to zero.
3. If nothing survives, do nothing — a failed fit is not evidence for any
   branch.
]

== Both stages are necessary

Neither stage suffices alone, for reasons that are structural rather than
empirical.

#defn[
*Stage 1 alone is insufficient.* When two branches are both exactly
through-origin their intercepts are equal to within numerical noise. Ranking
them is ranking noise; deferring leaves a known-bad offset in place.
]

#defn[
*Stage 2 alone is insufficient.* The smallest-$abs(f)$ prior is *symmetric about
zero*. Two branches whose fields are $+delta$ and $-delta$ are exactly
equidistant, so the prior cannot separate them — and after re-wrapping inside
the unwrapper, candidates $M$ and $-M$ can land at equal $abs(f)$. Only the
phase, through the intercept, breaks that symmetry.
]

The two stages are therefore complementary in a precise sense: the intercept is
blind to a *sign-preserving* shift of the field, and the prior is blind to a
*sign-reversing* one.

= The prior and its estimator

== What the prior actually asserts

The scanner's frequency adjustment sets the demodulation frequency to the centre
of the water resonance over the shim volume. The physical statement is therefore

#key[
The *bulk-tissue* field level is near zero — not the mean field, and not the
median field over whatever mask happens to be in use.
]

This matters because the field distribution over a brain is not symmetric. It is
a sharp peak at the bulk-tissue value plus long tails contributed by air–tissue
interfaces, sinuses, dropout and mask edges, where susceptibility gradients are
large. Any statistic that is a *quantile of the whole distribution* — the median
included — is displaced toward whichever tail is heavier.

== Why a median is the wrong estimator

Write the field distribution as a mixture
$
  p(f) = (1 - epsilon) dot p_"bulk" (f) + epsilon dot p_"tail" (f),
$
with $p_"bulk"$ narrow and centred on the quantity the shim actually set, and
$p_"tail"$ broad and asymmetric. The mode of $p$ is a consistent estimator of
the centre of $p_"bulk"$ for small $epsilon$; the median of $p$ is not. Its bias
grows with both the tail fraction $epsilon$ and the tail asymmetry.

Two consequences follow, and they are the same defect seen twice:

#defn[
- *Mask dilation is harmful.* Dilating a brain mask admits precisely the voxels
  that populate $p_"tail"$, raising $epsilon$. A median over a dilated mask is
  therefore more biased than the same median over an eroded one.
- *A bias of a fraction of a wrap is decisive.* The prior compares
  $abs(hat(f))$ against $1 slash (2 #dte)$. Any estimator bias adds directly to
  the distance from that boundary, so a subject whose true bulk field lies close
  to the half-wrap can be pushed across it by estimator bias alone — flipping
  the branch on the basis of an artefact of the statistic, not of the data.
]

Weighting by magnitude squared suppresses the tails partially, because
low-signal voxels are over-represented there. It is a mitigation, not a fix: the
estimand is still a quantile of the mixture.

== The half-sample mode

The mode is the correct estimand but the usual estimators need a bin width or a
kernel bandwidth, which would reintroduce exactly the kind of tuned constant
this design avoids. The *half-sample mode* has neither.

#defn[
Sort the samples. Repeatedly find the shortest contiguous interval containing
half the remaining points and discard everything outside it. Stop when three or
fewer remain and return their mean.
]

The justification is direct: samples are densest, per unit of value, where the
density is highest, so the shortest half-interval brackets the peak. Recursing
zooms in on it. Formally, if $x_((1)) <= ... <= x_((n))$ are the order
statistics and $h = ceil(n slash 2)$, the retained window is
$
  [x_((i^*)), space x_((i^* + h - 1))],
  quad quad i^* = arg min_i (x_((i + h - 1)) - x_((i))),
$
and all $n - h + 1$ candidate widths are obtained from a single vectorised
difference of two offset slices of the sorted array.

#defn[
*Properties.*
- *Parameter-free.* No bin width, no bandwidth, no threshold.
- *Cost $O(n log n)$*, dominated by the initial sort; each subsequent pass is
  linear in a geometrically shrinking array, so the total after sorting is
  $O(n)$. In particular it is cheaper than a weighted median, which must sort a
  weight array as well.
- *Convergence.* Near a smooth unimodal peak, halving the sample halves the
  window width, so the interval contracts geometrically and the estimate is
  insensitive to how many passes are taken.
- *Weight-independent.* The peak location does not depend on how samples are
  weighted, so the estimator inherits none of the weighting choices a median
  requires.
]

#defn[
*Assumptions.* Unimodality — with a genuinely bimodal field the method converges
on the denser mode, which for brain data is bulk tissue but is not automatic.
Ties in the arg-min are resolved arbitrarily; on continuous data they have
measure zero, on quantised or very small samples they do not. The estimator
needs enough samples for the density to be resolved.
]

= Structural limits

Three limits are properties of the acquisition, not of the algorithm.

#defn[
*Ambiguity A is not addressable.* By #link(<prop:alias>)[Prop. 1] the data is invariant under the
alias, so no rule can recover the true branch when the global field genuinely
exceeds $1 slash (2 #dte)$. The prior folds it back, and this is unavoidable.
Whether it is *reachable* is set by #dte, which is bounded below by the EPI
readout duration — echoes cannot be spaced closer than they can be read out —
and above by $T_2^*$, since several echoes must fit before the signal decays.
Those bounds confine #dte, and hence pin the half-wrap threshold well above
typical shim residuals.
]

#defn[
*Ultra-high field does not simply make this worse.* Shim residuals grow with
$B_0$, but $T_2^*$ shortens, which forces #dte down and therefore raises the
$1 slash (2 #dte)$ threshold. The two effects act in opposite directions on the
same quantity.
]

#defn[
*A shift of the multi-echo template is invisible here.* A multi-echo unwrapper
that global-corrects only its template echo, and derives the remaining echoes by
temporal unwrapping relative to it, propagates any whole-turn shift of the
template *proportionally to* $"TE"$. A proportional shift is a pure change of
slope: it moves the fitted field by $1 slash "TE"_0$ Hz and leaves the intercept
exactly zero. It is therefore a third failure mode, distinct from both
ambiguities above, and structurally undetectable by an intercept test.
]

= Summary

#key[
The branch error puts a constant $W(2 pi k M)$ on every echo, so it is visible
as a nonzero intercept when the echoes are extrapolated to $"TE" = 0$, and the
lattice spacing $abs(W(2 pi k))$ supplies the rejection scale analytically. When
several branches are through-origin the alias is exact and only a prior remains.
That prior asserts that the *bulk-tissue* field is near zero, which makes the
mode of the field distribution — not its median — the right estimand, and the
half-sample mode estimates it without introducing a tuned constant. The
intercept and the prior are blind to opposite symmetries, so both are required.
]
