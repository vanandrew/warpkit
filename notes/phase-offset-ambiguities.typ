#set document(
  title: "Phase offset ambiguities in multi-echo field mapping",
  author: "Andrew Van",
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

#let defn(body) = block(
  width: 100%, stroke: 0.5pt + luma(160), inset: 9pt, radius: 2pt, body,
)
#let proof(body) = block(inset: (left: 12pt), text(size: 10pt, body))

#align(center)[
  #text(size: 17pt, weight: "bold")[
    Phase offset ambiguities in multi-echo field mapping
  ]
  #v(0.9em)
  #text(size: 10.5pt)[Andrew Van]
  #v(0.15em)
  #text(size: 9.5pt)[July 28, 2026]
]

#v(1em)

#outline(indent: auto, depth: 2)

#v(1.2em)

= Introduction

Multi-echo field mapping estimates the field from the rate at which phase
accumulates across echoes. Every voxel also carries a receive-coil phase offset,
present at zero echo time, common to all echoes, and set by coil geometry rather
than by the field. This offset varies rapidly in space, steeply near coil
elements and discontinuously between receive channels, so measured phase is not
spatially smooth even where the field is. Spatial unwrapping assumes that
neighbouring voxels differ by less than half a turn, and raw phase violates that
assumption. Removing the offset restores the smoothness later processing needs.

The offset is never measured. It must be inferred from the same wrapped phase
that it corrupts, and the inference leaves one integer undetermined. This note
sets out the consequences of that integer, the rule conventionally used to fix
it, the conditions under which the rule fails, and the procedure MEDIC uses to
identify and repair those cases.

= Phase offset estimation

#defn[
*Notation.* A quantity is *wrapped* when confined to $(-pi, pi]$.

#table(
  columns: (auto, auto, 1fr),
  stroke: none,
  inset: (x: 4pt, y: 2.5pt),
  [$theta$, $f$],   [],           [coil offset (rad) and field offset (Hz)],
  [$phi_e$],        [wrapped],    [the recorded phase at echo $e$],
  [$Delta phi$],    [wrapped],    [the wrapped phase difference $W(phi_1 - phi_0)$],
  [#phiuw],         [unwrapped],  [that difference with its turn count restored],
  [$M$],            [integer],    [the unwrapper's error in whole turns],
  [$hat(theta)$],   [wrapped],    [the offset estimate],
  [$psi_e$],        [neither],    [recorded phase less an estimated offset],
)
]

== Signal model

Acquire $E$ echoes at times $t_0 < t_1 < ... < t_(E-1)$. Field evolution is
linear in time, so the phase at echo $e$ before wrapping is
$theta + 2 pi f t_e$. What is recorded is that value wrapped into one turn. With
$W(x) = arg(e^(i x))$,

$
  phi_e = W(theta + 2 pi f t_e).
$ <eq:model>

Without $W$, two echoes would determine $theta$ and $f$ by a straight-line fit.
With it the line is known only modulo $2 pi$, and infinitely many $(theta, f)$
pairs remain compatible with the data.

Two properties of $W$ are used below. It is invariant under whole turns,
$W(x + 2 pi m) = W(x)$ for integer $m$, so every real $x$ decomposes uniquely as
$x = W(x) + 2 pi m$. Wrapping discards $m$ and unwrapping recovers it. Second, a
difference of wrapped values returns the wrapped difference only after a second
wrap,

$
  W(W(x) - W(y)) = W(x - y),
$ <eq:diffid>

because $W(x) - W(y)$ spans $(-2 pi, 2 pi)$ and coincides with $W(x - y)$ only
when it happens to lie within one turn.

== Echo spacing

An echo-planar train produces echoes at constant spacing,

$
  t_e = "TE"_0 + e dot #dte, quad e = 0, 1, ..., E-1,
$ <eq:spacing>

with $"TE"_0$ the first echo time and #dte the spacing. Two field scales follow
from #dte alone. The *wrap spacing* $1 slash #dte$ separates neighbouring
candidate solutions for the field. The *half-wrap* $1 slash (2 #dte)$ lies
midway between them, and serves as the decision boundary for any rule that
prefers the smallest field.

== Recovering the offset

The construction below is MCPC-3D-S. Because $theta$ is common to every echo, it
cancels in a difference. For the first two echoes,

$
  Delta phi = W(phi_1 - phi_0)
  = W((theta + 2 pi f t_1) - (theta + 2 pi f t_0))
  = W(2 pi f #dte).
$ <eq:diff>

The middle step follows from @eq:diffid, since each $phi_e$ differs from
$theta + 2 pi f t_e$ by whole turns. The wrap remains outside the subtraction
because $phi_1 - phi_0$ is a difference of wrapped values.

$Delta phi$ is itself wrapped, but a difference of neighbouring echoes varies
slowly in space, so a region-growing algorithm can restore its missing turns.
Spatial unwrapping recovers phase differences between voxels rather than
absolute turn counts, so what it returns differs from the true unwrapped
difference by an unknown whole number of them:

$
  #phiuw = 2 pi f #dte + 2 pi M,
$ <eq:uw>

with $M = 0$ when the unwrapper is correct. Extrapolating to $t = 0$ then
recovers the offset. Writing $k = "TE"_0 slash #dte$,

$
  hat(theta) = W(phi_0 - k dot #phiuw).
$ <eq:est>

The outer $W$ is required because #phiuw is unbounded and $k dot #phiuw$ with
it. $M$ is the only quantity in @eq:est that the data leaves open.

= The half-wrap limit

Substituting @eq:uw into @eq:est, and reading the field from the same
difference, gives the solution implied by any $M$:

$
  hat(theta)_M = W(theta - 2 pi k M),
  quad quad
  hat(f)_M = f + M / (#dte).
$ <eq:pair>

The two errors are locked together. The field moves by $M$ whole wrap spacings
and the offset by $-2 pi k M$, and neither occurs without the other. The
candidate solutions form a discrete family, one per integer, separated by
$1 slash #dte$ in field.

#defn[
*Proposition 1.* For evenly spaced echoes and any integer $M$, the pair
$(hat(theta)_M, hat(f)_M)$ reproduces @eq:model identically:
$W(hat(theta)_M + 2 pi hat(f)_M t_e) = phi_e$ at every echo.
] <prop:alias>

#proof[
*Proof.* Working modulo $2 pi$, so that $W$ of either side agrees,
$
  hat(theta)_M + 2 pi hat(f)_M t_e
    &equiv theta - 2 pi k M + 2 pi (f + M slash #dte) t_e \
    &= theta + 2 pi f t_e + 2 pi M (t_e slash #dte - k).
$
By @eq:spacing, $t_e slash #dte = k + e$, so the parenthesis is the integer $e$,
the final term is a whole number of turns, and both sides wrap alike.
#h(1fr) $qed$
]

The cancellation depends on the even spacing. Written as
$t_e slash #dte = k + e$, the ratio $k$ locates the echo train on the #dte
lattice, and the $-2 pi k M$ carried by the offset is what cancels it. Since the
cancellation holds at every echo simultaneously, a longer echo train places no
further constraint on $M$. Unequal spacing would leave $t_e slash #dte - k$
non-integral at some echo and so produce a detectable residue, but an EPI train
is evenly spaced by construction.

== The rule ROMEO applies

No measurement selects $M$, so the choice must come from elsewhere. ROMEO's
global correction, which MEDIC calls, subtracts a single multiple of $2 pi$
from the unwrapped phase so that the median rounded turn count over the mask
becomes zero. The effect is to select the candidate whose field is smallest,

$
  hat(M) = op("argmin")_M abs(f + M slash #dte),
$ <eq:argmin>

which is equivalent to requiring $abs(hat(f)) < 1 slash (2 #dte)$.

The rule encodes a physical expectation rather than a measurement. Before
imaging, the scanner sets its demodulation frequency to the centre of the water
resonance, so the residual field is a small offset rather than an arbitrary one.
Where that expectation holds, the rule returns the correct solution. Where it
does not, the rule returns an alias, and by
#link(<prop:alias>)[Prop. 1] nothing in the data distinguishes the two.

= Why the rule fails on some frames

A wrong value of $M$, chosen once and applied to a whole acquisition, would
matter very little. Every frame would be displaced by the same $1 slash #dte$.
The field offset is already referenced to an arbitrary demodulation frequency,
so a fixed additive constant changes neither temporal contrast nor spatial
structure, and by #link(<prop:alias>)[Prop. 1] it could not be detected in any
case.

The difficulty is that $M$ is not chosen once. Three properties of the procedure
combine to produce a failure.

The first is that the choice is repeated on every frame. A functional
acquisition is processed frame by frame, and each frame resolves $M$ from its
own data, without reference to the frames around it.

The second is that the quantity compared against the half-wrap is an estimate. A
median rounded turn count is a location statistic over a mask of voxels, each
carrying its own field, and it absorbs whatever varies between frames, including
physiological change, subject motion, and noise in the voxels the mask admits.

The third is that the comparison has a hard boundary. Far from the half-wrap the
margin is wide, and no plausible perturbation of the estimate alters the
outcome. Close to the half-wrap the margin approaches zero.

For a subject whose field lies near the half-wrap, $M$ is therefore decided on
each frame independently, by a noisy statistic sitting close to a threshold, and
different frames fall on different sides of it. Because $M$ is an integer, a
frame that falls on the wrong side is displaced by a full wrap rather than by a
small amount. The resulting time series is stable to a fraction of a hertz
across most frames and displaced by $1 slash #dte$ on a scattered subset, with
no intermediate values.

The damage is temporal. A uniformly wrong field is harmless, but a field that
takes two values within a single acquisition corrupts the temporal structure the
acquisition was performed to measure.

= Detecting a failed choice

#link(<prop:alias>)[Prop. 1] appears to rule out any correction. It applies,
however, to a *matched* pair, meaning an offset $hat(theta)_M$ together with the
field $hat(f)_M$ that accompanies it. The pipeline does not produce the pair in
a single step. The offset is formed first, from #phiuw by @eq:est. The field is
derived afterwards, by removing that offset and unwrapping in time, and the
second unwrapping applies a global correction of its own without reference to
the $M$ the offset assumed.

When the two stages disagree, the result falls outside the family in @eq:pair.
The offset belongs to one candidate and the field to another, so
#link(<prop:alias>)[Prop. 1] does not apply to it. This is the situation on a
frame where the rule has tipped, because the two stages use different masks and
different statistics and can land on opposite sides of the boundary.

Removing the offset from the recorded phase leaves

$
  psi_e = phi_e - hat(theta)_M,
$ <eq:psi>

a difference of two wrapped values, and therefore not itself a phase.
Unwrapping it in time and fitting

$
  psi_e approx c + s dot t_e
$ <eq:fit>

gives a slope $s$, which is the field, and an intercept $c$, the value the line
extrapolates to at $t = 0$. A self-consistent reconstruction has had its offset
removed, so its line passes through the origin and $c = 0$.

#defn[
*Proposition 2.* If the fit @eq:fit is taken on the solution whose slope is the
true field $f$, then
$
  psi_e equiv 2 pi f t_e + c mod 2 pi,
  quad quad c = W(2 pi k M),
$
with $c$ identical at every echo.
] <prop:offset>

#proof[
*Proof.* $psi_e equiv (theta + 2 pi f t_e) - (theta - 2 pi k M)
= 2 pi f t_e + 2 pi k M$, and $2 pi k M equiv c mod 2 pi$. The constant does not
depend on $e$. #h(1fr) $qed$
]

Two properties make $c$ usable. Being identical at every echo, it displaces the
whole fitted line rather than averaging away across echoes. Being a single
global constant rather than a per-voxel effect, it can be aggregated over a mask
without dilution.

The intercept measures neither the error in the offset nor the error in the
field. It measures whether the two are mutually consistent, which is the
property the data can check. Its limitation follows from
#link(<prop:alias>)[Prop. 1]: a matched pair gives $c = 0$, so the intercept
cannot rank it against the correct solution. Where more than one candidate
remains consistent, the data has been exhausted.

= The selection algorithm

The procedure runs once per frame. Its output is an integer correction $N$
applied to #phiuw before the offset of @eq:est is formed, leaving a residual
error of $M + N$, and the aim is $N = -M$.

#defn[
*Step 1.* Consider $N in {-1, 0, +1}$. A single step is a full wrap of global
field, and the global correction has already brought the field inside the
half-wrap, so a correction of more than one wrap is not reachable.

*Step 2.* For each $N$, shift #phiuw by $2 pi N$, form the implied offset by
@eq:est, remove it from both echoes, unwrap in time, fit @eq:fit, and record
$abs(c)$ aggregated over a conservative brain mask.

*Step 3.* Discard any candidate whose $abs(c)$ exceeds a cutoff. By
#link(<prop:offset>)[Prop. 2], neighbouring candidates are separated in
intercept by $abs(W(2 pi k))$, so the cutoff is placed at half that separation,
which makes the test a nearest-neighbour assignment between two possibilities
rather than a tuned threshold. The implementation also measures the largest
intercept observed among the candidates and adopts the stricter of the two
scales, which keeps the test honest when the second unwrapping has already
absorbed part of the residue.

*Step 4.* If one candidate survives, take it, whatever its field. If none
survives, change nothing, since a failed fit is not evidence for any particular
candidate. If several survive, they are indistinguishable to the data, and
@eq:argmin selects the one with the smallest estimated $abs(hat(f))$.
]

The two stages answer different questions, and neither suffices alone. Step 3
uses evidence and rejects candidates the data contradicts. Step 4 uses an
assumption, and applies it only among candidates the data has already accepted,
which prevents the assumption from overriding the evidence.

Step 4 applies the same rule as @eq:argmin, so it is worth stating what it
changes. The global correction takes an unweighted median of *rounded* turn
counts over a dilated mask. Rounding discards the sub-wrap information that
indicates how close the decision is, and dilation admits edge voxels whose
fields are extreme. Step 4 estimates the field itself, weighted, over a
conservative mask. The rule is unchanged; the statistic it is applied to is not.

One degenerate case deserves mention. If $k$ is a whole number then
$W(2 pi k M) = 0$ for every $M$, so the intercept carries no information.
@eq:pair then gives $hat(theta)_M = W(theta)$ regardless of $M$, so no choice
alters the output. The implementation detects the zero separation and makes no
change, which is the correct behaviour.

= Estimating the field

Step 4 compares $abs(hat(f))$ against the half-wrap, so the reliability of the
comparison depends on what that estimate measures and on how steady it is
between frames.

== What the rule refers to

@eq:argmin is a statement about *bulk tissue*, since bulk tissue is what the
scanner's frequency adjustment centres. It is not a statement about the mean
field, nor about the median field over whatever mask happens to be available.

The field distribution across a brain is asymmetric. It has a narrow peak at the
bulk-tissue value together with long tails from air-tissue interfaces, sinuses,
signal dropout and mask edges, where susceptibility gradients are large. Any
quantile of the whole distribution is displaced toward whichever tail is
heavier. Writing the distribution as a mixture,

$
  p(f) = (1 - epsilon) dot p_"bulk" (f) + epsilon dot p_"tail" (f),
$

the rule refers to the centre of $p_"bulk"$, and the estimator should recover it
with $epsilon$ as small as possible. Two consequences follow.

Enlarging the mask makes the estimate worse. Dilating a brain mask admits
precisely the voxels that populate $p_"tail"$, which raises $epsilon$ and
displaces any quantile of $p$ further from the bulk centre. This reverses the
usual expectation that more data improves an estimate.

A bias of a fraction of a wrap is enough to change the decision. The comparison
is against the half-wrap, so estimator bias adds directly to the apparent
distance from the boundary. A subject whose bulk field already lies near the
half-wrap can be carried across it by bias alone, at which point the decision
reflects the statistic rather than the data.

== The estimator

MEDIC uses a magnitude-weighted median over an eroded brain mask. Erosion
lowers $epsilon$ geometrically by removing edge voxels, and the weighting lowers
it by suppressing low-signal voxels, which are over-represented in the tails.
The weight is the magnitude of the second echo, because the field is read from a
difference whose noise is dominated by the weaker of the two.

A median is used rather than a mean or an M-estimator. The bulk distribution is
broad and right-skewed, and the half-wrap boundary falls within its body rather
than out in a tail, so a statistic pulled toward the mean reads systematically
closer to the boundary without describing bulk tissue any more accurately.

Continuity is the binding requirement. The estimate decides a threshold
comparison that has to give the same answer on consecutive frames of one
subject, and an unstable estimator would reintroduce at this stage exactly the
inconsistency described in section 4. A median moves smoothly as the samples
move. Estimators defined by an $arg max$, including modes in any of their usual
forms, do not, since field distributions are frequently multimodal and a
near-tie between peaks makes the estimate jump between frames. Stability across
frames matters more here than freedom from bias.

