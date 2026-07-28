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
    Phase offset ambiguities in multi-echo field mapping
  ]
  #v(0.9em)
  #text(size: 10.5pt)[Andrew Van]
  #v(0.15em)
  #text(size: 9.5pt)[July 27, 2026]
]

#v(1em)

#outline(indent: auto, depth: 2)

#v(1.2em)

= Why the phase offset matters

Multi-echo field mapping reads the field off the *rate* at which phase
accumulates across echoes. That rate is the quantity of interest; the constant
it accumulates from is a nuisance. Every voxel carries a receive-coil phase
offset $theta$ — present already at $t = 0$, identical on every echo, and set by
coil geometry rather than by the field.

$theta$ cannot simply be ignored. It varies rapidly in space, steeply near coil
elements and discontinuously between channels, so the measured phase is not
spatially smooth even where the field is. Spatial unwrapping assumes
neighbouring voxels differ by less than half a turn, and on raw phase that
assumption fails. Removing $theta$ first restores the smoothness that everything
downstream depends on.

The difficulty is that $theta$ is never measured. It has to be inferred from the
same wrapped phase it corrupts, and that inference is not unique: several
distinct $(theta, f)$ pairs explain the same data. Some of the resulting
ambiguity can be resolved from additional echoes, and some of it provably
cannot.

This note draws that line. It enumerates the ways an estimate can be wrong,
shows which of them leave a trace in the data, and derives what to do about
each. Everything below follows from the signal model.

= The measurement

== Signal model

Acquire $E$ echoes at times $t_0 < t_1 < ... < t_(E-1)$. Field evolution is
linear in time, so before wrapping, the phase at echo $e$ would be
$theta + 2 pi f t_e$, with $f$ the *field offset* in Hz — the quantity wanted —
and $theta$ the coil offset described above.

What is recorded is that value wrapped into a single turn. Writing
$W(x) = arg(e^(i x))$ for wrapping to $(-pi, pi]$,

$
  phi_e = W(theta + 2 pi f t_e).
$ <eq:model>

#key[
The entire difficulty comes from $W$. Without it, two echoes would determine
$theta$ and $f$ exactly by fitting a straight line. With it, the line is
observed only modulo $2 pi$, and infinitely many $(theta, f)$ pairs remain
compatible with the data.
]

== Evenly spaced echoes

An echo-planar train produces echoes at constant spacing,

$
  t_e = "TE"_0 + e dot #dte, quad e = 0, 1, ..., E-1,
$

with $"TE"_0$ the first echo time and #dte the spacing. The single most
important derived quantity below is their ratio,

$
  k = "TE"_0 / (#dte),
$

dimensionless and fixed by the protocol. Two field scales also recur:

#defn[
- *Wrap spacing* $1 slash #dte$ (Hz) — how far apart the candidate field
  solutions sit.
- *Half-wrap* $1 slash (2 #dte)$ (Hz) — the midpoint between neighbouring
  candidates, and therefore the decision boundary for any rule preferring the
  smallest field.
]

== Removing the offset

Because $theta$ is common to every echo, it cancels in a difference. The phase
difference of the first two echoes removes the coil contribution and leaves a
quantity proportional to the field:

$
  Delta phi = W(phi_1 - phi_0) approx W(2 pi f #dte).
$

This is still wrapped, but a difference of *neighbouring* echoes varies slowly
in space, so a region-growing algorithm can restore the missing turns and
produce an unwrapped difference #phiuw. Extrapolating back to $t = 0$ then
recovers the offset:

$
  hat(theta) = W(phi_0 - k dot #phiuw).
$ <eq:est>

#key[
Spatial unwrapping recovers *relative* phase between voxels. It cannot know the
absolute number of turns, so it determines #phiuw only up to a global additive
constant $2 pi M$ for some unknown integer $M$. Pinning down $M$ is the decision
this note is about.
]

= A taxonomy of wrong answers

Suppose some estimate $(hat(theta), hat(f))$ has been produced. To classify how
it can be wrong, remove the estimated offset from the data and inspect what
remains:

$
  psi_e = phi_e - hat(theta).
$

Were the estimate perfect, $psi_e$ would be exactly proportional to $t_e$: a
straight line through the origin whose slope is the field. So fit

$
  psi_e approx c + s dot t_e
$ <eq:fit>

and read off two numbers — the *intercept* $c$ and the *slope* $s$. Every
possible error is a displacement in this $(c, s)$ plane, and the branch
ambiguity produces two cases.

#defn[
#set enum(numbering: "A.")
+ *Both move together.* Offset and field are both wrong, in a matched
  combination that leaves $psi_e$ unchanged. Invisible.
+ *The offset moves alone.* $c$ shifts while $s$ stays correct. Visible as a
  line that misses the origin.
]

Both are taken in turn below. The distinction matters because a test that
measures $c$ finds case B and is blind to case A *by construction* — not through
any weakness of implementation.

== Ambiguity A: an exact symmetry of the data

#defn[
*Proposition 1.* For evenly spaced echoes and any integer $N$, define
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
For evenly spaced echoes $t_e slash #dte = k + e$, so the bracket equals the
integer $e$ and the term vanishes under $W$. #h(1fr) $qed$
]

Offset error and field error are locked together here: shift the field by one
wrap spacing and the offset compensates exactly. The recorded data is
*bit-identical* — not approximately, but exactly.

#key[
No statistic computed from the phase can separate these alternatives, because
there is nothing to separate: they are the same measurement. Resolving A
requires information from outside the data, which is to say a prior.
]

=== Why more echoes do not help

It is natural to expect a five-echo acquisition to constrain the field better
than a two-echo one. For this question it does not.
#link(<prop:alias>)[Prop. 1] holds for all echoes simultaneously, and its period
$1 slash #dte$ depends only on the spacing, which every pair shares. All pairs
therefore alias at the same period and agree with one another.

Unequal spacing would break the tie — two pairs whose aliases have
incommensurate periods are simultaneously satisfied only by the true field — but
an EPI echo train produces equal spacing naturally, so that escape is not
usually available.

== Ambiguity B: the offset moves alone

Now suppose spatial unwrapping settled on the wrong number of turns, so #phiuw
is off by $2 pi M$, while the field has already been fixed by an earlier stage
and does not follow.

#defn[
*Proposition 2.* If #phiuw carries a branch error of $M$ wraps, @eq:est returns
$hat(theta)_M = W(theta - 2 pi k M)$, and the offset-removed phases satisfy
$
  psi_e equiv 2 pi f t_e + c mod 2 pi,
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

Comparing against @eq:fit: the slope is untouched and the intercept has moved to
$c = W(2 pi k M)$. Because $c$ is the *same on every echo*, it does not resemble
noise and does not average away — it is a rigid vertical displacement of the
whole line.

#key[
This is the one detectable case, and the reason is worth stating plainly: offset
and field have become mutually inconsistent, and consistency between them is
something the data can check.
]

= Detecting ambiguity B

By #link(<prop:offset>)[Prop. 2] a branch error of $M$ wraps places the constant
$c = W(2 pi k M)$ on every echo. Fitting the line of @eq:fit and reading its
intercept therefore measures the error directly. Because the effect is a single
global constant rather than a per-voxel one, the intercept can be aggregated
over many voxels, suppressing noise without diluting the signal.

The procedure follows: for each candidate $M$, reconstruct the offset it
implies, remove it, fit, and record $abs(c)$. The correct branch extrapolates
through the origin; a wrong one does not.

== The rejection scale is analytic

Any test needs a scale — how large must $abs(c)$ be before a candidate is
rejected? Here the model supplies it rather than the user. Candidate intercepts
lie on a lattice of spacing

$
  Delta c = abs(W(2 pi k)),
$ <eq:step>

a function of the echo times alone. Classifying against $Delta c slash 2$ is
nearest-neighbour assignment on that lattice — the midpoint between two adjacent
possibilities — rather than a tuned threshold.

== Where the test loses power

#defn[
*Integer $k$.* If $k$ is a whole number then $c = W(2 pi k M) = 0$ for every
$M$, so the intercept carries no information. But by
#link(<prop:offset>)[Prop. 2] the recovered offset $hat(theta)_M = W(theta)$ is
*also unchanged*, so no choice of branch can affect the output. The degeneracy
is harmless, and @eq:step returning zero is the correct signal to do nothing.
]

Away from integer $k$ the achievable separation varies smoothly, peaking near
$k = 2 slash 3$ and shrinking to either side. For small $k$ in particular,
several candidates can be exactly through-origin at once: the alias of
#link(<prop:alias>)[Prop. 1] becomes reachable *within the candidate set*, and
the intercept cannot rank the survivors. Since an EPI train has $"TE"_0$ shorter
than the echo spacing, $k < 1$ always, so this regime is not exotic.

= Choosing among indistinguishable candidates

Once more than one candidate is through-origin the data has been exhausted and
something else must decide. Two facts constrain what that can be.

#defn[
*A consistency test alone is insufficient.* Tied candidates have equal
intercepts to within numerical noise. Ranking them is ranking noise.
]

#defn[
*A field-magnitude prior alone is insufficient.* A rule preferring the smallest
$abs(f)$ is *symmetric about zero*: candidates at $+delta$ and $-delta$ are
equidistant and cannot be separated. Such pairs do arise, because re-wrapping
inside the unwrapper can map candidates $M$ and $-M$ to equal $abs(f)$.
]

#key[
The two are complementary in a precise sense. The intercept is blind to the
alias direction of A, along which offset and field move together; the magnitude
prior is blind to a sign reversal, since $+delta$ and $-delta$ are equidistant
from zero. Together they cover both, which is why a workable rule needs both
stages.
]

The resulting rule: reject candidates whose intercept is inconsistent; if one
survives, take it; if several survive, apply the prior among them; if none
survives, change nothing, since a failed fit is not evidence for any particular
candidate.

= The prior and how to estimate it

== What the prior actually asserts

Before imaging, the scanner sets its demodulation frequency to the centre of the
water resonance over the shim volume. The physical content of the prior is
therefore:

#key[
The *bulk-tissue* field is near zero. Not the mean field, and not the median
field over whatever mask happens to be convenient.
]

The distinction is not pedantry. The field distribution over a brain is
asymmetric: a narrow peak at the bulk-tissue value, plus long tails from
air–tissue interfaces, sinuses, signal dropout and mask edges, where
susceptibility gradients are large. Any statistic that is a *quantile of the
whole distribution* — the median included — is displaced toward whichever tail
is heavier.

== Estimating the level

Model the distribution as a mixture of a narrow bulk component and a broad tail
component,

$
  p(f) = (1 - epsilon) dot p_"bulk" (f) + epsilon dot p_"tail" (f),
$

with the centre of $p_"bulk"$ the quantity the shim actually set. Two practical
consequences follow.

#defn[
- *Enlarging the mask makes the estimate worse.* Dilating a brain mask admits
  precisely the voxels populating $p_"tail"$, raising $epsilon$ and displacing
  any quantile of $p$ further from the bulk centre. A generous mask is worse
  than a conservative one — the opposite of the usual intuition that more data
  helps.
- *A bias of a fraction of a wrap is decisive.* The prior compares
  $abs(hat(f))$ against the half-wrap $1 slash (2 #dte)$. Estimator bias adds
  directly to the distance from that boundary, so a subject whose bulk field
  lies near the half-wrap can be pushed across it by bias alone, and the choice
  then turns on an artefact of the statistic rather than a property of the data.
]

The estimator used is a magnitude-squared weighted median over an eroded brain
mask. The erosion and the weighting both reduce $epsilon$: erosion by excluding
edge voxels geometrically, weighting by suppressing the low-signal voxels that
are over-represented in the tails.

#key[
*Continuity is the binding requirement.* The estimate feeds a threshold
comparison against $1 slash (2 #dte)$, and that comparison must give the same
answer on consecutive frames of the same subject. A median is a continuous
functional of the data: perturb the samples slightly and it moves slightly.
Estimators defined by an $arg max$ — modes, in any of their usual forms — are
not, and near-ties in a multimodal distribution make them flip between frames.
For this decision, stability across frames outweighs freedom from bias.
]

= What remains unresolvable

Two limits are properties of the acquisition rather than of any algorithm.

#defn[
*Ambiguity A cannot be corrected.* By #link(<prop:alias>)[Prop. 1] the data is
invariant under the alias, so when the true field genuinely exceeds the
half-wrap, no rule recovers it: the prior folds it back, and that is that.
Whether the situation is *reachable* is governed by #dte, bounded below by the
readout duration — echoes cannot be spaced closer than they can be read out —
and above by $T_2^*$, since several echoes must fit before the signal decays.
Those bounds confine #dte, and so hold the half-wrap threshold above typical
shim residuals.
]

#defn[
*Higher field strength does not simply make it worse.* Shim residuals grow with
$B_0$, which argues for more risk. But $T_2^*$ shortens, forcing #dte down and
therefore raising the $1 slash (2 #dte)$ threshold. The two effects push the
same quantity in opposite directions.
]

= Summary

#key[
Wrapped phase determines the coil offset and the field only up to a family of
alternatives. Sorting them out means asking, of each candidate, whether the
offset and the field it implies are mutually consistent.

Two cases arise. When offset and field are both wrong in a matched way (*A*) the
data is unchanged, and only a prior can help. When the offset is wrong alone
(*B*) every echo carries the same constant, so extrapolating the echoes back to
$t = 0$ exposes it, against a rejection scale $abs(W(2 pi k))$ fixed by the echo
times.

Where several candidates remain consistent, the prior that decides is that
*bulk-tissue* field is near zero, estimated by a weighted median over a
conservative mask. Consistency and prior are blind to opposite symmetries, so a
workable rule needs both.
]
