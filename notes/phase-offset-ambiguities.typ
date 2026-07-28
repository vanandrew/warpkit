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

// `breakable: false` for short boxes that must not split across a page; the
// long specification box in section 8 is taller than a page and has to break.
#let defn(body, breakable: true) = block(
  width: 100%, stroke: 0.5pt + luma(160), inset: 9pt, radius: 2pt,
  breakable: breakable, body,
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
  [$N$],            [integer],    [index of a candidate; $N = 0$ is what the unwrapper returned],
  [$N^*$],          [integer],    [index of the correct candidate; not observable],
  [$hat(theta)$],   [wrapped],    [the offset estimate],
  [$psi_e$],        [neither],    [recorded phase less an estimated offset],
)

*Units.* Echo times are in seconds throughout, so a phase slope divided by
$2 pi$ is a frequency in Hz.
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
when it happens to land in $(-pi, pi]$ already.

== Echo spacing

An echo-planar train produces echoes at constant spacing,

$
  t_e = "TE"_0 + e dot #dte, quad e = 0, 1, ..., E-1,
$ <eq:spacing>

with $"TE"_0$ the first echo time and #dte the spacing. Two field scales follow
from #dte alone. The *wrap spacing* $1 slash #dte$ separates neighbouring
candidate solutions for the field. The *half-wrap* $1 slash (2 #dte)$ is half
that spacing, the distance from any candidate to the midpoint between it and its
neighbour, and serves as the decision boundary for any rule that prefers the
smallest field.

== Recovering the offset

The construction below is MCPC-3D-S. Because $theta$ is common to every echo, it
cancels in a difference. For the first two echoes,

$
  Delta phi = W(phi_1 - phi_0)
  = W((theta + 2 pi f t_1) - (theta + 2 pi f t_0))
  = W(2 pi f #dte).
$

The middle step follows from @eq:diffid, since each $phi_e$ differs from
$theta + 2 pi f t_e$ by whole turns. The wrap remains outside the subtraction
because $phi_1 - phi_0$ is a difference of wrapped values.

$Delta phi$ is itself wrapped, but a difference of neighbouring echoes varies
slowly in space, so a region-growing algorithm can restore its missing turns.
Spatial unwrapping recovers phase differences between voxels rather than
absolute turn counts, so what it returns may differ from the true unwrapped
difference by a whole number of turns. Write #phiuw for what it returns.
Extrapolating to $t = 0$ then recovers the offset, with
$k = "TE"_0 slash #dte$,

$
  hat(theta) = W(phi_0 - k dot #phiuw).
$ <eq:est>

The outer $W$ is required because #phiuw is unbounded and $k dot #phiuw$ with
it. That unrecovered number of turns is the only thing @eq:est leaves open.

= The candidate ladder

The turn count in #phiuw is unknown, so consider every value it could take.
Shifting what the unwrapper returned by whole turns,

$
  #phiuw^((N)) = #phiuw + 2 pi N, quad N in ZZ,
$ <eq:shift>

produces one *candidate* per integer, with $N = 0$ the value the unwrapper
(ROMEO) actually returned. A candidate is not a partial answer. Each one fixes an
offset and a field together, and both follow from what has already been
established.

The offset comes from putting @eq:shift into @eq:est. The shift enters multiplied
by $k$, so

$
  hat(theta)_N = W(phi_0 - k(#phiuw + 2 pi N)) = W(hat(theta)_0 - 2 pi k N).
$ <eq:offn>

The field is the slope of the unwrapped difference, phase gained per unit time
over the interval #dte, so

$
  hat(f)_N = (#phiuw^((N))) / (2 pi #dte) = hat(f)_0 + N / (#dte).
$ <eq:fldn>

One candidate is correct. Write $N^*$ for its index, so that
$hat(theta)_(N^*) equiv theta$ and $hat(f)_(N^*) = f$. Nothing observable
identifies $N^*$, which is the whole difficulty.

Two features of @eq:offn and @eq:fldn matter later. The first is that a single
integer fixes both quantities. There is no candidate carrying the correct field
alongside an incorrect offset, or the reverse: they move together, the field in
steps of $1 slash #dte$ and the offset in steps of $2 pi k$. The second is that
the candidates are *discrete*. The field is not uncertain within an interval; it
is determined up to a set of isolated values spaced $1 slash #dte$ apart, with
nothing in between available as an answer. Whatever goes wrong later is therefore
wrong by a whole wrap, never by a little.

== The rule ROMEO applies <sec:rule>

One candidate has to be chosen, and ROMEO's global correction is what chooses it.
After spatially unwrapping, it subtracts $2 pi$ times the median rounded turn
count over the mask, which leaves that median at zero,

$
  "median" round(#phiuw slash 2 pi) = 0.
$ <eq:cg>

That condition reads directly as a statement about the field. By @eq:fldn a turn
of the difference is $1 slash #dte$ of field, so rounding the turn count to zero
is the same as requiring

$
  abs(hat(f)_0) < 1 / (2 #dte),
$ <eq:half>

the field lying inside the *half-wrap*. Because neighbouring candidates are a
full wrap apart, exactly one of them satisfies @eq:half, so the rule selects the
candidate of smallest field magnitude: the rung of the ladder nearest zero.
@eq:cg tests that condition through one particular statistic, an unweighted
median of rounded turn counts over its own mask; which summary of the field is
the right one is taken up in @sec:field.

@eq:half is an assumption rather than a measurement, and its justification is
physical. Before imaging, the scanner sets its demodulation frequency to the
centre of the water resonance, so the residual field is expected to be a small
offset rather than an arbitrary one.

Where that expectation holds the rule returns $N^*$. Where it does not, it returns
a neighbouring rung instead, and nothing in the data marks the difference.

#defn[
*Proposition 1.* For evenly spaced echoes, every candidate predicts the same
recorded phase. That is, $W(hat(theta)_N + 2 pi hat(f)_N t_e)$ does not depend on
$N$, at any echo.
] <prop:alias>

The rule is therefore forced rather than merely convenient: no statistic computed
from the phase can rank the candidates. Proved in #link(<app:alias>)[Appendix A],
which draws out the consequences and sets out what the result does not cover.

= Why the rule fails on some frames <sec:fail>

A wrong candidate, chosen once and applied to a whole acquisition, would matter
very little. Every frame would be displaced by the same $1 slash #dte$ of field, and the
field offset is already referenced to an arbitrary demodulation frequency, so a
fixed additive constant changes neither temporal contrast nor spatial structure.

The difficulty is that the choice is not made once. Three properties of @eq:cg
combine to produce a failure.

The first is that it is evaluated on every frame. A functional acquisition is
processed frame by frame, and each frame applies @eq:cg to its own data, without
reference to the frames around it.

The second is that the left side of @eq:cg is a sample median over a mask of
voxels, each carrying its own field. It absorbs whatever varies between frames,
including physiological change, subject motion, and noise in the voxels the mask
admits.

The third is that @eq:half is a comparison against a fixed boundary. When
$abs(hat(f)_0)$ is far below $1 slash (2 #dte)$ the margin is wide and no
plausible perturbation of the median changes which candidate satisfies the
inequality. As the field approaches the half-wrap that margin approaches zero.

The practical consequence is that ROMEO's global correction is occasionally off by
a single turn. For a subject whose field lies near the half-wrap the candidate is
chosen on each frame independently, by a noisy median sitting close to a
threshold, so different frames fall on different sides of it. Because the
candidates are a full wrap apart, a frame that falls on the wrong side is
displaced by $1 slash #dte$ rather than by a small amount. The resulting time
series is stable to a fraction of a hertz across most frames and displaced by a
full wrap on a scattered subset, with no intermediate values.

The damage is temporal. A uniformly wrong field is harmless, but a field taking
two values within a single acquisition corrupts the temporal structure the
acquisition was performed to measure.

= Detecting a failed choice <sec:detect>

By #link(<prop:alias>)[Prop. 1] no test can rank one candidate against another.
The test in this section asks a different question: whether what the pipeline has
produced is a candidate at all.

The question has content because the offset and the field come from two separate
steps. The first uses #phiuw to form the offset, by @eq:offn. The second removes
that offset from every echo, spatially unwraps the first echo — applying @eq:cg a
second time, now to offset-removed phase — and unwraps the remaining echoes
against it to recover the field. Nothing carries the first step's choice of shift
into the second.

Consider a frame on which the first unwrapping has tipped, so the offset removed
belongs to a neighbouring rung rather than the correct one. The echoes still carry
the correct field, and the second unwrapping recovers it, because @eq:cg prefers
the small field and in offset-removed data the correct field is the small one. What
comes out therefore has an offset from one rung and a field from another. That
combination is not a rung of the ladder, so
#link(<prop:alias>)[Prop. 1] does not cover it, and it leaves a signature in the
data.

== The signature

Removing an offset $hat(theta)_N$ from the recorded phase leaves

$
  psi_e = phi_e - hat(theta)_N,
$

a difference of two wrapped values, and therefore not itself a phase. Unwrapping
it in the echo direction and fitting a line,

$
  psi_e approx c + s dot t_e,
$ <eq:fit>

gives a slope $s$, which is the field, and an intercept $c$, the value the line
extrapolates to at $t = 0$.

The intercept is the signature. Removing the offset that belongs to a given field
leaves phase that starts at zero, so a matched offset and field give a line
through the origin, $c = 0$. A mismatched pair does not, and by how much is fixed.

#defn[
*Proposition 2.* Let the echo unwrap return the correct field, while the offset
removed was $hat(theta)_N$ for some $N != N^*$. Then
$
  psi_e equiv 2 pi f t_e + c mod 2 pi,
  quad quad c = W(2 pi k (N - N^*)),
$
with $c$ identical at every echo.
] <prop:offset>

#proof[
*Proof.* By @eq:offn, $hat(theta)_N equiv hat(theta)_(N^*) - 2 pi k (N - N^*)
equiv theta - 2 pi k (N - N^*)$. Then
$psi_e equiv (theta + 2 pi f t_e) - (theta - 2 pi k (N - N^*))
= 2 pi f t_e + 2 pi k (N - N^*)$, and the constant does not depend on $e$.
#h(1fr) $qed$
]

Two properties make $c$ usable in practice. Being identical at every echo, it
displaces the whole fitted line rather than averaging away across echoes. Being one
global constant rather than a per-voxel effect, it can be aggregated over a mask
without dilution.

== What the intercept can and cannot settle

The intercept measures neither an error in the offset nor an error in the field. It
measures whether the two are mutually consistent, and consistency between them is
the one property the data can check.

That makes it a filter rather than a ranking. Among the shifts of @eq:shift, those
with a large intercept are not candidates and can be discarded. Those with
$c approx 0$ are candidates, and #link(<prop:alias>)[Prop. 1] then applies in
full: they predict identical data and the intercept cannot separate them. Choosing
among the survivors still requires @eq:half.

Both steps are needed, for different reasons. The filter is needed because a shift
can report a reasonable-looking field while carrying an offset that does not
generate it, and since that offset is subsequently removed from every echo, the
error propagates into everything downstream. The rule is needed because the filter
leaves more than one survivor. What the filter contributes is that when the first
unwrapping tips, the surviving set is not the same set, and applying the rule to
the shifted set recovers the frame.

= The selection algorithm

The procedure runs once per frame and returns one shift $N$ to apply to #phiuw.

#defn[
*Step 1.* Consider $N in {-1, 0, +1}$. One step is a full wrap of global field,
and @eq:cg has already brought the field inside the half-wrap, so a correction of
more than one wrap is not reachable.

*Step 2.* For each $N$, form $hat(theta)_N$ by @eq:offn, remove it from both
echoes, unwrap in time, fit @eq:fit, and record $abs(c)$ aggregated over a
conservative brain mask.

*Step 3.* Discard any candidate whose $abs(c)$ exceeds a cutoff. By
#link(<prop:offset>)[Prop. 2] the candidates of Step 1 have intercepts $0$ and
$plus.minus W(2 pi k)$, so the cutoff is placed at half that separation, making
the test a nearest-neighbour assignment between two possibilities rather than a
tuned threshold. The implementation also measures the largest intercept observed
among the candidates and adopts the stricter of the two scales, which keeps the
test honest when the echo unwrap has already absorbed part of the residue.

*Step 4.* If one candidate survives, take it, whatever its field. If none
survives, change nothing, since a failed fit is not evidence for any candidate.
If several survive, they are indistinguishable to the data, and @eq:half decides
among them by smallest $abs(tilde(f)_N)$, the field the reconstruction actually
returned.
]

Steps 3 and 4 answer different questions and neither suffices alone. Step 3 uses
evidence, rejecting candidates the data contradicts. Step 4 uses the assumption
of @eq:half, and applies it only among candidates the data has already accepted,
which is what stops the assumption from overriding the evidence.

Step 4 restates @eq:half, so it is worth being clear about what it changes.
@eq:cg evaluates the condition as an unweighted median of *rounded* turn counts
over a dilated mask. Rounding discards the sub-wrap information that says how
close the decision was, and dilation admits edge voxels whose fields are extreme.
Step 4 evaluates $abs(tilde(f)_N)$ directly, weighted, over a conservative mask.
The condition is the same; the statistic is not, and @sec:field is about why that
matters.

One degenerate case deserves mention. If $k$ is a whole number then
$W(2 pi k (N - N^*)) = 0$ for every $N$, so the intercept carries no
information. By @eq:offn the offset $hat(theta)_N = W(hat(theta)_0)$ is then also
independent of $N$, so no choice alters the output, and leaving $N = 0$ is
correct. A guard on $delta = 0$ states that intent, but it need not be what
enforces it: in floating point $W(2 pi k)$ evaluates to a rounding residue near
$10^(-16)$ rather than to zero, and the cutoff that residue sets is then small
enough that every candidate fails it, so the empty-survivor case of Step 4 returns
$N = 0$ regardless. Either route gives the same answer.

= Estimating the field <sec:field>

Step 4 compares $abs(tilde(f))$ against the half-wrap, so the reliability of the
comparison depends on what that estimate measures and on how steady it is
between frames.

@eq:half is a statement about *bulk tissue*, since bulk tissue is what the
scanner's frequency adjustment centres. It is not a statement about the mean
field, nor about the median field over whatever mask happens to be available.

A median is used rather than a mean or an M-estimator. The bulk distribution is
broad and right-skewed, with long tails from air-tissue interfaces, sinuses and
mask edges, and the half-wrap boundary falls within its body rather than out in a
tail. A statistic pulled toward the mean therefore reads systematically closer to
the boundary without describing bulk tissue any more accurately.

MEDIC uses that median over an eroded brain mask, weighted by squared magnitude.
Both choices work against the tails: erosion removes edge voxels geometrically,
and the weighting suppresses low-signal voxels, which the tails over-represent.
The weight is taken from the second echo rather than the first, because the field
is read from a difference whose noise is dominated by the weaker of the two.


= The algorithm in full

Collected here as a specification. All quantities are per-voxel unless a
reduction over a mask is written explicitly, and the whole procedure is applied
independently to each frame.

#defn[
*Input.* Magnitudes $m_0, m_1$ and wrapped phases $phi_0, phi_1$ at the first two
echo times $"TE"_0$ and $"TE"_1$; an unwrapping mask $Omega$; a conservative mask
$Omega_"c" subset.eq Omega$ for reductions.

*Output.* The phase offset $hat(theta)$.

*Constants.* $#dte = "TE"_1 - "TE"_0$, $quad k = "TE"_0 slash #dte$, $quad
delta = abs(W(2 pi k))$.

#v(0.5em)
*1. Unwrap the phase difference.*
$
  S = m_0 m_1 e^(i (phi_1 - phi_0)),
  quad
  Delta phi = arg S,
  quad
  #phiuw = "ROMEO"(Delta phi; abs(S), Omega).
$
ROMEO returns a spatially unwrapped field with @eq:cg imposed.

*2. Reconstruct each candidate.* For $N in {-1, 0, +1}$:
$
  hat(theta)_N &= W(phi_0 - k (#phiuw + 2 pi N)), \
  (psi_0^((N)), psi_1^((N))) &= "ROMEO"_"4D" ((phi_0, phi_1) - hat(theta)_N;
    (m_0, m_1), Omega, ("TE"_0, "TE"_1)), \
  s^((N)) &= (psi_1^((N)) - psi_0^((N))) slash #dte,
  quad quad
  tilde(f)_N = s^((N)) slash 2 pi, \
  c_N &= abs(op("med")_(Omega_"c") [psi_0^((N)) - s^((N)) "TE"_0]).
$
$"ROMEO"_"4D"$ imposes @eq:cg a second time, which is why $tilde(f)_N$ need not
equal the ladder field $hat(f)_N$ of @eq:fldn, and so why $c_N$ can be nonzero at
all.

*3. Set the rejection scale.*
$
  sigma = min(max_N c_N, delta),
  quad quad
  cal(C) = {N : c_N < sigma slash 2}.
$
If $delta$ is zero, return $hat(theta)_0$: $k$ is then integral, so by @eq:offn the
offset does not depend on $N$ and no choice alters the output. Test this within a
tolerance rather than against exact zero, since $W(2 pi k)$ evaluates to a small
rounding residue in floating point.

*4. Select.*
$
  hat(N) = cases(
    0 & "if" cal(C) = nothing,
    N & "if" cal(C) = {N},
    display(op("arg min")_(N in cal(C)) abs(op("med")_(Omega_"c")^(w) tilde(f)_N))
      & "otherwise," quad w = m_1^2.
  )
$

*5. Return* $hat(theta)_(hat(N))$.
]

Three details of the specification carry the reasoning of the earlier sections.
The two reductions differ deliberately: step 2 takes a plain median, since it
estimates one global constant, while step 4 takes a median weighted by $m_1^2$ of
a distribution whose shape matters, for the reasons in @sec:field. The scale
$sigma$ in step 3 takes the smaller of the analytic separation $delta$ and the
largest intercept observed, which makes the consistency test harder to pass and so
biases the procedure toward leaving $hat(N) = 0$. And the cases in step 4 are
ordered, not interchangeable: an empty $cal(C)$ returns $0$ rather than consulting
@eq:half, a singleton is taken whatever its field, and only a tie reaches
@eq:half.

== What the components must provide

Three things are deliberately left open: the spatial unwrapper, the two masks, and
the weighted median. What matters is the property each must satisfy.

*The unwrapper.* Steps 1 and 2 need one that imposes @eq:cg on what it returns,
step 2 on the echo it unwraps first. Nothing else about it is load-bearing: the
selection logic reads only the returned phase, and @sec:detect needs only that the
two unwrapping steps choose their shifts independently, since that is what pairs an
offset from one rung with a field from another and leaves a signature to detect.

*The masks.* $Omega$ is generous, brain plus a margin, because the unwrapper must
grow through the whole region of interest. $Omega_"c"$ is interior, excluding edge
and low-signal voxels, because both reductions summarise a distribution whose tails
come from exactly those voxels (@sec:field). Nothing depends on how either is
built, only on their differing in that way, with $Omega_"c"$ well inside $Omega$.

*The weighted median.* $op("med")^w$ is the value at which cumulative weight, over
voxels sorted by value, first reaches half the total. The convention matters only
at an exact tie, where any deterministic choice will do.

== Reference data

The failure of @sec:fail can be reproduced on public data.
#link("https://openneuro.org/datasets/ds006131")[`ds006131`] (v2.0.1, "PAFIN")
acquires five echoes at $"TE"_0 = 14.2$ ms and $#dte = 24.73$ ms, so
$delta = 2.68$ rad and the half-wrap is $20.22$ Hz. Two of its 51 subjects carry a
bulk field close enough to that boundary for @eq:cg to tip on individual frames:

#defn(breakable: false)[
#set par(justify: false)
#set text(size: 9.5pt)
`ds006131/sub-<id>/ses-1/func/` \
`sub-<id>_ses-1_task-bao_dir-AP_run-01_echo-{1..5}_part-{mag,phase}_bold.nii.gz` \
#v(0.35em)
$hat(N) != 0$ on 19 of 243 frames for sub-20828 and on 158 of 243 for sub-24630.
]

Afterwards neither field-map time series contains a wrap-sized step, where before
each carried one on every affected frame. The counts depend on the masks in use, so
treat them as a guide: what is robust is that these two subjects need a nonzero
$hat(N)$ on a scattered subset of frames and the other 49 do not. An implementation
returning $hat(N) = 0$ throughout on them has not reproduced the correction; one
returning nonzero shifts across many subjects is over-correcting.

#counter(heading).update(0)
#set heading(numbering: "A.1")

= Proof of Proposition 1 <app:alias>

#proof[
*Proof of #link(<prop:alias>)[Prop. 1].* Working modulo $2 pi$, so that $W$ of either side agrees, @eq:offn and
@eq:fldn give
$
  hat(theta)_N + 2 pi hat(f)_N t_e
    &equiv hat(theta)_0 - 2 pi k N + 2 pi (hat(f)_0 + N slash #dte) t_e \
    &= hat(theta)_0 + 2 pi hat(f)_0 t_e + 2 pi N (t_e slash #dte - k).
$
By @eq:spacing, $t_e slash #dte = k + e$, so the parenthesis is the integer $e$
and the final term is $2 pi N e$, a whole number of turns. Both sides therefore
wrap alike. #h(1fr) $qed$
]

== The geometry

// Drawing geometry in points. `y` is measured upward from the baseline and
// flipped at draw time. One whole turn is `turn` tall and candidate 0 gains
// `rise` per echo spacing, so the drawn k is te0 / dte-x.
#let pencil = {
  let te0 = 56.0
  let dte-x = 93.0
  let turn = 34.0
  let rise = 20.0
  let ypiv = 62.0
  let tend = 265.0
  let ox = 34.0
  let oy = 24.0
  let hbox = 224.0

  let echoes = (te0, te0 + dte-x, te0 + 2.0 * dte-x)

  let px(t) = (ox + t) * 1pt
  let py(y) = (hbox - oy - y) * 1pt
  let val(n, t) = ypiv + (rise + n * turn) / dte-x * (t - te0)

  let thin = (paint: luma(50), thickness: 0.9pt)
  let faint = (paint: luma(170), thickness: 0.4pt, dash: "dashed")
  let hair = (paint: luma(105), thickness: 0.4pt)

  let dot(t, y, filled) = place(
    dx: px(t) - 1.8pt,
    dy: py(y) - 1.8pt,
    circle(
      radius: 1.8pt,
      fill: if filled { black } else { white },
      stroke: 0.4pt + black,
    ),
  )

  // dimension line with end caps; `side` puts the label left (-1) or right (+1)
  let dim(t, ya, yb, label, side) = {
    place(line(start: (px(t), py(ya)), end: (px(t), py(yb)), stroke: hair))
    for y in (ya, yb) {
      place(line(
        start: (px(t) - 2pt, py(y)),
        end: (px(t) + 2pt, py(y)),
        stroke: hair,
      ))
    }
    place(
      dx: px(t) + (if side > 0 { 4pt } else { -26pt }),
      dy: py((ya + yb) / 2.0) - 5.5pt,
      box(
        width: 22pt,
        align(if side > 0 { left } else { right }, text(size: 8.5pt, label)),
      ),
    )
  }

  box(width: (ox + tend + 46.0) * 1pt, height: hbox * 1pt, {
    // baseline, and time guides drawn only up to the lowest candidate so they
    // never collide with the dimension lines
    place(line(
      start: (px(-8.0), py(6.0)),
      end: (px(tend + 6.0), py(6.0)),
      stroke: hair,
    ))
    for (t, lab) in (
      (0.0, $0$),
      (echoes.at(0), $"TE"_0$),
      (echoes.at(1), $"TE"_1$),
      (echoes.at(2), $"TE"_2$),
    ) {
      let low = calc.min(val(1, t), val(0, t), val(-1, t))
      place(line(start: (px(t), py(6.0)), end: (px(t), py(low - 6.0)), stroke: faint))
      place(
        dx: px(t) - 11pt,
        dy: py(-4.0),
        box(width: 22pt, align(center, text(size: 8.5pt, lab))),
      )
    }

    // the three candidate lines, labelled at the right
    for n in (1, 0, -1) {
      place(line(
        start: (px(0.0), py(val(n, 0.0))),
        end: (px(tend), py(val(n, tend))),
        stroke: thin,
      ))
      place(
        dx: px(tend) + 4pt,
        dy: py(val(n, tend)) - 5.5pt,
        text(size: 8.5pt, $N = #if n > 0 [$+1$] else if n == 0 [$0$] else [$-1$]$),
      )
    }

    // offsets at t = 0, the shared pivot, and the samples at later echoes
    for n in (1, 0, -1) {
      dot(0.0, val(n, 0.0), false)
      dot(echoes.at(1), val(n, echoes.at(1)), false)
      dot(echoes.at(2), val(n, echoes.at(2)), false)
    }
    dot(te0, ypiv, true)

    // the two step sizes, read straight off the picture
    dim(0.0, val(0, 0.0), val(1, 0.0), $2 pi k$, -1)
    dim(echoes.at(1), val(0, echoes.at(1)), val(1, echoes.at(1)), $2 pi$, 1)
    dim(echoes.at(2), val(0, echoes.at(2)), val(1, echoes.at(2)), $4 pi$, 1)

    // leader from an annotation in the empty upper left down to the pivot
    place(line(
      start: (px(te0 - 9.0), py(140.0)),
      end: (px(te0 + 1.0), py(68.0)),
      stroke: hair,
    ))
    place(dx: px(-6.0), dy: py(156.0), text(size: 8.5pt)[every candidate meets here])
    place(dx: px(-6.0), dy: py(202.0), text(size: 8.5pt)[unwrapped phase])
  })
}

#figure(pencil, caption: [
  Candidates as lines in phase against echo time, drawn for $k = 0.6$. All of
  them meet at $"TE"_0$, so the offset step $2 pi k$ is exactly the phase a
  one-wrap steeper slope gains by the first echo. Past $"TE"_0$ the lines
  separate by one further turn per echo spacing, so at echo $e$ they stand whole
  turns apart (open circles) and wrap to identical samples.
]) <fig:pencil>

#v(0.6em)

@fig:pencil is the same statement drawn. Against echo time each candidate is a
straight line, intercept $hat(theta)_N$ and slope $2 pi hat(f)_N$. Setting
$t = "TE"_0$ in the last line of the proof kills the parenthesis, so every
candidate takes the value $hat(theta)_0 + 2 pi hat(f)_0 "TE"_0$ at the first echo:
the candidates form a pencil of lines through one point. Past that point they
separate at one turn per echo spacing, standing $2 pi N e$ apart at $t_e$.
Wrapped phase cannot see a whole turn, so every line reproduces the same measured
samples.

The two step sizes are consequences rather than choices. Requiring two candidates
to differ by whole turns at every echo forces their slopes to differ by one turn
per echo spacing, which is the field step $1 slash #dte$; that fixed, their
offsets must differ by the $2 pi k$ such a slope has already gained at $"TE"_0$,
which is exactly what makes the two lines meet there.

== Consequences

Three consequences follow, and the rest of the note rests on all of them.

*The rule is unavoidable.* Since the candidates predict identical data, no
statistic computed from the phase can rank them. @eq:half is the only kind of
answer available, and any implementation must supply something like it.

*A longer echo train does not help.* The cancellation holds at every echo at once,
so a five-echo acquisition constrains $N$ no better than a two-echo one. The later
echoes are not independent measurements of the ambiguity. This depends on the even
spacing of @eq:spacing rather than on phase imaging in general, but an EPI train is
evenly spaced by construction.

*The undetectable errors are exactly enumerated.* #link(<prop:alias>)[Prop. 1] applies to a
candidate as defined in @eq:offn and @eq:fldn, an offset paired with the field of
the same $N$. Nothing outside that set is protected, which is what makes any
detection possible at all, and @sec:detect is about exploiting it.
