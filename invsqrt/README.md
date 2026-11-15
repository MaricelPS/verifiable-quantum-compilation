# invsqrt: phasing by 1/√x with a table-driven polynomial

Benchmark I of the competition. The circuit receives an integer x in a 22-qubit register, with
1 ≤ x ≤ 3,133,452, and must apply the phase e^{2πi/√x} with an error below 2⁻²⁴/√x, then return every ancilla to
|0⟩. The score is Q × C: logical qubits times a Toffoli-equivalent cost (Toffolis plus the T gates of the phase
rotations, priced by the precision they need).

| | Reference (Newton–Raphson) | This work |
|---|---|---|
| Qubits Q | 260 | **155** |
| Toffolis | 7,735 | **2,543** |
| Phase rotations | 37 | 33 |
| Cost C (Toffoli-equivalent) | 7,987.6 | **2,783.5** |
| Score Q × C | 2,076,774 | **431,442** |
| Improvement | ×1.00 | **×4.81** |
| Worst error / tolerance, all x | 0.48 | 0.95 |

Both columns come from the official verifier. [`verify_full.json`](verify_full.json) is the full run: all
200,540,928 shots (64 per value of x), PASS.

## Design

The reference computes 1/√x with Newton–Raphson iterations: wide multiplications that dominate both qubits and
Toffolis. This design trades them for a lookup table and two short multiplications.

1. **Normalize.** Shift x left bit by bit until its top bits are occupied (5 binary shift stages, each a zero test
   and a controlled rotation). Then M = x_norm / 2²⁰ lies in [1, 4) and 1/√x = 2^(s−10) · M^(−1/2). The parity of the
   shift picks one half of the coefficient table, which is what lets the shift go one bit at a time.
2. **Look up.** The top 8 bits of x_norm address a 256-entry table of piecewise-quadratic coefficients for M^(−1/2).
   The table is loaded with a QROM read and later uncomputed by measurement, which is much cheaper than
   running the read backwards.
3. **Horner, in subtract form.** y = a₀ − δ(a₁ − δ·a₂), where δ is the offset of M inside its segment. Partial
   products are truncated below each register's last useful bit (*widths* 28, 21, 12 bits).
4. **Denormalize and phase.** Shift y back by s and apply the phase bit by bit (Z rotations; the 2 lowest bits are
   left out because they are below the tolerance).
5. **Uncompute** steps 4 to 1.

`invsqrt_poly.model` reproduces the circuit's integer arithmetic exactly, so the worst error over all 3.1 million
inputs is known *before* building or simulating anything. That is what makes the search cheap: `optimize.py` runs a
greedy descent over widths, δ bits, bias and dropped phase bits, keeping only designs whose worst error stays below
0.985 of the tolerance, and builds the circuit only to count qubits and gates.

## Files

| File | Role |
|---|---|
| `invsqrt_poly.py` | `Design` parameters, exact classical model, circuit builder, cost estimate |
| `build.py` | Builds the submitted design and writes `invsqrt.kmb` (sha256 `a3ad364f…`, identical to the submission) |
| `optimize.py` | Greedy descent over the design parameters |
| `stage_costs.py`, `tune_precision.py` | First explorations: cost per stage of the reference circuit and its register widths |
| `kaggle_notebook.ipynb` | The Kaggle notebook that put this circuit into a submission next to the chemistry result |
| `verify_full.json` | Full official verification of the submitted circuit |

```bash
export COMP_DIR=/path/to/competition-data      # folder with kickmix/ and verifiers/
python invsqrt/build.py invsqrt.kmb
python "$COMP_DIR/verifiers/invsqrt/verify_submission_invsqrt.py" invsqrt.kmb
```

## What did not work

- **Controlled adders instead of AND rows** in the Horner step: more Toffolis.
- **A shorter scratch register** for the products: also more Toffolis.
- **Normalizing by pairs of bits**, as the reference does. The QROM read needs a complete 2ᵏ-entry table, and
  part of it stays unused. Normalizing by single bits, with the parity of the shift selecting one half of the
  table, halves the table instead.

The best public circuit reaches Q = 62 and C = 1,514. Getting there would take a deeper redesign of the
normalization and of the multiplications, not further tuning of this one.
