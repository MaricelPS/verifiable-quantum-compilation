# dqi: a cheaper Reed–Solomon decoder for Decoded Quantum Interferometry

Benchmark II of the competition. Decoded Quantum Interferometry (DQI) needs, inside the quantum computer, a decoder
for a Reed–Solomon code over GF(2¹²) with n = 70 syndromes and m = 4,095 positions. The circuit receives the
syndromes in 70 quantum registers and must:

- multiply the amplitude by (−1)^(Σⱼ e_j · M_j), where e is the decoded error and M_j are classical measurement
  outcomes (*measurement-based uncomputation*, MBUC);
- restore the syndromes and return every ancilla to |0⟩.

The verifier simulates 102,400 random shots. The score is R · Q² · T: a repetition factor R that grows with the
error rate (R = 1.19 with zero errors, because of a 2⁻¹²⁸-confidence bound), qubits squared, and Toffolis per shot.

| | Reference decoder | This work |
|---|---|---|
| Qubits Q | 1,893 | **1,867** |
| Toffolis per shot T | 5,725,108 | **4,146,652** |
| Errors in 102,400 shots | 0 | 0 |
| Score R · Q² · T | 2.444 × 10¹³ | **1.722 × 10¹³** |
| Improvement | ×1.000 | **×1.419** |

Full official verification: [`results/verify_reference.json`](results/verify_reference.json),
[`results/verify_tower.json`](results/verify_tower.json). The circuit built from this folder is byte-identical to
the submitted one (sha256 `ff4a44db…`).

## Where the cost was

The reference decoder (organizers, after Proos and Zalka) runs a reversible *synchronized* extended Euclidean
algorithm (EEA) to find the error-locator polynomial σ and the evaluator ω, evaluates them at all 4,095 positions
(Chien search) to find the error values (Forney's formula), kicks back the phase, and runs the EEA backwards.
Profiling by ablation (`profile_stages.py`: rebuild with one stage removed, subtract) showed:

![Toffoli breakdown](../figures/dqi_toffoli_breakdown.png)

Two numbers stood out:

- **Forney** spends 315 of its ~350 Toffolis per position on a single GF(2¹²) inversion, σ'(x)⁻¹, and does it at all
  4,095 positions.
- **The EEA sweep** does 73 controlled GF(2¹²) multiplications per iteration (75 Toffolis each) for 444 iterations,
  plus two divisions per iteration (402 Toffolis each).

Both are field arithmetic, so that is where I attacked.

## The idea: tower-field arithmetic as AND networks

GF(2¹²) can be built in layers: GF(2³) → GF(2⁶) = GF(2³)[v]/(v² + v + λ) → GF(2¹²) = GF(2⁶)[w]/(w² + w + μ).
In that representation:

- **Multiplication** is Karatsuba at every layer: 3 × 3 × 6 = **54 ANDs** (kickmix's built-in: 63 Toffolis).
- **Inversion** reduces to the subfield: x⁻¹ = conj(x) · N(x)⁻¹, where the norm N(x) = a² + ab + μb² lives in
  GF(2⁶). Recursing once more, the only inverse left is in GF(2³), where x⁻¹ = x² · x⁴ is a single product of two
  linear functions. Total: **78 ANDs** (built-in: 315 Toffolis).

Every nonlinear step is a *bilinear map*: an AND of a linear form of one register with a linear form of another,
XORed into the output. That gives two properties that the cost model rewards:

1. **One Toffoli per AND, nothing else.** The linear forms are prepared with CNOTs in place (free) and undone after.
   The basis change between kickmix's polynomial basis and the tower basis is a 12 × 12 GF(2) matrix, folded into
   the forms, so registers never need converting.
2. **Free uncomputation.** A register holding a bilinear form can be cleared by measuring it in the X basis and
   fixing the phase with CZ gates conditioned on the outcomes. No Toffolis, so the cost of a temporary value is only
   the cost of computing it once.

On top of that, two tricks specific to the decoder:

- **Gate the inverse, not the result.** Forney only needs σ'⁻¹ where σ(x) = 0. The reference computes the inverse and
  then ANDs ω with the zero flag (12 Toffolis). Here the flag ANDs the 3-bit GF(2³) norm (3 Toffolis); a zero norm
  propagates to a zero inverse, so the phase is trivial wherever σ(x) ≠ 0. The same gate replaces the control on the
  EEA quotient divisions.
- **Zero test by AND tree, uncomputed by measurement.** 11 Toffolis per position instead of 2 × 12.

| Per call | Reference | This work |
|---|---|---|
| Forney, per position | ≈ 353 | ≈ 94 |
| EEA quotient b/a | 402 | ≈ 135 |
| EEA controlled multiply | 75 | 66 (12 to gate q, 54 to multiply) |

Everything else (the EEA control flow, the array shifts and swaps, the sub-step schedule) is the reference's.

## Testing

The tower circuits are simulated with kickmix and checked for value, clean ancillas and no stray phase:

- multiplication on 4,096 random triples (out ^= x · y), exactly 54 Toffolis;
- inversion on **all** 4,096 inputs, gated and ungated, exactly 78 and 81 Toffolis;
- the basis change is a field isomorphism (2,000 random products);
- the full decoder passes the official verifier on a small instance (m = 300, n = 10).

```bash
export COMP_DIR=/path/to/competition-data      # folder with kickmix/ and verifiers/
python -m pytest dqi/tests -q                   # 5 tests, a few seconds
python dqi/decoder.py dqi.kmb                   # full circuit, ~1 minute, 773 MB
python "$COMP_DIR/verifiers/dqi/verify_submission_dqi.py" dqi.kmb
```

The official verifier also reports `reaction_depth`, which needs more than 7 GB of RAM for a circuit of this size.
It is not part of the score; on small machines run a copy of the verifier with that one line removed.

## What is left, and why I did not do it yet

The EEA multiply sweep is now 65% of the Toffolis. The synchronized EEA cycles through 4 sub-steps; every
iteration pays for all of them, quantum-controlled, although only one is active:

- sub-steps 0 and 3 do the same 73-position multiply sweep (on the remainder and on the cofactor halves of the
  packed arrays), about 72 iterations each;
- sub-steps 1 and 2 only shift, about 40 and 32 iterations.

`schedule_analysis.py` records the bookkeeping registers after every iteration
([`results/schedule_analysis.txt`](results/schedule_analysis.txt)). The schedule is nearly identical for all inputs
early on, which suggests skipping sub-steps by iteration number. But the shots drift off the common schedule at
roughly one in 4,000 per division round (about the chance that a GF(2¹²) coefficient is zero): 0.7% of shots are off
the majority sub-step by iteration 130, 2% by iteration 150. The score's confidence bound makes even 0.1% of failed shots expensive (R rises ~10% per 10⁻³
error rate), so the schedule cannot be guessed. It has to be changed.

The promising redesign is to fuse sub-steps 0 and 3: apply each quotient term to the remainder and the cofactor in
the same sweep, as the textbook EEA does. If the packed arrays can keep both alignments, about a third of all
iterations disappear, an estimated 25–30% of T.

## Files

| File | Role |
|---|---|
| `tower.py` | Tower field: classical arithmetic, basis change, bilinear term lists, quantum emitters, `TowerInverse` |
| `decoder.py` | `TowerDecoder`, subclass of the reference decoder; builds `dqi.kmb` |
| `tests/` | Simulation tests of the tower circuits and a small end-to-end verification |
| `profile_stages.py` | Toffolis per stage by ablation, for both decoders (`results/stage_profile.csv`) |
| `schedule_analysis.py` | EEA sub-step schedule across random inputs |
| `kaggle_notebook.ipynb` | The Kaggle notebook that built, verified and packaged the submission (sources embedded) |
| `results/` | Official verifier outputs and analysis tables |
