# Results

## Training

Loss = MSE(q) + α_pos · pos_err + α_rot · rot_err, evaluated on all three splits:

| Split      | Total loss | Pos error (m) | Rot error (Frobenius) |
|------------|-----------|---------------|----------------------|
| Train      | 0.00944   | 0.00475       | 0.0469               |
| Validation | 0.00971   | 0.00488       | 0.0482               |
| Test       | 0.00977   | 0.00491       | 0.0486               |

Train/val/test gaps are small — no significant overfitting.

---

## MLP warm start vs NR cold start (random targets)

Cold start: q_init = zeros. Warm start: q_init = MLP prediction.

| Method     | Mean iterations | Converged |
|------------|----------------|-----------|
| NR cold    | 4.51           | 99.78%    |
| NR warm    | 1.87           | 100.00%   |

---

## Large jumps — pick-and-place cycle (A → home → B)

The hardest scenario: the robot passes through home between two random configs, making q_prev a poor initialisation for the next target.

| Method         | Mean iterations | Converged | Mean pos err (m) | Max pos err (m) |
|----------------|----------------|-----------|-----------------|----------------|
| NR cold (q_prev) | 12.17        | 99.3%     | 0.0042          | 1.0504         |
| NR warm (MLP)  | 1.98           | 100.0%    | 0.0001          | 0.0011         |

**6× fewer iterations on average. Max positional error reduced from 1.05 m to 0.001 m.**  
This is where the MLP warm start is most valuable.

---

## MLP alone (no NR refinement)

| Metric              | Value   |
|---------------------|---------|
| Mean pos error      | 1.18 cm |
| Median pos error    | 0.64 cm |
| Max pos error       | 34.27 cm |
| Mean rot error      | 5.17°   |
| Median rot error    | 2.96°   |

The median position error (6.4 mm) is within the manufacturer's stated accuracy of 5–8 mm. Rotation error is too large for standalone use — NR refinement is needed for full precision.

---

For full methodology see [learned_warmstart_for_nr_ik_viperx300.docx](learned_warmstart_for_nr_ik_viperx300.docx).
