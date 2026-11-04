# Spurious-edge arm (coupled world), state regime, h = 1

| variant | n | jac24 | jac21 | jac23 | gate24 | gate_edges | o2_id | o2_dec | ratio | all_id | all_dec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 3 | 0.089 ± 0.065 | 0.295 ± 0.013 | 0.255 ± 0.074 | – | – | 0.292 ± 0.026 | 0.302 ± 0.039 | 1.032 ± 0.042 | 0.335 ± 0.030 | 0.339 ± 0.033 |
| hard | 3 | 0.000 ± 0.000 | 0.297 ± 0.035 | 0.109 ± 0.095 | 0.194 ± 0.021 | 2.667 ± 0.577 | 0.343 ± 0.048 | 0.348 ± 0.047 | 1.013 ± 0.011 | 0.367 ± 0.010 | 0.366 ± 0.016 |
| oracle | 3 | 0.000 ± 0.000 | 0.320 ± 0.042 | 0.263 ± 0.069 | – | – | 0.299 ± 0.034 | 0.316 ± 0.041 | 1.058 ± 0.025 | 0.352 ± 0.030 | 0.359 ± 0.021 |

jac2j = normalized Jacobian ∂ẑ2/∂z_j (true parents of 2: objects 1 and 3; 4 is the spurious predictor).
o2_* = object-2 decoded error / copy baseline; all_* = mean over objects 0–3. ratio = o2_dec / o2_id.
