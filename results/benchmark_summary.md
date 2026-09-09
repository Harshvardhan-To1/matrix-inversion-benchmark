# Matrix inversion benchmark

- Host: x86_64, 4 torch threads, CPU only
- Dataset: random well-conditioned matrices `A = G/sqrt(n) + 2I`, float32
- Timing: per matrix (batch size 1), warmup 5, held-out split
- `residual` = mean of `||X @ A - I||_F / sqrt(n)`; `rel err` = mean of `||X - A^-1||_F / ||A^-1||_F`

## Dimension 10 (200 samples)

| Method | Mean (ms) | Median (ms) | P95 (ms) | Residual | Rel err |
|---|---:|---:|---:|---:|---:|
| LAPACK getri (numpy.linalg.inv) | 0.005 | 0.005 | 0.006 | 3.32e-08 | 4.76e-08 |
| QR decomposition | 0.014 | 0.014 | 0.017 | 1.88e-07 | 1.81e-07 |
| LU decomposition (scipy lu_factor/lu_solve) | 0.015 | 0.015 | 0.018 | 8.78e-08 | 8.55e-08 |
| SVD | 0.022 | 0.021 | 0.027 | 8.19e-08 | 8.11e-08 |
| Newton-Schulz iteration | 0.039 | 0.037 | 0.044 | 1.08e-07 | 9.25e-08 |
| Gauss-Jordan elimination | 0.045 | 0.044 | 0.046 | 1.07e-07 | 1.01e-07 |
| InverseNet-MLP | 0.052 | 0.051 | 0.063 | 4.16e-01 | 3.57e-01 |
| InverseNet-NS (learned) | 0.119 | 0.117 | 0.129 | 4.09e-04 | 5.03e-04 |

## Dimension 100 (200 samples)

| Method | Mean (ms) | Median (ms) | P95 (ms) | Residual | Rel err |
|---|---:|---:|---:|---:|---:|
| LU decomposition (scipy lu_factor/lu_solve) | 0.094 | 0.092 | 0.103 | 1.60e-07 | 1.38e-07 |
| QR decomposition | 0.262 | 0.260 | 0.280 | 3.53e-07 | 3.26e-07 |
| InverseNet-NS (learned) | 0.364 | 0.361 | 0.405 | 3.60e-04 | 4.03e-04 |
| Newton-Schulz iteration | 0.405 | 0.405 | 0.430 | 2.75e-07 | 2.17e-07 |
| LAPACK getri (numpy.linalg.inv) | 0.547 | 0.128 | 0.162 | 3.39e-08 | 4.81e-08 |
| InverseNet-MLP | 0.779 | 0.767 | 0.824 | 4.94e-01 | 5.27e-01 |
| SVD | 1.207 | 1.197 | 1.282 | 1.80e-07 | 1.48e-07 |
| Gauss-Jordan elimination | 1.412 | 1.412 | 1.476 | 3.39e-07 | 3.04e-07 |

## Dimension 500 (200 samples)

| Method | Mean (ms) | Median (ms) | P95 (ms) | Residual | Rel err |
|---|---:|---:|---:|---:|---:|
| LAPACK getri (numpy.linalg.inv) | 5.640 | 5.635 | 5.782 | 3.43e-08 | 4.84e-08 |
| Newton-Schulz iteration | 18.034 | 18.161 | 18.709 | 4.67e-07 | 3.64e-07 |
| LU decomposition (scipy lu_factor/lu_solve) | 25.512 | 4.227 | 110.047 | 2.63e-07 | 2.16e-07 |
| SVD | 36.570 | 36.586 | 37.244 | 3.23e-07 | 2.55e-07 |
| QR decomposition | 51.854 | 12.154 | 113.639 | 4.00e-07 | 3.56e-07 |
| InverseNet-NS (learned) | 114.846 | 116.980 | 122.144 | 4.82e-03 | 7.29e-03 |
| Gauss-Jordan elimination | 187.505 | 187.501 | 189.323 | 7.71e-07 | 6.88e-07 |
