# Run 30 Surgical Optimization Report

## Slice Results

| Slice | Count | wA | wB | wT | BA (Final) |
|---|---|---|---|---|---|
| default | 7574 | 0.27 | 0.0 | 0.73 | 0.6115 |
| jpgjpeg | 7267 | 0.27 | 0.0 | 0.73 | 0.5985 |
| png | 301 | 0.18 | 0.12 | 0.7 | 0.7967 |
| thumbnail | 5551 | 0.25 | 0.01 | 0.74 | 0.5443 |
| low_res | 0 | 0.33 | 0.33 | 0.34 | 0.5000 |
| 10k-50k | 2 | 0.0 | 0.0 | 1.0 | 0.5000 |
| 50k-500k | 627 | 0.15 | 0.01 | 0.84 | 0.7504 |
| >500k | 1394 | 0.28 | 0.01 | 0.71 | 0.7861 |
| squareish | 6010 | 0.24 | 0.0 | 0.76 | 0.5695 |
| portrait_tall | 182 | 0.25 | 0.01 | 0.74 | 0.7823 |

## Impact Analysis (vs Run 24 Baseline)
- **Rescued Errors**: 548 images previously wrong are now correct.
- **Poisoned Successes**: 543 images previously correct are now wrong.
- **Net Change**: +5 images.

See `sample_rescued.csv` for details on rescued images.

See `sample_poisoned.csv` for details on poisoned images.
