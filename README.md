# scorecard-quantization

Code for *Integer-point rounding of additive scorecards: grade migration is first
order, AUC change is second order, and offset-aware rounding reduces migration*
(H. Kang, Y. Yang, M. Yang; submitted to Computational Statistics and Data Analysis).

## Contents

| Path | Purpose |
|---|---|
| `sqlib.py` | Library: WoE scorecard construction, lattice rounding, Irwin–Hall laws, exact-convolution and closed-form predictors, AUC-gradient fluctuation, residual-randomization Monte Carlo, diagnostics, offset-aware rounding (OAR), cutoff recalibration |
| `exp/e1_synthetic_accuracy.py` | Factorial simulation, 360 scorecards (Table 1, Fig. 1) |
| `exp/e2_coverage_stationarity.py` | Prediction-interval coverage and stationarity of the AUC (Table 2, Fig. 2a–b) |
| `exp/e2c_gradient_check.py` | AUC fluctuation versus sample size and the gradient formula (Fig. 2c) |
| `exp/e3_rounding_synthetic.py` | Rounding algorithms with train/test split (Table 3) |
| `exp/e4_real_data.py` | Four credit data sets: lattices, predictions, intervals, algorithms, bootstrap refits (Tables 4–5, Figs. 3–4) |
| `exp/e4b_test_intervals.py` | Intervals computed on the evaluation sample (Table 2) |
| `exp/make_tables.py`, `exp/make_figures.py` | LaTeX tables, figures and `results/numbers.json` with every number quoted in the text |

## Reproduce

```bash
pip install -r requirements.txt
bash fetch_data.sh      # downloads the public data into ./data
bash run_all.sh         # runs all experiments, then writes tables/ and figures/
```

All scripts are run from the repository root. Seeds are fixed in every script.

## Minimal use

```python
import numpy as np
from sqlib import Scorecard, IrwinHall, Predictor, OffsetRounding, cutoffs, delta_from_pdo

sc = Scorecard(n_bins=8).fit(X_train, y_train)          # pandas DataFrame, 0/1 target
cells = sc.cells(X_train); f = sc.score(cells)
D = delta_from_pdo(pdo=20, grid=5)                        # lattice spacing
cuts = cutoffs(f, K=10)                                   # equal-frequency grades
pred = Predictor(f, y_train, cells, IrwinHall(2 * sc.P + 2))
print(pred.migration_ec(D, cuts), pred.dauc_ec(D))        # expected migration and AUC change
oar = OffsetRounding(D).fit(sc.cvec, cells, sc.b0, f, cuts)
feat = np.repeat(np.arange(sc.P), sc.sizes)
points = -5 * oar.points(sc.cvec, feat)                   # integer points table (grid g = 5)
```

## Data

Taiwan (Yeh and Lien, 2009), Give Me Some Credit (Kaggle, 2011), FICO HELOC
(FICO, 2018) and Statlog German credit (Hofmann, 1994). The data are not
redistributed here; `fetch_data.sh` downloads them from the listed sources.
Sex, marital status, age and foreign-worker status are excluded from the models.

## Licence

MIT
