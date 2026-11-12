#!/usr/bin/env bash
# Reproduces every table and figure of the manuscript (about 30 minutes on 2 cores).
set -euo pipefail
cd "$(dirname "$0")"
[ -f data/taiwan.csv ] || bash fetch_data.sh
python exp/e1_synthetic_accuracy.py
python exp/e2_coverage_stationarity.py
python exp/e2c_gradient_check.py
python exp/e3_rounding_synthetic.py
for d in German HELOC Taiwan GMSC; do python exp/e4_real_data.py $d; done
python exp/e4b_test_intervals.py
python exp/make_tables.py
python exp/make_figures.py
