# Data

## Raw EEG Data

Raw EEG files are **not included** in this repository due to size. Download directly from OpenNeuro:

- **Discovery cohort (eyes-closed)**: https://openneuro.org/datasets/ds004504
  - Miltiadous A, et al. (2023). A dataset of EEG recordings from Alzheimer's disease, Frontotemporal dementia, and Healthy subjects. *Scientific Data*, 10, 862.
  
- **Cross-condition (eyes-open)**: https://openneuro.org/datasets/ds006036
  - Same subjects as ds004504, photic stimulation paradigm.

- **Preclinical validation**: https://openneuro.org/datasets/ds007427
  - Henao Isaza V, et al. (2025). Comprehensive methodology for sample enrichment in EEG biomarker studies for Alzheimer's risk classification. *PLOS ONE*.

## Derived Data (included)

- `discovery_ds004504/rho_gradient_results.csv` — Per-subject ρ values for all frequency bands and channel groups, plus DV gradients.
- `preclinical_ds007427/rho_preclinical_results.csv` — Per-subject ρ values for PSEN1 carriers and non-carriers (add after analysis completes).

## Reproducing

1. Download the relevant dataset from OpenNeuro
2. Run: `python code/discovery/rho_analysis.py /path/to/ds004504`
3. Or: `python code/preclinical/rho_preclinical.py /path/to/ds007427`
