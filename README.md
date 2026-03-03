# pH_cortical_dynamics

**Port-Hamiltonian Framework for Cortical Dynamics in Alzheimer's Disease**

This repository contains code, data, and figures for:

> **A port-Hamiltonian framework reveals dorsoventral gradient disruption as an early signature of Alzheimer's disease**
>
> Arash Salardini
>
> Glenn Biggs Institute for Alzheimer's & Neurodegenerative Diseases, UT Health San Antonio

Target journal: *Proceedings of the National Academy of Sciences (PNAS)*

---

## Overview

We apply port-Hamiltonian systems theory to model cortical dynamics measured by EEG. The AR(2) eigenvalue modulus (ρ) captures each brain region's position on the stability–oscillation continuum, which in the port-Hamiltonian framework reflects the balance between energy storage and dissipation.

### Key Findings

1. **Theoretical framework**: Port-Hamiltonian formulation of cortical E-I circuits predicts that ρ indexes the dissipation rate, with higher ρ reflecting less dissipative (more oscillatory) dynamics.

2. **Toy model validation**: A biologically realistic E-I circuit model confirms near-perfect mapping between ρ and dissipation (r = −0.998).

3. **Discovery cohort** (ds004504, Greece, N=88): AD patients show flattened dorsoventral ρ gradient compared to controls (alpha band: Cohen's d = 1.175, p < 0.0001). Healthy controls replicate the MEG frequency-specific gradient: alpha/beta ventral-dominant, theta dorsal-dominant.

4. **Cross-condition replication** (ds006036, same subjects, eyes-open photic stimulation): Effect attenuates but persists (alpha d = 0.610, p = 0.019), with beta remaining stable (d = 0.807).

5. **Dose-response**: DV gradient tracks cognitive severity across the full sample (ρₛ = −0.434, p < 0.0001 vs MMSE). Effect is between-group (AD vs controls) rather than within-group.

6. **Preclinical validation** (ds007427, Colombia, N=43): Presymptomatic PSEN1 E280A carriers show globally elevated delta-band ρ (d = 0.92, p = 0.006) compared to non-carrier family members, with trending reductions in beta/broadband ρ (d ≈ −0.58, p ≈ 0.06). The DV gradient is intact in both groups — gradient disruption emerges only with clinical disease, while global dissipative imbalance precedes it by ~15–20 years.

---

## Repository Structure

```
pH_cortical_dynamics/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── code/
│   ├── toymodel/
│   │   └── ph_ei_toymodel.py            # E-I circuit port-Hamiltonian simulation
│   ├── discovery/
│   │   └── rho_analysis.py              # Main analysis: ds004504 (eyes-closed)
│   ├── preclinical/
│   │   └── rho_preclinical.py           # Preclinical validation: ds007427
│   └── plotting/
│       ├── plot_results.py              # Discovery cohort figures
│       └── dose_response.py             # MMSE dose-response figures
│
├── data/
│   ├── discovery_ds004504/
│   │   └── rho_gradient_results.csv     # Per-subject ρ values, all 88 subjects
│   ├── preclinical_ds007427/
│   │   └── preclinical_summary_stats.csv  # Summary statistics
│   └── README.md                        # Data provenance & access instructions
│
├── figures/
│   ├── main/
│   │   ├── Fig1_pH_ToyModel.png         # E-I toy model: ρ vs dissipation
│   │   ├── dose_response_mmse.png/pdf   # MMSE dose-response
│   │   └── Fig_preclinical_validation.png/pdf  # PSEN1 carrier results
│   └── supplementary/
│       ├── Fig_Validation_AR2.png       # AR(2) model validation
│       └── eeg_rho_analysis.png         # EEG analysis overview
│
├── manuscript/
│   └── PNAS_PortHamiltonian_Draft.docx
│
└── notebooks/
    └── rho_AD_analysis_colab.ipynb      # Google Colab notebook
```

---

## Datasets

All EEG data are publicly available from OpenNeuro:

| Dataset | OpenNeuro ID | Description | N | Channels | Fs |
|---------|-------------|-------------|---|----------|-----|
| Discovery (eyes-closed) | [ds004504](https://openneuro.org/datasets/ds004504) | AD/FTD/Control resting EEG | 88 | 19 (10-20) | 500 Hz |
| Cross-condition (eyes-open) | [ds006036](https://openneuro.org/datasets/ds006036) | Same subjects, photic stimulation | 88 | 19 (10-20) | 500 Hz |
| Preclinical validation | [ds007427](https://openneuro.org/datasets/ds007427) | PSEN1 E280A carriers vs non-carriers | 44 | 57 (10-10) | 1000 Hz |

**Citation for datasets:**
- Miltiadous et al. (2023). *Sci Data*, 10, 862. (ds004504/ds006036)
- Henao Isaza et al. (2025). *PLOS ONE*. (ds007427)

---

## Quick Start

### Requirements

```bash
pip install numpy pandas scipy mne matplotlib seaborn
```

### Run Discovery Analysis

```bash
# Download ds004504 from OpenNeuro, then:
cd /path/to/ds004504
python rho_analysis.py
python plot_results.py
```

### Run Preclinical Analysis

```bash
# Download ds007427 from OpenNeuro, then:
cd /path/to/ds007427
python rho_preclinical.py
```

---

## Methods Summary

### AR(2) Eigenvalue Modulus (ρ)

For each EEG channel and frequency band, we fit a second-order autoregressive model:

```
x(t) = a₁·x(t-1) + a₂·x(t-2) + ε(t)
```

The eigenvalue modulus ρ = max(|λ₁|, |λ₂|) of the companion matrix captures the dominant dynamical mode:
- ρ → 1: oscillatory, low dissipation
- ρ → 0: rapidly decaying, high dissipation

We compute ρ over sliding 4-second epochs (50% overlap) and take the median, applied to bandpass-filtered signals (delta 1–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz, broadband 1–45 Hz).

### Dorsoventral Gradient

Channels are grouped by dorsoventral position:
- **Dorsal**: C3, Cz, C4 (19-ch) / C1-C6, FC1-FC2, FCZ (57-ch)
- **Ventral temporal**: T5, T6 (19-ch) / TP7, TP8, CP5, CP6 (57-ch)

The DV gradient = ρ(dorsal) − ρ(ventral temporal).

### Port-Hamiltonian Framework

The cortical E-I circuit is modeled as:

```
ẋ = (J - R)∇H(x) + Bu
```

where J captures energy-conserving oscillatory dynamics, R captures dissipation, and ρ maps monotonically onto the dissipation rate (R matrix eigenvalues).

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Contact

Arash Salardini, MD
Division of Cognitive and Behavioral Neurology
Glenn Biggs Institute for Alzheimer's & Neurodegenerative Diseases
UT Health San Antonio
