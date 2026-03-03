"""
Preclinical Validation: Rho Gradient in Presymptomatic PSEN1 Carriers
=====================================================================
Port-Hamiltonian Prediction: Dorsoventral rho gradient is already
disrupted in presymptomatic PSEN1 E280A carriers vs non-carrier
family members, decades before symptom onset.

Dataset: OpenNeuro ds007427 (Henao Isaza et al., PLOS ONE 2025)
  Colombian PSEN1 E280A kindred (Antioquia)
  G1 = presymptomatic PSEN1 carriers (~25 subjects with EEG)
  G2 = non-carrier family controls (~18 subjects with EEG)
  57-channel resting EEG (eyes closed), 1000 Hz, BrainVision format

Usage:
  cd C:\\Users\\salardini\\Documents\\EEG3
  python rho_preclinical.py

  Or specify path:
  python rho_preclinical.py C:\\Users\\salardini\\Documents\\EEG3
"""

import sys
import os
import numpy as np
import pandas as pd
import mne
from scipy import signal
from scipy.stats import ttest_ind, spearmanr, mannwhitneyu
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# ============================================================
# CONFIGURATION — 10-10 extended montage
# ============================================================

# With 57 channels, we can define a richer dorsoventral gradient.
# All channel names are UPPERCASE in this dataset.
CHANNEL_GROUPS = {
    'dorsal':           ['CZ', 'C1', 'C2', 'C3', 'C4', 'FCZ', 'FC1', 'FC2'],
    'centroparietal':   ['CPZ', 'CP1', 'CP2', 'CP3', 'CP4'],
    'parietal':         ['PZ', 'P1', 'P2', 'P3', 'P4'],
    'frontal':          ['FZ', 'F1', 'F2', 'F3', 'F4'],
    'frontopolar':      ['FP1', 'FPZ', 'FP2'],
    'lateral_frontal':  ['F5', 'F6', 'F7', 'F8'],
    'lateral_central':  ['C5', 'C6', 'FC5', 'FC6'],
    'temporal':         ['T7', 'T8'],
    'ventral_temporal':  ['TP7', 'TP8', 'CP5', 'CP6'],
    'posterior_temporal': ['P5', 'P6', 'P7', 'P8'],
    'parieto_occipital': ['PO3', 'POZ', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8'],
    'occipital':        ['O1', 'OZ', 'O2'],
}

# Dorsoventral position: 1.0 = most dorsal, 0.0 = most ventral
DV_POSITION = {
    'dorsal':           1.0,
    'centroparietal':   0.85,
    'frontal':          0.75,
    'lateral_central':  0.65,
    'parietal':         0.70,
    'frontopolar':      0.50,
    'lateral_frontal':  0.55,
    'temporal':         0.30,
    'ventral_temporal':  0.15,
    'posterior_temporal': 0.10,
    'parieto_occipital': 0.25,
    'occipital':        0.20,
}

FREQ_BANDS = {
    'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
    'beta': (13, 30), 'broadband': (1, 45),
}

GROUP_LABELS = {
    'G1': 'PSEN1+',
    'G2': 'Non-carrier',
    'CTR': 'Control',
    'DCL': 'MCI',
    'DTA': 'AD',
}


# ============================================================
# CORE FUNCTIONS
# ============================================================

def rho_ar2(s):
    """AR(2) eigenvalue modulus from a 1D signal segment."""
    s = (s - s.mean()) / (s.std() + 1e-12)
    Y = s[2:]
    X = np.column_stack([s[1:-1], s[:-2]])
    c = np.linalg.lstsq(X, Y, rcond=None)[0]
    roots = np.roots([1, -c[0], -c[1]])
    return np.max(np.abs(roots))


def rho_epoched(s, fs, epoch_s=4.0, overlap=0.5):
    """Median rho over sliding 4-s epochs."""
    n = int(epoch_s * fs)
    step = int(n * (1 - overlap))
    rhos = []
    for i in range(0, len(s) - n, step):
        seg = s[i:i + n]
        if seg.std() < 1e-10:
            continue
        r = rho_ar2(seg)
        if 0 < r < 1.0:
            rhos.append(r)
    return np.median(rhos) if len(rhos) >= 3 else np.nan


def bpf(s, fs, lo, hi):
    """Zero-phase Butterworth bandpass."""
    b, a = signal.butter(4, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return signal.filtfilt(b, a, s)


def find_eeg_files(root):
    """Find BrainVision .vhdr files in BIDS structure."""
    files = []
    for f in sorted(Path(root).rglob('*.vhdr')):
        # Skip derivatives
        if 'derivative' in str(f).lower():
            continue
        sub = next((p for p in f.parts if p.startswith('sub-')), None)
        if sub:
            files.append((sub, str(f)))
    return files


def load_eeg(fp):
    """Load BrainVision EEG file, return MNE Raw."""
    return mne.io.read_raw_brainvision(fp, preload=True)


def process_subject(sid, fp, pts_df):
    """Process one subject: load, filter, compute rho per band x region."""
    raw = load_eeg(fp)

    # Pick only EEG channels (exclude HEO, VEO)
    eeg_chs = [ch for ch in raw.ch_names if ch not in ['HEO', 'VEO']]
    raw.pick_channels(eeg_chs)

    fs = raw.info['sfreq']
    raw.filter(0.5, 45, verbose=False)
    try:
        raw.notch_filter(60, verbose=False)  # 60 Hz for Colombia (Americas)
    except Exception:
        pass

    # Downsample to 500 Hz to match discovery cohort and avoid rho ceiling
    if fs > 500:
        raw.resample(500, verbose=False)
        fs = 500.0

    # Match metadata from participants.tsv
    # Subject IDs in participants.tsv: sub-G1002, sub-G2003, sub-CTR011, etc.
    group = '?'
    mmse = np.nan
    age = np.nan
    sex = '?'

    if len(pts_df) > 0 and 'participant_id' in pts_df.columns:
        row = pts_df[pts_df['participant_id'] == sid]
        if len(row) == 0:
            # Try with sub- prefix
            row = pts_df[pts_df['participant_id'] == f'sub-{sid}' if not sid.startswith('sub-') else sid]
        if len(row) == 0:
            # Extract group from subject ID: sub-G1002 -> G1, sub-G2003 -> G2
            if 'G1' in sid:
                group = 'G1'
            elif 'G2' in sid:
                group = 'G2'
            elif 'CTR' in sid:
                group = 'CTR'
            elif 'DCL' in sid:
                group = 'DCL'
            elif 'DTA' in sid:
                group = 'DTA'
        else:
            r = row.iloc[0]
            # Try different column names for group
            for col in ['Group', 'group', 'diagnosis']:
                if col in r.index:
                    group = str(r[col])
                    break
            # If still no group, infer from ID
            if group == '?' or group == 'nan':
                if 'G1' in sid:
                    group = 'G1'
                elif 'G2' in sid:
                    group = 'G2'
            # MMSE
            for col in ['MMSE', 'mmse', 'MMSE_score']:
                if col in r.index:
                    try:
                        mmse = float(r[col])
                    except (ValueError, TypeError):
                        pass
                    break
            # Age
            for col in ['age', 'Age', 'edad']:
                if col in r.index:
                    try:
                        age = float(r[col])
                    except (ValueError, TypeError):
                        pass
                    break
            # Sex
            for col in ['sex', 'Sex', 'gender']:
                if col in r.index:
                    sex = str(r[col])
                    break
    else:
        # No participants file — infer group from ID
        if 'G1' in sid:
            group = 'G1'
        elif 'G2' in sid:
            group = 'G2'
        elif 'CTR' in sid:
            group = 'CTR'

    res = {'subject': sid, 'group': group, 'mmse': mmse, 'age': age, 'sex': sex}

    # Compute rho for each band x region
    for bn, (flo, fhi) in FREQ_BANDS.items():
        for gn, chs in CHANNEL_GROUPS.items():
            vals = []
            for ch in chs:
                if ch in raw.ch_names:
                    d = raw.get_data(picks=[ch])[0]
                    if bn != 'broadband':
                        d = bpf(d, fs, flo, fhi)
                    r = rho_epoched(d, fs)
                    if not np.isnan(r):
                        vals.append(r)
            res[f'rho_{bn}_{gn}'] = np.mean(vals) if vals else np.nan

    return res


# ============================================================
# MAIN
# ============================================================

def main():
    data_root = os.getcwd()
    if len(sys.argv) >= 2:
        data_root = sys.argv[1]

    root = Path(data_root)
    if not root.exists():
        print("Path not found: " + str(root))
        sys.exit(1)
    print("Data root: " + str(root))

    # Load participants
    pf = list(root.rglob('participants.tsv'))
    if pf:
        pts = pd.read_csv(pf[0], sep='\t')
        print(f"Participants file: {pf[0]}")
        print(f"Total listed: {len(pts)}")
        if 'Group' in pts.columns:
            print(pts['Group'].value_counts().to_string())
        elif 'group' in pts.columns:
            print(pts['group'].value_counts().to_string())
        print()
    else:
        print("WARNING: participants.tsv not found")
        pts = pd.DataFrame()

    # Find EEG files
    files = find_eeg_files(root)
    print(f"EEG files found: {len(files)}")
    if not files:
        print("No .vhdr files found! Check folder structure.")
        sys.exit(1)

    # Show which groups have data
    groups_found = {}
    for sid, fp in files:
        for g in ['G1', 'G2', 'CTR', 'DCL', 'DTA']:
            if g in sid:
                groups_found[g] = groups_found.get(g, 0) + 1
                break
    print("Groups with EEG data:")
    for g, n in sorted(groups_found.items()):
        print(f"  {g} ({GROUP_LABELS.get(g, g)}): n={n}")

    # Inspect first file
    raw0 = load_eeg(files[0][1])
    eeg_chs = [ch for ch in raw0.ch_names if ch not in ['HEO', 'VEO']]
    print(f"\nChannels ({len(eeg_chs)} EEG): {eeg_chs[:10]}... (+ {len(eeg_chs)-10} more)")
    print(f"Sfreq: {raw0.info['sfreq']} Hz")
    print(f"Duration: {raw0.times[-1]:.0f} s")
    del raw0

    # Process all subjects
    print(f"\n{'='*50}")
    print("Processing subjects...")
    print(f"{'='*50}")
    results = []
    for i, (sid, fp) in enumerate(files):
        print("  [{:2d}/{}] {}".format(i + 1, len(files), sid), end='')
        try:
            res = process_subject(sid, fp, pts)
            results.append(res)
            g = GROUP_LABELS.get(res['group'], res['group'])
            print("  {}  age={:.0f}  MMSE={}".format(g, res['age'], res['mmse']))
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except BaseException as e:
            print("  ERROR: {}".format(e))

    df = pd.DataFrame(results)
    print(f"\nProcessed: {len(df)} subjects")

    # Compute DV gradients (dorsal - ventral_temporal)
    for band in FREQ_BANDS:
        dk = f'rho_{band}_dorsal'
        vk = f'rho_{band}_ventral_temporal'
        if dk in df.columns and vk in df.columns:
            df[f'dv_{band}'] = df[dk] - df[vk]

    # Also compute full gradient slope via linear regression on DV position
    for band in FREQ_BANDS:
        slopes = []
        for _, row in df.iterrows():
            positions = []
            rhos = []
            for gn, pos in DV_POSITION.items():
                val = row.get(f'rho_{band}_{gn}', np.nan)
                if not np.isnan(val):
                    positions.append(pos)
                    rhos.append(val)
            if len(positions) >= 4:
                z = np.polyfit(positions, rhos, 1)
                slopes.append(z[0])
            else:
                slopes.append(np.nan)
        df[f'slope_{band}'] = slopes

    # Save
    out_csv = root / 'rho_preclinical_results.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved: {out_csv}")

    # ============================================================
    # STATISTICS
    # ============================================================
    print(f"\n{'='*60}")
    print("PRECLINICAL VALIDATION RESULTS")
    print("PSEN1 E280A carriers (G1) vs non-carrier family (G2)")
    print(f"{'='*60}")

    g1 = df[df['group'] == 'G1']
    g2 = df[df['group'] == 'G2']
    print(f"\nG1 (PSEN1+):     n={len(g1)}, age={g1['age'].mean():.1f}±{g1['age'].std():.1f}")
    print(f"G2 (Non-carrier): n={len(g2)}, age={g2['age'].mean():.1f}±{g2['age'].std():.1f}")

    # Global mean rho (average across ALL regions)
    print(f"\n--- GLOBAL MEAN RHO (averaged across all regions) ---")
    for band in FREQ_BANDS:
        region_cols = [f'rho_{band}_{gn}' for gn in CHANNEL_GROUPS if f'rho_{band}_{gn}' in df.columns]
        if not region_cols:
            continue
        df[f'global_rho_{band}'] = df[region_cols].mean(axis=1)

    # Re-slice after adding columns
    g1 = df[df['group'] == 'G1']
    g2 = df[df['group'] == 'G2']

    for band in FREQ_BANDS:
        col = f'global_rho_{band}'
        if col not in df.columns:
            continue
        v1 = g1[col].dropna()
        v2 = g2[col].dropna()

        if len(v1) >= 3 and len(v2) >= 3:
            t, p = ttest_ind(v1, v2)
            pooled_sd = np.sqrt((v1.var() + v2.var()) / 2)
            d = (v1.mean() - v2.mean()) / pooled_sd if pooled_sd > 0 else np.nan
            u, p_u = mannwhitneyu(v1, v2, alternative='two-sided')
            print(f"\n  {band.upper()}:")
            print(f"    PSEN1+:      {v1.mean():.6f} ± {v1.sem():.6f}  (n={len(v1)})")
            print(f"    Non-carrier: {v2.mean():.6f} ± {v2.sem():.6f}  (n={len(v2)})")
            print(f"    t={t:.3f}, p={p:.4f}, Cohen's d={d:.3f}")
            print(f"    Mann-Whitney U={u:.0f}, p={p_u:.4f}")

    print(f"\n--- DV GRADIENT (dorsal - ventral_temporal rho) ---")
    for band in FREQ_BANDS:
        gk = f'dv_{band}'
        if gk not in df.columns:
            continue
        print(f"\n  {band.upper()}:")
        v1 = g1[gk].dropna()
        v2 = g2[gk].dropna()
        print(f"    PSEN1+:      {v1.mean():+.5f} ± {v1.sem():.5f}  (n={len(v1)})")
        print(f"    Non-carrier: {v2.mean():+.5f} ± {v2.sem():.5f}  (n={len(v2)})")

        # One-sample t-tests: is each group's gradient != 0?
        from scipy.stats import ttest_1samp
        if len(v1) >= 3:
            t1, p1 = ttest_1samp(v1, 0)
            print(f"    PSEN1+ gradient != 0?   t={t1:.3f}, p={p1:.4f}")
        if len(v2) >= 3:
            t2, p2 = ttest_1samp(v2, 0)
            print(f"    Non-carrier gradient != 0?  t={t2:.3f}, p={p2:.4f}")

        if len(v1) >= 3 and len(v2) >= 3:
            t, p = ttest_ind(v1, v2)
            pooled_sd = np.sqrt((v1.var() + v2.var()) / 2)
            d = (v1.mean() - v2.mean()) / pooled_sd if pooled_sd > 0 else np.nan
            # Also Mann-Whitney U (non-parametric)
            u, p_u = mannwhitneyu(v1, v2, alternative='two-sided')
            print(f"    G1 vs G2: t={t:.3f}, p={p:.4f}, Cohen's d={d:.3f}")
            print(f"    Mann-Whitney U={u:.0f}, p={p_u:.4f}")

    print(f"\n--- GRADIENT SLOPE (linear fit across all regions) ---")
    for band in FREQ_BANDS:
        sk = f'slope_{band}'
        if sk not in df.columns:
            continue
        print(f"\n  {band.upper()}:")
        v1 = g1[sk].dropna()
        v2 = g2[sk].dropna()
        print(f"    PSEN1+:      {v1.mean():+.5f} ± {v1.sem():.5f}  (n={len(v1)})")
        print(f"    Non-carrier: {v2.mean():+.5f} ± {v2.sem():.5f}  (n={len(v2)})")

        if len(v1) >= 3 and len(v2) >= 3:
            # One-sample: does each group have a non-zero slope?
            from scipy.stats import ttest_1samp
            t1, p1 = ttest_1samp(v1, 0)
            t2, p2 = ttest_1samp(v2, 0)
            print(f"    PSEN1+ slope != 0?      t={t1:.3f}, p={p1:.4f}")
            print(f"    Non-carrier slope != 0? t={t2:.3f}, p={p2:.4f}")
            # Two-sample: do they differ?
            t, p = ttest_ind(v1, v2)
            pooled_sd = np.sqrt((v1.var() + v2.var()) / 2)
            d = (v1.mean() - v2.mean()) / pooled_sd if pooled_sd > 0 else np.nan
            print(f"    G1 vs G2: t={t:.3f}, p={p:.4f}, Cohen's d={d:.3f}")

    # Per-region rho means
    print(f"\n--- PER-REGION RHO (alpha band) ---")
    print(f"  {'Region':<22s}  {'PSEN1+':>10s}  {'Non-carrier':>12s}  {'diff':>8s}  {'p':>8s}")
    for gn in CHANNEL_GROUPS:
        k = f'rho_alpha_{gn}'
        if k not in df.columns:
            continue
        v1 = g1[k].dropna()
        v2 = g2[k].dropna()
        if len(v1) >= 3 and len(v2) >= 3:
            t, p = ttest_ind(v1, v2)
            diff = v1.mean() - v2.mean()
            print(f"  {gn:<22s}  {v1.mean():.5f}  {v2.mean():.5f}  {diff:+.5f}  {p:.4f}")

    print(f"\nDone. Results saved to {out_csv}")


if __name__ == '__main__':
    main()
