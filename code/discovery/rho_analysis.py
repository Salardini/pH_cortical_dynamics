"""
Rho Gradient Analysis in Alzheimer's Disease EEG
=================================================
Port-Hamiltonian Prediction 1: AD flattens the dorsoventral rho gradient.

Dataset: OpenNeuro ds004504 (Miltiadous et al. 2023)
  36 AD, 23 FTD, 29 controls, 19-channel resting EEG, 500 Hz

Usage:
  Place this script in your EEG data folder and run:
    python rho_analysis.py

  Or specify the path explicitly:
    python rho_analysis.py path_to_eeg_folder
"""

import sys
import os
import numpy as np
import pandas as pd
import mne
from scipy import signal
from scipy.stats import ttest_ind, spearmanr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# ============================================================
# CONFIGURATION
# ============================================================

CHANNEL_GROUPS = {
    'dorsal':           ['C3', 'Cz', 'C4'],
    'ventral_temporal':  ['T5', 'T6'],
    'lateral_temporal':  ['T3', 'T4'],
    'frontal':           ['F3', 'Fz', 'F4'],
    'frontopolar':       ['Fp1', 'Fp2'],
    'parietal':          ['P3', 'Pz', 'P4'],
    'occipital':         ['O1', 'O2'],
}

ALIAS = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}

DV_POSITION = {
    'ventral_temporal': 0.0, 'lateral_temporal': 0.2, 'occipital': 0.3,
    'frontopolar': 0.4, 'frontal': 0.6, 'parietal': 0.7, 'dorsal': 1.0,
}

FREQ_BANDS = {
    'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
    'beta': (13, 30), 'broadband': (1, 45),
}

GROUP_LABELS = {'A': 'AD', 'F': 'FTD', 'C': 'Control'}


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
    """Find all .set/.edf/.fif files, prefer preprocessed."""
    files = []
    for ext in ['*.set', '*.edf', '*.fif']:
        for f in sorted(Path(root).rglob(ext)):
            sub = next((p for p in f.parts if p.startswith('sub-')), None)
            if sub:
                files.append((sub, str(f)))
    # Prefer preprocessed
    preproc = [(s, f) for s, f in files
               if 'derivative' in f.lower() or 'preprocess' in f.lower()]
    return preproc if preproc else files


def load_eeg(fp):
    """Load EEG file, return MNE Raw."""
    if fp.endswith('.set'):
        return mne.io.read_raw_eeglab(fp, preload=True)
    elif fp.endswith('.edf'):
        return mne.io.read_raw_edf(fp, preload=True)
    else:
        return mne.io.read_raw_fif(fp, preload=True)


def fix_channel_names(raw):
    """Standardize channel names to match our CHANNEL_GROUPS definitions.
    Handles: uppercase (FP1->Fp1, FZ->Fz, CZ->Cz, PZ->Pz)
    and newer nomenclature (T7->T3, T8->T4, P7->T5, P8->T6)."""
    # Build a mapping from whatever is in the file to our standard names
    # Our standard names: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2,
    #   F7, F8, T3, T4, T5, T6, Fz, Cz, Pz, F3, F4
    standard = {
        'FP1': 'Fp1', 'FP2': 'Fp2',
        'F3': 'F3', 'F4': 'F4', 'F7': 'F7', 'F8': 'F8',
        'C3': 'C3', 'C4': 'C4',
        'P3': 'P3', 'P4': 'P4',
        'O1': 'O1', 'O2': 'O2',
        'T3': 'T3', 'T4': 'T4', 'T5': 'T5', 'T6': 'T6',
        'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz',
        # Newer 10-10 names
        'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6',
        # Already correct
        'Fp1': 'Fp1', 'Fp2': 'Fp2',
        'Fz': 'Fz', 'Cz': 'Cz', 'Pz': 'Pz',
    }
    rmap = {}
    for ch in raw.ch_names:
        cu = ch.strip()
        target = standard.get(cu.upper(), None)
        if target and target != cu:
            rmap[cu] = target
    if rmap:
        try:
            raw.rename_channels(rmap)
        except Exception:
            pass
    return raw


def process_subject(sid, fp, pts_df):
    """Process one subject: load, filter, compute rho per band x region."""
    raw = fix_channel_names(load_eeg(fp))
    fs = raw.info['sfreq']
    raw.filter(0.5, 45, verbose=False)
    try:
        raw.notch_filter(50, verbose=False)
    except Exception:
        pass

    # Match metadata
    sub_num = sid.replace('sub-', '')
    group = '?'
    mmse = np.nan
    if len(pts_df) > 0 and 'participant_id' in pts_df.columns:
        row = pts_df[pts_df['participant_id'].str.endswith(sub_num)]
        if len(row) == 0:
            row = pts_df[pts_df['participant_id'].str.contains(sub_num)]
        if len(row) > 0:
            if 'Group' in row.columns:
                group = row['Group'].values[0]
            elif 'group' in row.columns:
                group = row['group'].values[0]
            # Try different MMSE column names
            for col in ['MMSE', 'mmse', 'MMSE_score', 'mmse_score']:
                if col in row.columns:
                    try:
                        mmse = float(row[col].values[0])
                    except (ValueError, TypeError):
                        pass
                    break

    res = {'subject': sid, 'group': group, 'mmse': mmse}

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
    # Use current working directory (where you ran the command from)
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
        print(f"Participants: {len(pts)}")
        print(pts['Group'].value_counts().to_string())
    else:
        print("WARNING: participants.tsv not found")
        pts = pd.DataFrame()

    # Find EEG files
    files = find_eeg_files(root)
    print(f"\nEEG files: {len(files)}")
    if not files:
        print("No EEG files found! Check folder structure.")
        sys.exit(1)

    # Inspect first file
    raw0 = fix_channel_names(load_eeg(files[0][1]))
    print(f"Channels ({len(raw0.ch_names)}): {raw0.ch_names}")
    print(f"Sfreq: {raw0.info['sfreq']} Hz")
    print(f"Duration: {raw0.times[-1]:.0f} s")
    del raw0

    # Process all
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
            print("  {}  MMSE={}".format(g, res['mmse']))
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except BaseException as e:
            print("  ERROR: {}".format(e))

    df = pd.DataFrame(results)
    print(f"\nProcessed: {len(df)} subjects")

    # Compute DV gradients
    for band in FREQ_BANDS:
        dk = f'rho_{band}_dorsal'
        vk = f'rho_{band}_ventral_temporal'
        if dk in df.columns and vk in df.columns:
            df[f'dv_{band}'] = df[dk] - df[vk]

    # Save
    out_csv = root / 'rho_gradient_results.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved: {out_csv}")

    # Print stats
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    for band in FREQ_BANDS:
        gk = f'dv_{band}'
        if gk not in df.columns:
            continue
        print(f"\n--- {band.upper()} ---")
        for grp in ['C', 'A', 'F']:
            v = df[df['group'] == grp][gk].dropna()
            lab = GROUP_LABELS.get(grp, grp)
            print(f"  {lab:10s}: {v.mean():+.5f} +/- {v.sem():.5f}  (n={len(v)})")

        ad = df[df['group'] == 'A'][gk].dropna()
        cn = df[df['group'] == 'C'][gk].dropna()
        ftd = df[df['group'] == 'F'][gk].dropna()

        if len(ad) > 3 and len(cn) > 3:
            t, p = ttest_ind(ad, cn)
            d = (ad.mean() - cn.mean()) / np.sqrt((ad.var() + cn.var()) / 2)
            print(f"  AD vs CN:  t={t:.3f}, p={p:.4f}, d={d:.3f}")

        if len(ftd) > 3 and len(cn) > 3:
            t, p = ttest_ind(ftd, cn)
            d = (ftd.mean() - cn.mean()) / np.sqrt((ftd.var() + cn.var()) / 2)
            print(f"  FTD vs CN: t={t:.3f}, p={p:.4f}, d={d:.3f}")

        m = df['mmse'].notna() & df[gk].notna()
        if m.sum() > 10:
            r, p = spearmanr(df.loc[m, 'mmse'], df.loc[m, gk])
            print(f"  MMSE corr: rho_s={r:.3f}, p={p:.4f}")

    print(f"\nDone. Run plot_results.py to generate figures.")


if __name__ == '__main__':
    main()
