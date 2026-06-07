#!/usr/bin/env python
"""tau_rho_analysis.py — extend the ds004504 EEG analysis with the intrinsic
TIMESCALE tau, computed the SAME way as the CamCAN/MEG pipeline (ACF-integral,
meg_axes.compute_tau) so the two datasets are comparable. rho is recomputed on the
identical epochs (AR(2) modulus, as in rho_analysis.py) for a clean rho/tau pairing.

Outputs data/discovery_ds004504/tau_rho_results.csv with, per subject:
  rho_{band}_{region}, tau_{band}_{region}, dv_rho_{band}, dv_tau_{band}, group, age, sex, mmse

Usage: python tau_rho_analysis.py [eeg_root]   (default: ~/data/eeg_ad/ds004504)
"""
import sys, os, numpy as np, pandas as pd, mne
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.expanduser("~/projects/Dorso-Ventral-Gradient"))
from rho_analysis import (CHANNEL_GROUPS, FREQ_BANDS, FREQ_BANDS as FB,
                          rho_ar2, bpf, load_eeg, fix_channel_names, find_eeg_files)
from meg_axes.metrics import compute_tau
import warnings; warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")

EEG_ROOT=os.path.expanduser(sys.argv[1] if len(sys.argv)>1 else "~/data/eeg_ad/ds004504")
OUT="/home/salardini/projects/pH_cortical_dynamics/data/discovery_ds004504/tau_rho_results.csv"

def epoched(s, fs, fn, epoch_s=4.0, overlap=0.5, lo=0.0, hi=1.0):
    """median of per-epoch estimator fn over sliding 4-s windows."""
    n=int(epoch_s*fs); step=int(n*(1-overlap)); vals=[]
    for i in range(0, len(s)-n, step):
        seg=s[i:i+n]
        if seg.std()<1e-10: continue
        v=fn(seg, fs)
        if v is not None and np.isfinite(v) and lo<v<hi: vals.append(v)
    return np.median(vals) if len(vals)>=3 else np.nan

rho_fn = lambda seg,fs: rho_ar2(seg)                              # 0<rho<1
tau_fn = lambda seg,fs: compute_tau(seg,fs).tau_integral          # seconds

def process(sid, fp):
    raw=fix_channel_names(load_eeg(fp)); fs=raw.info["sfreq"]
    raw.filter(0.5,45,verbose=False)
    try: raw.notch_filter(50,verbose=False)
    except Exception: pass
    res={"subject":sid}
    for bn,(flo,fhi) in FREQ_BANDS.items():
        for gn,chs in CHANNEL_GROUPS.items():
            rr,tt=[],[]
            for ch in chs:
                if ch in raw.ch_names:
                    d=raw.get_data(picks=[ch])[0]
                    if bn!="broadband": d=bpf(d,fs,flo,fhi)
                    r=epoched(d,fs,rho_fn,lo=0,hi=1.0)
                    t=epoched(d,fs,tau_fn,lo=0,hi=10.0)
                    if np.isfinite(r): rr.append(r)
                    if np.isfinite(t): tt.append(t)
            res[f"rho_{bn}_{gn}"]=np.mean(rr) if rr else np.nan
            res[f"tau_{bn}_{gn}"]=np.mean(tt) if tt else np.nan
    return res

def main():
    files=find_eeg_files(EEG_ROOT)
    print(f"EEG files: {len(files)} under {EEG_ROOT}")
    pts=pd.read_csv(Path(EEG_ROOT)/"participants.tsv",sep="\t")
    rows=[]
    for i,(sid,fp) in enumerate(files):
        try:
            rows.append(process(sid,fp)); print(f"  [{i+1:2d}/{len(files)}] {sid} ok")
        except Exception as e:
            print(f"  [{i+1:2d}/{len(files)}] {sid} ERROR {e}")
    df=pd.DataFrame(rows)
    # metadata
    pts=pts.rename(columns={"participant_id":"subject","Group":"group","Age":"age",
                            "Gender":"sex","MMSE":"mmse"})
    df=df.merge(pts[["subject","group","age","sex","mmse"]],on="subject",how="left")
    for b in FREQ_BANDS:
        df[f"dv_rho_{b}"]=df[f"rho_{b}_dorsal"]-df[f"rho_{b}_ventral_temporal"]
        df[f"dv_tau_{b}"]=df[f"tau_{b}_dorsal"]-df[f"tau_{b}_ventral_temporal"]
    df.to_csv(OUT,index=False)
    print(f"\nsaved {OUT}  ({len(df)} subjects)")
    # quick sanity: group means of broadband dv_tau and rho-tau correlation
    print("\nbroadband dv_tau by group:")
    print(df.groupby("group").dv_tau_broadband.mean().round(5).to_string())
    rr=df[[c for c in df if c.startswith('rho_')]].mean(axis=1)
    tt=df[[c for c in df if c.startswith('tau_')]].mean(axis=1)
    from scipy.stats import spearmanr
    print(f"\nglobal rho-tau Spearman across subjects: {spearmanr(rr,tt,nan_policy='omit')[0]:+.3f}")

if __name__=="__main__":
    main()
