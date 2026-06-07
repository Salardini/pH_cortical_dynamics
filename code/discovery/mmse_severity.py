#!/usr/bin/env python
"""mmse_severity.py — is the rho DV-gradient a graded SEVERITY marker or a binary
STATE marker of neurodegeneration?

Tests the association between rho metrics and MMSE:
  (1) across the full healthy->dementia spectrum,
  (2) within the dementia groups pooled,
  (3) within AD and FTD SEPARATELY -- because MMSE is memory/orientation/language
      weighted and is sensitive to AD decline but relatively blind to the
      dysexecutive/behavioural decline of FTD, so a pooled correlation is invalid.

Also expresses each subject's broadband gradient as 'equivalent extra aging' (years),
using the healthy controls' own dv-vs-age slope as the normal-aging reference
(same metric, same modality -- a cross-modality slope, e.g. from CamCAN MEG, would
NOT be unit-compatible).

Result (see manuscript notes): the gradient tracks MMSE across the full spectrum
(driven by the healthy-vs-dementia separation; controls sit at the MMSE=30 ceiling)
but NOT within either disease -- including AD, where MMSE has a wide valid range
(4-23). The marker behaves as a STATE detector, not a within-disease severity gauge.
Differential MMSE sensitivity does not explain the null: AD is flat where MMSE works.

Usage:  python mmse_severity.py
"""
import pandas as pd, numpy as np, os, urllib.request
from scipy.stats import linregress, spearmanr
import statsmodels.formula.api as smf

ROOT="/home/salardini/projects/pH_cortical_dynamics"
disc=pd.read_csv(f"{ROOT}/data/discovery_ds004504/rho_gradient_results.csv")
ptpath=f"{ROOT}/data/discovery_ds004504/participants.tsv"
if not os.path.exists(ptpath):
    urllib.request.urlretrieve("https://s3.amazonaws.com/openneuro.org/ds004504/participants.tsv",ptpath)
pts=pd.read_csv(ptpath,sep="\t").rename(columns={"participant_id":"subject"})
d=disc.merge(pts[["subject","Age","Gender"]],on="subject",how="left")

def global_rho(df,b): return df[[c for c in df.columns if c.startswith(f"rho_{b}_")]].mean(axis=1)
d["grho_alpha"]=global_rho(d,"alpha"); d["grho_beta"]=global_rho(d,"beta")

# equivalent extra aging (broadband) using NC's own dv-vs-age slope
NC=d[d.group=="C"]
nc_slope=linregress(NC.Age,NC.dv_broadband).slope          # only broadband has p<.05 in NC
d["extra_aging_yr"]=(d.dv_broadband-NC.dv_broadband.mean())/nc_slope

METRICS=[("dv_broadband","dv_broadband"),("dv_alpha","dv_alpha"),
         ("extra_aging_yr","extra_aging_yr"),("global_rho_alpha","grho_alpha"),
         ("global_rho_beta","grho_beta")]

def corr_block(sub,title):
    print(f"\n=== {title}  (n={len(sub)}, MMSE mean {sub.mmse.mean():.1f} sd {sub.mmse.std():.1f}) ===")
    print(f"  {'metric':18s}{'Spearman r':>12s}{'p':>9s}{'age+sex adj p':>15s}")
    for nm,col in METRICS:
        s=sub[[col,"mmse","Age","Gender"]].dropna()
        if s.mmse.nunique()<3: print(f"  {nm:18s}   (MMSE has no variance -- ceiling)"); continue
        r,p=spearmanr(s[col],s.mmse)
        adj=smf.ols(f"{col} ~ mmse + Age + C(Gender)",s).fit().pvalues["mmse"]
        star="*" if p<0.05 else " "
        print(f"  {nm:18s}{r:>+12.3f}{p:>9.3f}{star}{adj:>15.3f}")

print("MMSE by group:")
print(d.groupby("group").mmse.agg(["count","mean","min","max"]).round(1).to_string())
corr_block(d, "FULL SPECTRUM (NC+AD+FTD)")
corr_block(d[d.group.isin(["A","F"])], "DEMENTIA POOLED (AD+FTD)")
corr_block(d[d.group=="A"], "ALZHEIMER ONLY  [MMSE valid, wide range]")
corr_block(d[d.group=="F"], "FTD ONLY  [MMSE relatively insensitive]")

ad=d[d.group=="A"].dropna(subset=["extra_aging_yr","mmse"])
lr=linregress(ad.mmse,ad.extra_aging_yr)
print(f"\nAD: equivalent aging per MMSE point lost = {-lr.slope:.2f} yr/point (p={lr.pvalue:.3f})")
print("\nConclusion: state marker, not within-disease severity gauge "
      "(AD flat where MMSE is valid).")
