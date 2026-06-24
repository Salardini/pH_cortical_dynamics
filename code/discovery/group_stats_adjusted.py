#!/usr/bin/env python
"""group_stats_adjusted.py — all group contrasts on the rho DV-gradient (and global
rho), reporting raw t-test p, age+sex-adjusted p (OLS ANCOVA), and BH-FDR q across
the 5 bands within each contrast. Includes the pooled all-dementia (AD+FTD) vs NC
test that motivates the 'general neurodegeneration' framing.

Demographics for ds004504 are fetched from OpenNeuro (participants.tsv) and cached
alongside the results CSV. ds007427 carries age/sex inline.
"""
import pandas as pd, numpy as np, os, urllib.request
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT="/home/salardini/projects/pH_cortical_dynamics"
BANDS=["delta","theta","alpha","beta","broadband"]

# ---- discovery demographics (cache participants.tsv in the repo) ----
disc=pd.read_csv(f"{ROOT}/data/discovery_ds004504/rho_gradient_results.csv")
ptpath=f"{ROOT}/data/discovery_ds004504/participants.tsv"
if not os.path.exists(ptpath):
    urllib.request.urlretrieve("https://s3.amazonaws.com/openneuro.org/ds004504/participants.tsv",ptpath)
pts=pd.read_csv(ptpath,sep="\t").rename(columns={"participant_id":"subject"})
disc=disc.merge(pts[["subject","Age","Gender"]],on="subject",how="left")
disc=disc.rename(columns={"Age":"age","Gender":"sex"})

pre=pd.read_csv(f"{ROOT}/data/preclinical_ds007427/rho_preclinical_results.csv")
pre=pre[pre.group.isin(["G1","G2"])].copy()

def global_rho(df,b):
    return df[[c for c in df.columns if c.startswith(f"rho_{b}_")]].mean(axis=1)

def contrast(df, posmask, negmask, posn, negn, metric, label):
    """one contrast x one metric across all bands -> rows with raw/adj/fdr."""
    rows=[]; raws=[]
    for b in BANDS:
        y = df[f"dv_{b}"] if metric=="dv" else global_rho(df,b)
        s=pd.DataFrame({"y":y,"grp":posmask.astype(int),"age":df["age"],"sex":df["sex"]})
        s=s[posmask|negmask].dropna()
        a=s[s.grp==1].y; c=s[s.grp==0].y
        raw=ttest_ind(a,c).pvalue
        adj=smf.ols("y ~ grp + age + C(sex)",s).fit().pvalues["grp"]
        rows.append([label,metric,b,len(a),len(c),a.mean(),c.mean(),raw,adj]); raws.append(raw)
    q=multipletests(raws,method="fdr_bh")[1]
    for r,qq in zip(rows,q): r.append(qq)
    return rows

allrows=[]
# discovery contrasts
gA=disc.group=="A"; gF=disc.group=="F"; gC=disc.group=="C"
for metric in ["dv","global_rho"]:
    allrows+=contrast(disc,gA,gC,"AD","NC",metric,"AD vs NC")
    allrows+=contrast(disc,gF,gC,"FTD","NC",metric,"FTD vs NC")
    allrows+=contrast(disc,gA,gF,"AD","FTD",metric,"AD vs FTD")
    allrows+=contrast(disc,(gA|gF),gC,"DEM","NC",metric,"ALL-DEM vs NC")  # pooled
# preclinical contrast
g1=pre.group=="G1"; g2=pre.group=="G2"; pre=pre.rename(columns={})  # age/sex already named
for metric in ["dv","global_rho"]:
    allrows+=contrast(pre,g1,g2,"PSEN1+","NC",metric,"PreSEN vs NC")

out=pd.DataFrame(allrows,columns=["contrast","metric","band","nA","nB","meanA","meanB",
                                  "raw_p","adj_p_agesex","fdr_q"])
out.to_csv(f"{ROOT}/data/group_stats_adjusted.csv",index=False)

def show(metric):
    print(f"\n================= metric: {metric} =================")
    print(f"{'contrast':16s}{'band':10s}{'meanA':>10s}{'meanB':>10s}{'raw_p':>9s}{'adj_p':>9s}{'fdr_q':>9s}")
    for _,r in out[out.metric==metric].iterrows():
        s="*" if r.raw_p<0.05 else " "; qs="†" if r.fdr_q<0.05 else " "
        print(f"{r.contrast:16s}{r.band:10s}{r.meanA:>+10.5f}{r.meanB:>+10.5f}"
              f"{r.raw_p:>9.4f}{s}{r.adj_p_agesex:>9.4f}{r.fdr_q:>9.4f}{qs}")
show("dv"); show("global_rho")
print("\n* raw p<.05   † BH-FDR q<.05 (within contrast x metric, 5 bands)")
print(f"\nsaved -> data/group_stats_adjusted.csv  ({len(out)} rows)")
print(f"cached demographics -> data/discovery_ds004504/participants.tsv")
