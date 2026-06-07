#!/usr/bin/env python
"""clinical_discrimination.py — diagnostic performance of the rho DV-gradient as a
neurodegeneration marker. Reports ROC-AUC, sensitivity & specificity (at Youden's J),
for AD/NC, FTD/NC, and pooled ALL-DEM/NC, plus the decisive CROSS-DIAGNOSIS TRANSFER
test (train on AD, test on unseen FTD) that quantifies how 'general' the marker is.

Honest small-N practice:
  - multivariate logistic uses leave-one-out CV (out-of-sample probabilities)
  - univariate (single best band) reported as the rank AUC
  - cross-diagnosis: model NEVER sees the test disease during training
"""
import pandas as pd, numpy as np, os, urllib.request
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT="/home/salardini/projects/pH_cortical_dynamics"
BANDS=["delta","theta","alpha","beta","broadband"]
DVCOLS=[f"dv_{b}" for b in BANDS]

disc=pd.read_csv(f"{ROOT}/data/discovery_ds004504/rho_gradient_results.csv")
ptpath=f"{ROOT}/data/discovery_ds004504/participants.tsv"
if not os.path.exists(ptpath):
    urllib.request.urlretrieve("https://s3.amazonaws.com/openneuro.org/ds004504/participants.tsv",ptpath)
pts=pd.read_csv(ptpath,sep="\t").rename(columns={"participant_id":"subject"})
disc=disc.merge(pts[["subject","Age","Gender"]],on="subject",how="left").dropna(subset=DVCOLS)

def youden(y,score):
    fpr,tpr,thr=roc_curve(y,score); j=tpr-fpr; k=j.argmax()
    return thr[k],tpr[k],1-fpr[k]   # threshold, sensitivity, specificity

def boot_auc_ci(y,score,B=2000,seed=0):
    """percentile bootstrap 95% CI for AUC (resample subject pairs)."""
    rng=np.random.RandomState(seed); y=np.asarray(y); score=np.asarray(score)
    n=len(y); aucs=[]
    for _ in range(B):
        idx=rng.randint(0,n,n)
        if len(np.unique(y[idx]))<2: continue
        aucs.append(roc_auc_score(y[idx],score[idx]))
    return np.percentile(aucs,2.5),np.percentile(aucs,97.5)

def cv_probs(X,y):
    clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))
    return cross_val_predict(clf,X,y,cv=LeaveOneOut(),method="predict_proba")[:,1]

def evaluate(df,posmask,negmask,name):
    d=df[posmask|negmask]; y=posmask[posmask|negmask].astype(int).values
    # multivariate panel (5 dv bands), LOO-CV
    p=cv_probs(d[DVCOLS].values,y)
    auc=roc_auc_score(y,p); thr,sens,spec=youden(y,p); lo,hi=boot_auc_ci(y,p)
    # best single band (alpha dv), rank AUC
    s=d["dv_alpha"].values
    if roc_auc_score(y,s)<0.5: s=-s
    auc1=roc_auc_score(y,s); _,se1,sp1=youden(y,s); lo1,hi1=boot_auc_ci(y,s)
    print(f"  {name:16s} n+={y.sum():2d} n-={ (y==0).sum():2d} | "
          f"PANEL AUC={auc:.3f} [{lo:.2f}-{hi:.2f}] sens={sens:.2f} spec={spec:.2f} | "
          f"alpha AUC={auc1:.3f} [{lo1:.2f}-{hi1:.2f}] sens={se1:.2f} spec={sp1:.2f}")
    return name,y.sum(),(y==0).sum(),auc,lo,hi,sens,spec,auc1,lo1,hi1,(y,p,auc)

gA=disc.group=="A"; gF=disc.group=="F"; gC=disc.group=="C"
print("=== within-cohort diagnostic performance (DV-gradient) ===")
rows=[]; roc_store={}
for nm,pos,neg in [("AD vs NC",gA,gC),("FTD vs NC",gF,gC),("ALL-DEM vs NC",(gA|gF),gC)]:
    r=evaluate(disc,pos,neg,nm); rows.append(r[:11]); roc_store[nm]=r[11]

# ---- CROSS-DIAGNOSIS TRANSFER: train AD vs NC, test on FTD vs NC (FTD unseen) ----
print("\n=== cross-diagnosis transfer (model never sees the test disease) ===")
def transfer(train_pos,test_pos,neg,label):
    clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))
    Xtr=disc[train_pos|neg][DVCOLS].values; ytr=train_pos[train_pos|neg].astype(int).values
    clf.fit(Xtr,ytr)
    te=disc[test_pos|neg]; yte=test_pos[test_pos|neg].astype(int).values
    p=clf.predict_proba(te[DVCOLS].values)[:,1]
    auc=roc_auc_score(yte,p); thr,sens,spec=youden(yte,p); lo,hi=boot_auc_ci(yte,p)
    print(f"  train {label:18s} AUC={auc:.3f} [{lo:.2f}-{hi:.2f}] sens={sens:.2f} spec={spec:.2f}")
    return label,auc,lo,hi,sens,spec
tr=[]
tr.append(transfer(gA,gF,gC,"AD -> predict FTD"))
tr.append(transfer(gF,gA,gC,"FTD -> predict AD"))

# ---- ROC figure ----
plt.figure(figsize=(6.2,6))
for nm,col in zip(["AD vs NC","FTD vs NC","ALL-DEM vs NC"],["#1f77b4","#2ca02c","#d62728"]):
    y,p,auc=roc_store[nm]; fpr,tpr,_=roc_curve(y,p)
    plt.plot(fpr,tpr,color=col,lw=2,label=f"{nm}  AUC={auc:.2f}")
plt.plot([0,1],[0,1],"--",color="gray",lw=1)
plt.xlabel("1 - specificity (FPR)"); plt.ylabel("sensitivity (TPR)")
plt.title("DV-gradient ρ — diagnostic ROC (LOO-CV)"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig(f"{ROOT}/figures/main/roc_neurodegeneration.png",dpi=140)
print("\nsaved figures/main/roc_neurodegeneration.png")

out=pd.DataFrame(rows,columns=["contrast","n_pos","n_neg","panel_auc","panel_lo","panel_hi",
                               "sens","spec","alpha_auc","alpha_lo","alpha_hi"])
tr=pd.DataFrame(tr,columns=["transfer","auc","auc_lo","auc_hi","sens","spec"])
out.to_csv(f"{ROOT}/data/clinical_discrimination.csv",index=False)
tr.to_csv(f"{ROOT}/data/clinical_transfer.csv",index=False)
print("saved data/clinical_discrimination.csv , data/clinical_transfer.csv")
