"""
Plot rho gradient results.

Usage:
  python plot_results.py rho_gradient_results.csv
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from pathlib import Path

DV = {
    'ventral_temporal': 0.0, 'lateral_temporal': 0.2, 'occipital': 0.3,
    'frontopolar': 0.4, 'frontal': 0.6, 'parietal': 0.7, 'dorsal': 1.0,
}
GL = {'A': 'AD', 'F': 'FTD', 'C': 'Control'}
GC = {'AD': '#d62728', 'FTD': '#ff7f0e', 'Control': '#2e75b6'}
BANDS = ['delta', 'theta', 'alpha', 'beta', 'broadband']


def main():
    if len(sys.argv) < 2:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'rho_gradient_results.csv')
    else:
        csv_path = sys.argv[1]

    df = pd.read_csv(csv_path)
    out_dir = Path(csv_path).parent
    print(f"Loaded {len(df)} subjects from {csv_path}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Rho Gradient: AD vs Controls (pH Prediction 1)',
                 fontsize=16, fontweight='bold', y=1.02)

    # ---- Panel A: Broadband rho by DV position ----
    ax = axes[0, 0]
    for grp in ['C', 'A', 'F']:
        gdf = df[df['group'] == grp]
        ms, es, ps = [], [], []
        for region, dv in sorted(DV.items(), key=lambda x: x[1]):
            k = f'rho_broadband_{region}'
            if k in gdf.columns:
                v = gdf[k].dropna()
                if len(v):
                    ms.append(v.mean())
                    es.append(v.sem())
                    ps.append(dv)
        if ms:
            lab = GL.get(grp, grp)
            ax.errorbar(ps, ms, yerr=es, marker='o', lw=2.5, capsize=5,
                        label=lab, color=GC.get(lab, 'gray'), markersize=8)
    ax.set_xlabel('DV Position (0=ventral, 1=dorsal)', fontsize=12)
    ax.set_ylabel('rho (broadband)', fontsize=12)
    ax.set_title('A. Broadband rho by cortical region', fontweight='bold')
    ax.legend(fontsize=11)

    # ---- Panel B: DV gradient boxplot ----
    ax = axes[0, 1]
    if 'dv_broadband' in df.columns:
        gdata = []
        for grp in ['C', 'A', 'F']:
            for v in df[df['group'] == grp]['dv_broadband'].dropna():
                gdata.append({'Group': GL.get(grp, grp), 'Gradient': v})
        gdf2 = pd.DataFrame(gdata)
        if len(gdf2):
            sns.boxplot(data=gdf2, x='Group', y='Gradient', ax=ax,
                        order=['Control', 'AD', 'FTD'],
                        palette=[GC[g] for g in ['Control', 'AD', 'FTD']])
            sns.stripplot(data=gdf2, x='Group', y='Gradient', ax=ax,
                          order=['Control', 'AD', 'FTD'],
                          color='k', alpha=0.3, size=4)
            ax.axhline(0, color='k', ls='--', alpha=0.3)
            ax.set_title('B. DV Gradient by Group', fontweight='bold')
            ax.set_ylabel('rho(dorsal) - rho(ventral)', fontsize=12)

    # ---- Panel C: MMSE correlation ----
    ax = axes[0, 2]
    if 'dv_broadband' in df.columns:
        m = df['mmse'].notna() & df['dv_broadband'].notna()
        if m.sum() > 10:
            for grp in ['C', 'A', 'F']:
                gm = m & (df['group'] == grp)
                lab = GL.get(grp, grp)
                ax.scatter(df.loc[gm, 'mmse'], df.loc[gm, 'dv_broadband'],
                           c=GC.get(lab, 'gray'), s=60, alpha=0.7,
                           edgecolors='k', lw=0.5, label=lab)
            r, p = spearmanr(df.loc[m, 'mmse'], df.loc[m, 'dv_broadband'])
            z = np.polyfit(df.loc[m, 'mmse'], df.loc[m, 'dv_broadband'], 1)
            xfit = np.linspace(df.loc[m, 'mmse'].min(), df.loc[m, 'mmse'].max(), 50)
            ax.plot(xfit, np.polyval(z, xfit), 'k--', alpha=0.5)
            ax.set_xlabel('MMSE Score', fontsize=12)
            ax.set_ylabel('DV Gradient', fontsize=12)
            ax.set_title(f'C. Gradient vs Cognition\n(rho_s={r:.3f}, p={p:.4f})',
                         fontweight='bold')
            ax.legend(fontsize=10)

    # ---- Panels D1-D3: Band-specific ----
    for idx, band in enumerate(['theta', 'alpha', 'beta']):
        ax = axes[1, idx]
        gk = f'dv_{band}'
        if gk in df.columns:
            gdata = []
            for grp in ['C', 'A', 'F']:
                for v in df[df['group'] == grp][gk].dropna():
                    gdata.append({'Group': GL.get(grp, grp), 'Gradient': v})
            gdf3 = pd.DataFrame(gdata)
            if len(gdf3):
                sns.boxplot(data=gdf3, x='Group', y='Gradient', ax=ax,
                            order=['Control', 'AD', 'FTD'],
                            palette=[GC[g] for g in ['Control', 'AD', 'FTD']])
                sns.stripplot(data=gdf3, x='Group', y='Gradient', ax=ax,
                              order=['Control', 'AD', 'FTD'],
                              color='k', alpha=0.3, size=4)
                ax.axhline(0, color='k', ls='--', alpha=0.3)
                ax.set_title(f'D{idx+1}. {band.title()} band', fontweight='bold')
                ax.set_ylabel('rho(dorsal) - rho(ventral)', fontsize=12)

    plt.tight_layout()
    fig_path = out_dir / 'rho_gradient_AD_results.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure saved: {fig_path}")


if __name__ == '__main__':
    main()
