"""
Figure 3: Port-Hamiltonian Decomposition of AR(2) Dynamics
Key result: ||J||_F = (1 + rho^2) / sqrt(2) — pure function of rho.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, ttest_ind

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 13,
                     'axes.linewidth': 1.2, 'figure.facecolor': 'white'})

freq_centers = {'delta': 2.5, 'theta': 6, 'alpha': 10.5, 'beta': 21, 'broadband': 15}
bands = ['delta', 'theta', 'alpha', 'beta', 'broadband']
regions = ['dorsal', 'ventral_temporal', 'lateral_temporal', 'frontal',
           'frontopolar', 'parietal', 'occipital']
dv_pos = {'dorsal': 1.0, 'frontal': 0.75, 'parietal': 0.6,
          'frontopolar': 0.5, 'lateral_temporal': 0.3,
          'occipital': 0.2, 'ventral_temporal': 0.0}

def jr_ratio(rho, fhz, fs=500):
    omega = 2*np.pi*fhz/fs
    a1 = 2*rho*np.cos(omega)
    jn = (1+rho**2)/np.sqrt(2)
    sn = np.sqrt(a1**2 + (1-rho**2)**2/2)
    return jn/(sn+1e-15)

def diss_frac(rho, fhz, fs=500):
    omega = 2*np.pi*fhz/fs
    a1 = 2*rho*np.cos(omega)
    jn = (1+rho**2)/np.sqrt(2)
    sn = np.sqrt(a1**2 + (1-rho**2)**2/2)
    return sn/(jn+sn)

df = pd.read_csv('/home/claude/pH_cortical_dynamics/data/discovery_ds004504/rho_gradient_results.csv')
controls = df[df['group']=='C']
ad = df[df['group']=='A']

c_ctrl, c_ad = '#3498db', '#e74c3c'
rho_r = np.linspace(0.5, 0.999, 500)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# ===== PANEL A: Analytical norms =====
ax = axes[0,0]
ax.plot(rho_r, (1+rho_r**2)/np.sqrt(2), 'b-', lw=4,
        label=r'$\|\hat{J}\|_F = (1+\rho^2)/\sqrt{2}$')

fcolors = [('#d95f02','Alpha 10.5 Hz'), ('#e7298a','Beta 21 Hz'),
           ('#66a61e','Theta 6 Hz'), ('#7570b3','Delta 2.5 Hz')]
fhzs = [10.5, 21, 6, 2.5]
for (col,lab), fhz in zip(fcolors, fhzs):
    omega = 2*np.pi*fhz/500
    a1v = 2*rho_r*np.cos(omega)
    sn = np.sqrt(a1v**2 + (1-rho_r**2)**2/2)
    ax.plot(rho_r, sn, '--', lw=2.5, color=col, label=r'$\|\hat{S}\|_F$  '+lab)

ax.set_xlabel(r'$\rho$', fontsize=16)
ax.set_ylabel('Frobenius norm', fontsize=15)
ax.set_title('A.   Companion matrix decomposition', fontweight='bold', fontsize=16, color='#1a3c6e')
ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
ax.set_xlim(0.5, 1.0); ax.set_ylim(0.85, 2.1)
ax.tick_params(labelsize=13)
ax.text(0.97, 0.05, r'$\|\hat{J}\|$ depends only on $\rho$'+'\n(no free parameters)',
        transform=ax.transAxes, fontsize=12, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='#e8f0fe', ec='#2e75b6'))

# ===== PANEL B: Dissipative fraction =====
ax = axes[0,1]
for (col,lab), fhz in zip(fcolors, fhzs):
    df_vals = np.array([diss_frac(r, fhz) for r in rho_r])
    ax.plot(rho_r, df_vals, '-', lw=3, color=col, label=lab)

ax.set_xlabel(r'$\rho$', fontsize=16)
ax.set_ylabel(r'Dissipative fraction  $\|\hat{S}\| / (\|\hat{J}\|+\|\hat{S}\|)$', fontsize=14)
ax.set_title(r'B.   Higher $\rho$ = lower dissipative fraction', fontweight='bold', fontsize=16, color='#1a3c6e')
ax.legend(fontsize=12, loc='upper right')
ax.set_xlim(0.5, 1.0); ax.set_ylim(0, 0.65)
ax.tick_params(labelsize=13)
ax.text(0.52, 0.58, 'R-dominated\n(dissipative)', fontsize=14, fontweight='bold', color='#1f77b4')
ax.text(0.85, 0.06, 'J-dominated\n(oscillatory)', fontsize=14, fontweight='bold', color='#d62728')

# ===== PANEL C: Alpha J/R gradient across cortex =====
ax = axes[1,0]
for gdf, glab, col, mk in [(controls,'Controls (n=29)',c_ctrl,'o'),
                             (ad,'AD (n=36)',c_ad,'s')]:
    pos, means, sems = [], [], []
    for reg in regions:
        rv = gdf[f'rho_alpha_{reg}'].dropna().values
        jrs = np.array([jr_ratio(r, 10.5) for r in rv])
        pos.append(dv_pos[reg]); means.append(np.mean(jrs)); sems.append(np.std(jrs)/np.sqrt(len(jrs)))
    pos, means, sems = np.array(pos), np.array(means), np.array(sems)
    o = np.argsort(pos)
    ax.errorbar(pos[o], means[o], yerr=sems[o], fmt=f'{mk}-', color=col,
                capsize=6, capthick=2, markersize=10, linewidth=2.5, label=glab)

ax.set_xlabel('Dorsoventral position (0=ventral, 1=dorsal)', fontsize=15)
ax.set_ylabel(r'$\|\hat{J}\| \;/\; \|\hat{S}\|$   (alpha band)', fontsize=14)
ax.set_title(r'C.   AD flattens the $\hat{J}/\hat{R}$ gradient', fontweight='bold', fontsize=16, color='#1a3c6e')
ax.legend(fontsize=13, loc='upper left')
ax.tick_params(labelsize=13)

# ===== PANEL D: DV gradient by band =====
ax = axes[1,1]
x = np.arange(5); w = 0.35
for gi, (gdf, glab, col) in enumerate([(controls,'Controls',c_ctrl),(ad,'AD',c_ad)]):
    gm, gs_ = [], []
    for band in bands:
        dr = gdf[f'rho_{band}_dorsal'].dropna().values
        vr = gdf[f'rho_{band}_ventral_temporal'].dropna().values
        n = min(len(dr),len(vr))
        g = [jr_ratio(dr[i],freq_centers[band]) - jr_ratio(vr[i],freq_centers[band]) for i in range(n)]
        gm.append(np.mean(g)); gs_.append(np.std(g)/np.sqrt(len(g)))
    ax.bar(x-w/2+gi*w, gm, w, yerr=gs_, color=col, alpha=0.85, edgecolor='white', capsize=5, label=glab)

ax.axhline(0, color='gray', lw=0.5)
ax.set_xticks(x); ax.set_xticklabels(['Delta','Theta','Alpha','Beta','Broadband'], fontsize=13)
ax.set_ylabel(r'DV $\|\hat{J}\|/\|\hat{S}\|$ gradient'+'\n(dorsal $-$ ventral)', fontsize=14)
ax.set_title('D.   J/R gradient disruption by band', fontweight='bold', fontsize=16, color='#1a3c6e')
ax.legend(fontsize=13); ax.tick_params(labelsize=13)
ax.text(0.02, 0.95, 'Negative = ventral more J-dominated\n(healthy organization)',
        transform=ax.transAxes, fontsize=11, va='top',
        bbox=dict(boxstyle='round', fc='wheat', alpha=0.85))

plt.suptitle('Figure 3:  Port-Hamiltonian Decomposition of Cortical EEG Dynamics',
             fontsize=18, fontweight='bold', color='#1a3c6e', y=1.005)
plt.tight_layout()
plt.savefig('/home/claude/pH_cortical_dynamics/figures/main/Fig3_JR_decomposition.png',
            dpi=250, bbox_inches='tight', facecolor='white')
plt.savefig('/home/claude/pH_cortical_dynamics/figures/main/Fig3_JR_decomposition.pdf',
            bbox_inches='tight', facecolor='white')
print("Figure saved!")

# Stats
print("\nWithin-band rho vs dissipative fraction (controls):")
for band in bands:
    rhos, dfs = [], []
    for reg in regions:
        for r in controls[f'rho_{band}_{reg}'].dropna():
            rhos.append(r); dfs.append(diss_frac(r, freq_centers[band]))
    print(f"  {band:12s}: r = {pearsonr(rhos,dfs)[0]:.4f}")

d_c, v_c = controls['rho_alpha_dorsal'].dropna().values, controls['rho_alpha_ventral_temporal'].dropna().values
d_a, v_a = ad['rho_alpha_dorsal'].dropna().values, ad['rho_alpha_ventral_temporal'].dropna().values
gc = [jr_ratio(d_c[i],10.5)-jr_ratio(v_c[i],10.5) for i in range(min(len(d_c),len(v_c)))]
ga = [jr_ratio(d_a[i],10.5)-jr_ratio(v_a[i],10.5) for i in range(min(len(d_a),len(v_a)))]
t,p = ttest_ind(gc,ga)
d = (np.mean(gc)-np.mean(ga))/np.sqrt((np.var(gc)+np.var(ga))/2)
print(f"\nAlpha J/R gradient: Controls vs AD: t={t:.3f}, p={p:.6f}, d={d:.3f}")
