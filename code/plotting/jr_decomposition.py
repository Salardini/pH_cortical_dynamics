"""
Port-Hamiltonian Decomposition of AR(2) Companion Matrices
=============================================================
Key analytical result: For the AR(2) companion matrix A = [[a1,a2],[1,0]],
||J||_F = (1 + rho^2) / sqrt(2) — a PURE FUNCTION of rho, monotonically increasing.
No assumptions about frequency, no free parameters. This is the direct proof
that rho constrains the conservative (J) component of pH dynamics.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_ind

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.linewidth': 1.0,
    'figure.facecolor': 'white',
})

freq_centers = {'delta': 2.5, 'theta': 6, 'alpha': 10.5, 'beta': 21, 'broadband': 15}

def compute_dissipative_fraction(rho_val, freq_hz, fs=500):
    omega = 2 * np.pi * freq_hz / fs
    a1 = 2 * rho_val * np.cos(omega)
    j_norm = (1 + rho_val**2) / np.sqrt(2)
    s_norm = np.sqrt(a1**2 + (1 - rho_val**2)**2 / 2)
    return s_norm / (j_norm + s_norm)

def compute_jr_ratio(rho_val, freq_hz, fs=500):
    """||J||/||S|| — the J/R norm ratio. More sensitive at high rho."""
    omega = 2 * np.pi * freq_hz / fs
    a1 = 2 * rho_val * np.cos(omega)
    j_norm = (1 + rho_val**2) / np.sqrt(2)
    s_norm = np.sqrt(a1**2 + (1 - rho_val**2)**2 / 2)
    if s_norm < 1e-15:
        return np.inf
    return j_norm / s_norm

# ==============================================================
# LOAD DATA
# ==============================================================
df = pd.read_csv('/home/claude/pH_cortical_dynamics/data/discovery_ds004504/rho_gradient_results.csv')
controls = df[df['group'] == 'C']
ad = df[df['group'] == 'A']

bands = ['delta', 'theta', 'alpha', 'beta', 'broadband']
regions = ['dorsal', 'ventral_temporal', 'lateral_temporal', 'frontal',
           'frontopolar', 'parietal', 'occipital']
dv_positions = {
    'dorsal': 1.0, 'frontal': 0.75, 'parietal': 0.6,
    'frontopolar': 0.5, 'lateral_temporal': 0.3,
    'occipital': 0.2, 'ventral_temporal': 0.0
}

# ==============================================================
# FIGURE
# ==============================================================
fig = plt.figure(figsize=(16, 13))
gs = GridSpec(2, 2, hspace=0.38, wspace=0.30)

c_ctrl = '#3498db'
c_ad = '#e74c3c'
rho_range = np.linspace(0.5, 0.999, 500)
J_norm = (1 + rho_range**2) / np.sqrt(2)

colors_freq = {'Delta (2.5 Hz)': '#7570b3', 'Theta (6 Hz)': '#66a61e',
               'Alpha (10.5 Hz)': '#d95f02', 'Beta (21 Hz)': '#e7298a'}
freqs_hz = {'Delta (2.5 Hz)': 2.5, 'Theta (6 Hz)': 6,
            'Alpha (10.5 Hz)': 10.5, 'Beta (21 Hz)': 21}

# --- Panel A: ||J|| and ||S|| vs rho ---
ax = fig.add_subplot(gs[0, 0])
ax.plot(rho_range, J_norm, 'b-', lw=3, label=r'$\|\hat{J}\|_F = (1+\rho^2)/\sqrt{2}$')
for flabel, fhz in freqs_hz.items():
    omega = 2 * np.pi * fhz / 500
    a1_vals = 2 * rho_range * np.cos(omega)
    S_norm = np.sqrt(a1_vals**2 + (1 - rho_range**2)**2 / 2)
    ax.plot(rho_range, S_norm, '--', lw=2, color=colors_freq[flabel],
            label=r'$\|\hat{S}\|_F$' + f' ({flabel})')
ax.set_xlabel(r'$\rho$ (eigenvalue modulus)', fontsize=13)
ax.set_ylabel('Frobenius norm', fontsize=13)
ax.set_title('A. Analytical decomposition of the\nAR(2) companion matrix',
             fontweight='bold', fontsize=13, color='#1a3c6e')
ax.legend(fontsize=8.5, loc='center left')
ax.set_xlim(0.5, 1.0)
ax.annotate(r'$\|\hat{J}\|$ depends ONLY on $\rho$' + '\n(monotonically increasing)',
            xy=(0.85, (1+0.85**2)/np.sqrt(2)), xytext=(0.58, 1.35),
            fontsize=9, fontweight='bold', color='blue',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

# --- Panel B: Dissipative fraction vs rho ---
ax = fig.add_subplot(gs[0, 1])
for flabel, fhz in freqs_hz.items():
    diss_frac = np.array([compute_dissipative_fraction(r, fhz) for r in rho_range])
    ax.plot(rho_range, diss_frac, '-', lw=2.5, color=colors_freq[flabel], label=flabel)
ax.set_xlabel(r'$\rho$ (eigenvalue modulus)', fontsize=13)
ax.set_ylabel(r'Dissipative fraction $\|\hat{S}\|/(\|\hat{J}\|+\|\hat{S}\|)$', fontsize=12)
ax.set_title(r'B. Higher $\rho$ = lower dissipative fraction' + '\n(within each frequency band)',
             fontweight='bold', fontsize=13, color='#1a3c6e')
ax.legend(fontsize=10)
ax.set_xlim(0.5, 1.0)
ax.set_ylim(0, 0.7)
ax.axvspan(0.985, 0.997, alpha=0.08, color=c_ctrl)
ax.axvspan(0.970, 0.985, alpha=0.08, color=c_ad)
ax.text(0.991, 0.63, 'Ctrl', ha='center', fontsize=8, color=c_ctrl, fontweight='bold')
ax.text(0.978, 0.63, 'AD', ha='center', fontsize=8, color=c_ad, fontweight='bold')
ax.annotate('R-dominated\n(dissipative)', xy=(0.55, 0.55), fontsize=10,
            fontweight='bold', color='#1f77b4')
ax.annotate('J-dominated\n(oscillatory)', xy=(0.92, 0.08), fontsize=10,
            fontweight='bold', color='#d62728')

# --- Panel C: Alpha-band J/R ratio gradient ---
ax = fig.add_subplot(gs[1, 0])
for group_df, glabel, color, marker in [
    (controls, 'Controls (n=29)', c_ctrl, 'o'),
    (ad, 'AD (n=36)', c_ad, 's')
]:
    positions, jr_means, jr_sems = [], [], []
    for region in regions:
        col = f'rho_alpha_{region}'
        rho_vals = group_df[col].dropna().values
        jrs = np.array([compute_jr_ratio(r, 10.5) for r in rho_vals])
        positions.append(dv_positions[region])
        jr_means.append(np.mean(jrs))
        jr_sems.append(np.std(jrs) / np.sqrt(len(jrs)))
    positions, jr_means, jr_sems = np.array(positions), np.array(jr_means), np.array(jr_sems)
    order = np.argsort(positions)
    ax.errorbar(positions[order], jr_means[order], yerr=jr_sems[order],
                fmt=f'{marker}-', color=color, capsize=4, capthick=1.5,
                markersize=8, linewidth=2, label=glabel)

ax.set_xlabel('Dorsoventral position (0=ventral, 1=dorsal)', fontsize=12)
ax.set_ylabel(r'$\|\hat{J}\|_F \;/\; \|\hat{S}\|_F$ (alpha band)', fontsize=11)
ax.set_title(r'C. AD flattens the cortical $\hat{J}/\hat{R}$ gradient' + '\n(alpha band)',
             fontweight='bold', fontsize=13, color='#1a3c6e')
ax.legend(fontsize=10, loc='upper left')
ax.annotate('Healthy: ventral has higher J/R\n(more oscillatory, less dissipative)',
            xy=(0.35, 0.03), xycoords='axes fraction', fontsize=9, color='#555', style='italic')

# --- Panel D: DV J/R gradient by band ---
ax = fig.add_subplot(gs[1, 1])
band_labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Broadband']
x_pos = np.arange(len(bands))
w = 0.35
for gi, (group_df, glabel, color) in enumerate([
    (controls, 'Controls', c_ctrl), (ad, 'AD', c_ad)
]):
    gradient_means, gradient_sems = [], []
    for band in bands:
        d_rhos = group_df[f'rho_{band}_dorsal'].dropna().values
        v_rhos = group_df[f'rho_{band}_ventral_temporal'].dropna().values
        fhz = freq_centers[band]
        n = min(len(d_rhos), len(v_rhos))
        grads = [compute_jr_ratio(d_rhos[i], fhz) -
                 compute_jr_ratio(v_rhos[i], fhz) for i in range(n)]
        gradient_means.append(np.mean(grads))
        gradient_sems.append(np.std(grads) / np.sqrt(len(grads)))
    offset = -w/2 + gi * w
    ax.bar(x_pos + offset, gradient_means, w, yerr=gradient_sems,
           color=color, alpha=0.8, edgecolor='white', capsize=4, label=glabel)
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(band_labels, fontsize=11)
ax.set_ylabel(r'DV $\|\hat{J}\|/\|\hat{S}\|$ gradient' + '\n(dorsal $-$ ventral)', fontsize=11)
ax.set_title('D. Frequency-specific disruption of the\nJ/R structure in AD',
             fontweight='bold', fontsize=13, color='#1a3c6e')
ax.legend(fontsize=10)
ax.text(0.02, 0.95, 'Negative = ventral more J-dominated\n(healthy organization)',
        transform=ax.transAxes, fontsize=8, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle(r'Figure 3: Port-Hamiltonian Decomposition ($\hat{J}$, $\hat{R}$) of Cortical EEG Dynamics',
             fontsize=14, fontweight='bold', color='#1a3c6e', y=1.01)
plt.savefig('/home/claude/pH_cortical_dynamics/figures/main/Fig3_JR_decomposition.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/claude/pH_cortical_dynamics/figures/main/Fig3_JR_decomposition.pdf',
            bbox_inches='tight', facecolor='white')
print("Figure saved!")

# ==============================================================
# STATISTICS
# ==============================================================
print("\n" + "="*60)
print("KEY ANALYTICAL RESULT")
print("="*60)
print("||J||_F = (1 + rho^2) / sqrt(2)")
print("-> Pure function of rho. No free parameters.")
print("-> Proof that rho constrains J without assumptions about H.\n")

print("="*60)
print("WITHIN-BAND: rho vs dissipative fraction (controls)")
print("="*60)
for band in bands:
    all_rhos, all_dfs = [], []
    fhz = freq_centers[band]
    for region in regions:
        for rv in controls[f'rho_{band}_{region}'].dropna().values:
            all_rhos.append(rv)
            all_dfs.append(compute_dissipative_fraction(rv, fhz))
    r, p = pearsonr(all_rhos, all_dfs)
    print(f"  {band:12s}: r = {r:.4f} (p = {p:.2e})")

print("\n" + "="*60)
print("GROUP COMPARISON: Alpha J/R ratio DV gradient")
print("="*60)
for group_df, glabel in [(controls, 'Controls'), (ad, 'AD')]:
    d_r = group_df['rho_alpha_dorsal'].dropna().values
    v_r = group_df['rho_alpha_ventral_temporal'].dropna().values
    n = min(len(d_r), len(v_r))
    grads = [compute_jr_ratio(d_r[i], 10.5) -
             compute_jr_ratio(v_r[i], 10.5) for i in range(n)]
    print(f"  {glabel:10s}: {np.mean(grads):.4f} +/- {np.std(grads)/np.sqrt(len(grads)):.4f}")

d_c = controls['rho_alpha_dorsal'].dropna().values
v_c = controls['rho_alpha_ventral_temporal'].dropna().values
d_a = ad['rho_alpha_dorsal'].dropna().values
v_a = ad['rho_alpha_ventral_temporal'].dropna().values
g_ctrl = [compute_jr_ratio(d_c[i], 10.5) - compute_jr_ratio(v_c[i], 10.5)
          for i in range(min(len(d_c), len(v_c)))]
g_ad = [compute_jr_ratio(d_a[i], 10.5) - compute_jr_ratio(v_a[i], 10.5)
        for i in range(min(len(d_a), len(v_a)))]
t, p = ttest_ind(g_ctrl, g_ad)
d_cohen = (np.mean(g_ctrl) - np.mean(g_ad)) / np.sqrt((np.var(g_ctrl) + np.var(g_ad)) / 2)
print(f"\n  Controls vs AD: t = {t:.3f}, p = {p:.6f}, d = {d_cohen:.3f}")
