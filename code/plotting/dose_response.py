import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.stats import spearmanr, pearsonr, linregress

df = pd.read_csv('/mnt/user-data/uploads/rho_gradient_results.csv')

# Group colors and labels
colors = {'A': '#d62728', 'C': '#2ca02c', 'F': '#1f77b4'}
labels = {'A': 'AD', 'C': 'Control', 'F': 'FTD'}
markers = {'A': 'o', 'C': 's', 'F': '^'}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# ---- Row 1: DV gradient vs MMSE for key bands ----
bands = ['alpha', 'beta', 'broadband']
band_titles = ['Alpha (8-13 Hz)', 'Beta (13-30 Hz)', 'Broadband (1-45 Hz)']

for ax, band, title in zip(axes[0], bands, band_titles):
    gk = f'dv_{band}'
    for grp in ['C', 'F', 'A']:
        mask = df['group'] == grp
        ax.scatter(df.loc[mask, 'mmse'], df.loc[mask, gk] * 1e4,
                   c=colors[grp], label=labels[grp], marker=markers[grp],
                   s=50, alpha=0.7, edgecolors='white', linewidth=0.5)

    # Overall regression line
    valid = df[['mmse', gk]].dropna()
    slope, intercept, r_val, p_val, _ = linregress(valid['mmse'], valid[gk] * 1e4)
    x_line = np.linspace(valid['mmse'].min(), valid['mmse'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5, linewidth=1.5)

    # Spearman
    rs, ps = spearmanr(valid['mmse'], valid[gk])
    ax.set_title(f'{title}', fontsize=13, fontweight='bold')
    ax.set_xlabel('MMSE Score', fontsize=11)
    ax.set_ylabel(f'DV Gradient (ρ × 10⁴)', fontsize=11)
    ax.text(0.05, 0.95, f'ρₛ = {rs:.3f}\np = {ps:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.legend(fontsize=9, loc='lower right')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

# ---- Row 2 Left: Per-region rho vs MMSE (alpha band) ----
ax = axes[1, 0]
region_keys = ['rho_alpha_dorsal', 'rho_alpha_parietal', 'rho_alpha_frontal',
               'rho_alpha_lateral_temporal', 'rho_alpha_ventral_temporal', 'rho_alpha_occipital']
region_labels = ['Dorsal', 'Parietal', 'Frontal', 'Lat. Temporal', 'Vent. Temporal', 'Occipital']
region_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(region_keys)))

for rk, rl, rc in zip(region_keys, region_labels, region_colors):
    valid = df[['mmse', rk]].dropna()
    rs, ps = spearmanr(valid['mmse'], valid[rk])
    slope, intercept, _, _, _ = linregress(valid['mmse'], valid[rk])
    x_line = np.linspace(valid['mmse'].min(), valid['mmse'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color=rc, linewidth=2,
            label=f'{rl} (ρₛ={rs:.3f})', alpha=0.8)

ax.set_title('Alpha ρ vs MMSE by Region', fontsize=13, fontweight='bold')
ax.set_xlabel('MMSE Score', fontsize=11)
ax.set_ylabel('ρ (alpha band)', fontsize=11)
ax.legend(fontsize=8, loc='lower right')

# ---- Row 2 Middle: Box plot of DV gradient by group ----
ax = axes[1, 1]
data_by_group = [df[df['group'] == g]['dv_alpha'].dropna() * 1e4 for g in ['C', 'F', 'A']]
bp = ax.boxplot(data_by_group, labels=['Control', 'FTD', 'AD'],
                patch_artist=True, widths=0.6)
box_colors = [colors['C'], colors['F'], colors['A']]
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
# Overlay individual points
for i, (grp, data) in enumerate(zip(['C', 'F', 'A'], data_by_group)):
    x = np.random.normal(i + 1, 0.08, size=len(data))
    ax.scatter(x, data, c=colors[grp], alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_title('Alpha DV Gradient by Group', fontsize=13, fontweight='bold')
ax.set_ylabel('DV Gradient (ρ × 10⁴)', fontsize=11)

# ---- Row 2 Right: Global mean rho vs MMSE ----
ax = axes[1, 2]
region_cols = [c for c in df.columns if c.startswith('rho_alpha_')]
df['global_rho_alpha'] = df[region_cols].mean(axis=1)

for grp in ['C', 'F', 'A']:
    mask = df['group'] == grp
    ax.scatter(df.loc[mask, 'mmse'], df.loc[mask, 'global_rho_alpha'],
               c=colors[grp], label=labels[grp], marker=markers[grp],
               s=50, alpha=0.7, edgecolors='white', linewidth=0.5)

valid = df[['mmse', 'global_rho_alpha']].dropna()
rs, ps = spearmanr(valid['mmse'], valid['global_rho_alpha'])
slope, intercept, _, _, _ = linregress(valid['mmse'], valid['global_rho_alpha'])
x_line = np.linspace(valid['mmse'].min(), valid['mmse'].max(), 100)
ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5, linewidth=1.5)
ax.text(0.05, 0.05, f'ρₛ = {rs:.3f}\np = {ps:.4f}',
        transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_title('Global Alpha ρ vs MMSE', fontsize=13, fontweight='bold')
ax.set_xlabel('MMSE Score', fontsize=11)
ax.set_ylabel('Mean ρ (alpha, all regions)', fontsize=11)
ax.legend(fontsize=9, loc='upper left')

plt.suptitle('Dose-Response: ρ Gradient Tracks Cognitive Severity\n(Discovery Cohort, ds004504)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/dose_response_mmse.png', dpi=300, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/dose_response_mmse.pdf', bbox_inches='tight')
print("Done")

# Print detailed stats
print("\n=== DOSE-RESPONSE STATISTICS ===")
for band in ['delta', 'theta', 'alpha', 'beta', 'broadband']:
    gk = f'dv_{band}'
    valid = df[['mmse', gk]].dropna()
    rs, ps = spearmanr(valid['mmse'], valid[gk])
    rp, pp = pearsonr(valid['mmse'], valid[gk])
    print(f"\n{band.upper()} DV gradient vs MMSE (N={len(valid)}):")
    print(f"  Spearman: rho={rs:.4f}, p={ps:.6f}")
    print(f"  Pearson:  r={rp:.4f},   p={pp:.6f}")

    # Within AD only
    ad = df[df['group'] == 'A'][['mmse', gk]].dropna()
    if len(ad) >= 5:
        rs_ad, ps_ad = spearmanr(ad['mmse'], ad[gk])
        print(f"  Within AD only (n={len(ad)}): rho={rs_ad:.4f}, p={ps_ad:.6f}")

# Global rho vs MMSE
print("\nGLOBAL ALPHA RHO vs MMSE:")
valid = df[['mmse', 'global_rho_alpha']].dropna()
rs, ps = spearmanr(valid['mmse'], valid['global_rho_alpha'])
print(f"  All subjects (N={len(valid)}): rho={rs:.4f}, p={ps:.6f}")
for grp, lab in [('A', 'AD'), ('C', 'Control'), ('F', 'FTD')]:
    sub = df[df['group'] == grp][['mmse', 'global_rho_alpha']].dropna()
    if len(sub) >= 5:
        rs_g, ps_g = spearmanr(sub['mmse'], sub['global_rho_alpha'])
        print(f"  {lab} only (n={len(sub)}): rho={rs_g:.4f}, p={ps_g:.6f}")
