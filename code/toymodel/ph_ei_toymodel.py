import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import expm
import os, gc

os.makedirs('/home/claude/figs', exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 12, 'axes.linewidth': 1.2})
dt = 1/256

def ei_model(w_ei, w_ie, tau_e=0.02, tau_i=0.01, w_ee=5, w_ii=5):
    return np.array([[-1/tau_e + w_ee, -w_ei],
                     [w_ie, -1/tau_i - w_ii]])

def decompose(A):
    S = (A + A.T) / 2
    K = (A - A.T) / 2
    return S, K  # symmetric (dissipative), antisymmetric (conservative)

def get_rho_freq(A_c):
    A_d = expm(A_c * dt)
    evs = np.linalg.eigvals(A_d)
    rho = np.max(np.abs(evs))
    dom = evs[np.argmax(np.abs(evs))]
    freq = np.abs(np.angle(dom)) / (2*np.pi*dt) if np.abs(dom.imag) > 1e-10 else 0
    return rho, freq, evs

def impulse(A_c, n=600):
    A_d = expm(A_c * dt)
    x = np.array([1.0, 0.0])
    out = np.zeros((n, 2))
    for i in range(n):
        out[i] = x; x = A_d @ x
    return out

# ==============================================================
# MAIN 4-PANEL FIGURE
# ==============================================================
fig = plt.figure(figsize=(18, 16))

# --- A: ρ vs dissipation (tau_e), for different coupling strengths ---
ax1 = fig.add_subplot(2, 2, 1)
tau_range = np.linspace(0.005, 0.06, 100)
for w_coup, col, ls in [(20, '#1f77b4', '-'), (40, '#ff7f0e', '-'), 
                          (60, '#2ca02c', '-'), (80, '#d62728', '-')]:
    rhos = []
    for tau_e in tau_range:
        A_c = ei_model(w_coup, w_coup, tau_e=tau_e)
        r, _, _ = get_rho_freq(A_c)
        rhos.append(r)
    ax1.plot(tau_range * 1000, rhos, color=col, lw=2.5, label=f'w_EI = w_IE = {w_coup}')

ax1.set_xlabel('Excitatory time constant τ_E (ms)', fontsize=13)
ax1.set_ylabel('ρ  (eigenvalue modulus)', fontsize=13)
ax1.set_title('A.  ρ increases with τ_E (lower dissipation)',
              fontweight='bold', fontsize=14, color='#1a3c6e')
ax1.legend(fontsize=10, title='E-I coupling')
ax1.text(45, 0.63, 'Dorsal\n(fast τ, low ρ)', fontsize=11, color='#1f77b4', 
         fontweight='bold', ha='center')
ax1.text(10, 0.85, 'Ventral\n(slow τ, high ρ)', fontsize=11, color='#d62728',
         fontweight='bold', ha='center')

# --- B: ρ vs ||J||/||R|| ratio (the key relationship) ---
ax2 = fig.add_subplot(2, 2, 2)
jr_ratios, rho_vals = [], []
# Sweep both coupling and tau
for w_coup in np.linspace(10, 80, 20):
    for tau_e in np.linspace(0.008, 0.05, 20):
        A_c = ei_model(w_coup, w_coup, tau_e=tau_e)
        S, K = decompose(A_c)
        jr = np.linalg.norm(K) / (np.linalg.norm(S) + 1e-10)
        r, f, _ = get_rho_freq(A_c)
        jr_ratios.append(jr)
        rho_vals.append(r)

ax2.scatter(jr_ratios, rho_vals, c='#2e75b6', s=15, alpha=0.5, edgecolors='none')
# Fit trend
jr_arr, rho_arr = np.array(jr_ratios), np.array(rho_vals)
idx = np.argsort(jr_arr)
from numpy.polynomial import polynomial as P
coeffs = P.polyfit(jr_arr[idx], rho_arr[idx], 3)
xfit = np.linspace(jr_arr.min(), jr_arr.max(), 100)
yfit = P.polyval(xfit, coeffs)
ax2.plot(xfit, yfit, 'k-', lw=2.5, label='Polynomial fit')
r_corr = np.corrcoef(jr_arr, rho_arr)[0, 1]
ax2.text(0.1, 0.6, f'r = {r_corr:.3f}', fontsize=16, fontweight='bold', color='#2e75b6')
ax2.set_xlabel('||J|| / ||R||  (conservative / dissipative)', fontsize=13)
ax2.set_ylabel('ρ', fontsize=13)
ax2.set_title('B.  ρ tracks the J/R balance across parameters',
              fontweight='bold', fontsize=14, color='#1a3c6e')
ax2.legend(fontsize=10)

# --- C: Phase portraits at three regimes ---
ax3 = fig.add_subplot(2, 2, 3)
configs = [
    (20, 0.01, '#1f77b4', 'Dorsal-like: fast τ, low coupling'),
    (40, 0.02, '#ff7f0e', 'Intermediate'),
    (60, 0.04, '#d62728', 'Ventral-like: slow τ, high coupling'),
]
for w_c, tau_e, col, lab in configs:
    A_c = ei_model(w_c, w_c, tau_e=tau_e)
    rho, freq, _ = get_rho_freq(A_c)
    traj = impulse(A_c)
    ax3.plot(traj[:, 0], traj[:, 1], color=col, lw=1.5, alpha=0.8,
             label=f'{lab}\nρ={rho:.3f}')
    ax3.scatter(traj[0, 0], traj[0, 1], color=col, s=80, zorder=5, edgecolors='k')
ax3.set_xlabel('x_E  (excitatory)', fontsize=13)
ax3.set_ylabel('x_I  (inhibitory)', fontsize=13)
ax3.set_title('C.  Phase portraits: dorsal vs ventral regimes',
              fontweight='bold', fontsize=14, color='#1a3c6e')
ax3.legend(fontsize=9, loc='lower left')
ax3.axhline(0, color='k', lw=0.5, alpha=0.3)
ax3.axvline(0, color='k', lw=0.5, alpha=0.3)

# --- D: Eigenvalue spectrum in complex plane ---
ax4 = fig.add_subplot(2, 2, 4)
theta = np.linspace(0, 2*np.pi, 100)
ax4.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.15, lw=1.5)
ax4.axhline(0, color='k', lw=0.5, alpha=0.3)
ax4.axvline(0, color='k', lw=0.5, alpha=0.3)

# Plot eigenvalues for a sweep
for tau_e in np.linspace(0.008, 0.05, 8):
    for w_c in [20, 40, 60]:
        A_c = ei_model(w_c, w_c, tau_e=tau_e)
        _, _, evs = get_rho_freq(A_c)
        rho = np.max(np.abs(evs))
        # Color by rho
        color = plt.cm.coolwarm((rho - 0.5) / 0.5)
        for ev in evs:
            ax4.scatter(ev.real, ev.imag, c=[color], s=40, edgecolors='k', lw=0.3, zorder=5)

# Annotate
ax4.annotate('High ρ\n(ventral)', xy=(0.9, 0.15), fontsize=11, color='#d62728',
             fontweight='bold', ha='center')
ax4.annotate('Low ρ\n(dorsal)', xy=(0.55, 0.0), fontsize=11, color='#1f77b4',
             fontweight='bold', ha='center')
ax4.set_xlabel('Real', fontsize=13); ax4.set_ylabel('Imaginary', fontsize=13)
ax4.set_title('D.  Eigenvalue spectrum across regimes',
              fontweight='bold', fontsize=14, color='#1a3c6e')
ax4.set_aspect('equal')
ax4.set_xlim(0.3, 1.1); ax4.set_ylim(-0.4, 0.4)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0.5, 1.0))
sm.set_array([])
cb = plt.colorbar(sm, ax=ax4, label='ρ', shrink=0.7)

plt.suptitle('Figure 1:  E-I Circuit as a Port-Hamiltonian System\nρ reflects the balance of conservative (J) and dissipative (R) dynamics',
             fontsize=15, fontweight='bold', color='#1a3c6e', y=1.03)
plt.tight_layout()
plt.savefig('/home/claude/figs/fig_toymodel.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
gc.collect()
print("Main figure done!")

# ==============================================================
# VALIDATION: AR(2) recovery from simulated pH signals
# ==============================================================
print("Running validation...")
fig2, ax = plt.subplots(1, 1, figsize=(7, 6.5))
np.random.seed(42)
true_r, est_r = [], []

params = []
for w_c in np.linspace(15, 70, 8):
    for tau_e in np.linspace(0.008, 0.05, 8):
        params.append((w_c, tau_e))

for w_c, tau_e in params:
    A_c = ei_model(w_c, w_c, tau_e=tau_e)
    A_d = expm(A_c * dt)
    rho_true = np.max(np.abs(np.linalg.eigvals(A_d)))
    true_r.append(rho_true)
    
    # Simulate 10s noisy signal
    x = np.zeros((2560, 2))
    for t in range(1, 2560):
        x[t] = A_d @ x[t-1] + np.random.randn(2) * 0.05
    sig = x[:, 0]
    
    # AR(2) → roots → max modulus
    Y = sig[2:]
    X = np.column_stack([sig[1:-1], sig[:-2]])
    c = np.linalg.lstsq(X, Y, rcond=None)[0]
    roots = np.roots([1, -c[0], -c[1]])
    est_r.append(np.max(np.abs(roots)))
    del x, sig

true_r, est_r = np.array(true_r), np.array(est_r)
ax.scatter(true_r, est_r, c='#2e75b6', s=50, edgecolors='k', lw=0.5, alpha=0.8)
ax.plot([0.4, 1.0], [0.4, 1.0], 'k--', lw=1, alpha=0.5)
rc = np.corrcoef(true_r, est_r)[0, 1]
ax.text(0.45, 0.95, f'r = {rc:.3f}', fontsize=16, fontweight='bold', color='#2e75b6')
ax.set_xlabel('True ρ (from E-I circuit model)', fontsize=13)
ax.set_ylabel('Estimated ρ (AR(2) on observed signal)', fontsize=13)
ax.set_title('Validation: AR(2) recovers ρ from\nnoisy port-Hamiltonian E-I signals',
             fontweight='bold', fontsize=14, color='#1a3c6e')
ax.set_xlim(0.4, 1.0); ax.set_ylim(0.4, 1.0)
plt.tight_layout()
plt.savefig('/home/claude/figs/fig_validation.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Validation correlation: r = {rc:.4f}")
print("All done!")
