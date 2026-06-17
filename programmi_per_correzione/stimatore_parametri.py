import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# poi modificato da me ops, sta nel mio terminale perché non prendeva quando lho fatto, poi lo caricherò...
 
# ==============================================================================
# DATI SPERIMENTALI -- SOSTITUISCI QUI CON I TUOI DATI REALI
# ==============================================================================
 
freq = np.array([200.0, 500.0, 2000.0, 5000.0, 10000.0, 25000.0, 50000.0,
                  100000.0, 150000.0])  # [Hz]
 
V_in_meas = np.array([10.6121, 11.0312, 10.8983, 10.4859, 10.1983, 10.0466,
                       9.9467, 10.0159, 10.0179])  # [V]
sigma_V_in = np.array([0.0217, 0.0152, 0.0310, 0.0216, 0.0170, 0.0245,
                        0.0158, 0.0149, 0.0235])  # [V]
 
V_GEN = 10.0  # valore vero noto del generatore [V]
 
# Valori noti dall'analisi nel dominio del tempo (onda quadra), usati SOLO
# come punto di partenza (p0) del fit -- il fit resta libero di muoversi.
SR_noto_tempo_di_salita_ns = 20      # [ns]
Overshoot_noto_pct = 15.0            # [%]
Droop_noto_pct_per_ms = 3.0         # [%/ms]
 
# Numero di campioni Monte Carlo per la propagazione delle incertezze
N_MC = 20000
 
# ==============================================================================
# MODELLO DI FUNZIONE DI TRASFERIMENTO (droop + 2deg ordine)
# ==============================================================================
 
def H_model(f, f_d, f_n, zeta):
    s_norm_d = 1j * f / f_d
    H_droop = s_norm_d / (1.0 + s_norm_d)
 
    x = f / f_n
    H_2ord = 1.0 / (1.0 - x**2 + 1j * 2.0 * zeta * x)
 
    return H_droop * H_2ord
 
 
def model_amp(f, f_d, f_n, zeta):
    return np.abs(H_model(f, f_d, f_n, zeta))
 
 
# ==============================================================================
# PUNTO DI PARTENZA (p0) DERIVATO DAI VALORI NOTI DALL'ONDA QUADRA
# ==============================================================================
 
# Da overshoot noto -> zeta iniziale
os_frac = Overshoot_noto_pct / 100.0
zeta0 = np.sqrt(np.log(os_frac)**2 / (np.pi**2 + np.log(os_frac)**2))
 
# Da tempo di salita noto -> f_n iniziale (formula standard, Ogata)
t_rise0 = SR_noto_tempo_di_salita_ns * 1e-9
wn_tr0 = 1.76 * zeta0**3 - 0.417 * zeta0**2 + 1.039 * zeta0 + 1.0
wn0 = wn_tr0 / t_rise0
f_n0 = wn0 / (2 * np.pi)
 
# Da droop noto [%/ms] -> f_d iniziale.
# Droop%/ms ~ derivata di H_droop verso f->0 valutata su una scala temporale
# di 1 ms; approssimazione: tau_droop tale che in 1 ms la caduta sia
# Droop_noto_pct_per_ms %. Per H_droop(t) ~ exp(-t/tau_droop) (decadimento
# esponenziale dello zero passa-alto), Droop%/ms = 100*(1-exp(-1ms/tau_droop)).
droop_frac_per_ms = Droop_noto_pct_per_ms / 100.0
tau_droop0 = -1e-3 / np.log(max(1.0 - droop_frac_per_ms, 1e-9))
f_d0 = 1.0 / (2 * np.pi * tau_droop0)
 
print("=" * 70)
print("PUNTO DI PARTENZA (p0) DERIVATO DAI VALORI NOTI (ONDA QUADRA)")
print("=" * 70)
print(f"  f_d0 (da droop)     = {f_d0:.4g} Hz")
print(f"  f_n0 (da SR)        = {f_n0:.4g} Hz  ({f_n0/1e9:.3g} GHz)")
print(f"  zeta0 (da overshoot)= {zeta0:.4g}")
print("  ATTENZIONE: f_n0 e' ~9000 volte oltre la massima frequenza misurata")
print("  (150 kHz). Il fit verra' lasciato libero, ma e' atteso un forte")
print("  spostamento di f_n lontano da questo valore iniziale, dato che i")
print("  dati a queste frequenze non contengono informazione su una")
print("  dinamica così rapida (vedi nota nel docstring iniziale).")
print("=" * 70)
 
# ==============================================================================
# FIT NON LINEARE PESATO (solo ampiezza; la fase di V_in e' pure rumore,
# vedi analisi preliminare: valori ~1e-7 rad senza alcun trend, mentre
# l'incertezza e' ~2e-7 rad)
# ==============================================================================
 
H_meas_amp = V_in_meas / V_GEN
sigma_amp = sigma_V_in / V_GEN
 
p0 = [f_d0, f_n0, zeta0]
# bounds larghi ma fisicamente non negativi; f_n puo' arrivare fino a GHz
bounds = ([1e-6, 1.0, 1e-3], [freq.min(), 1e10, 3.0])
 
try:
    popt, pcov = curve_fit(
        model_amp, freq, H_meas_amp,
        p0=p0, sigma=sigma_amp, absolute_sigma=True,
        bounds=bounds, maxfev=50000
    )
    fit_ok = True
except Exception as e:
    print(f"\n  ATTENZIONE: il fit non e' andato a buon fine: {e}")
    popt, pcov = p0, np.eye(3) * np.nan
    fit_ok = False
 
f_d_fit, f_n_fit, zeta_fit = popt
perr = np.sqrt(np.abs(np.diag(pcov)))
f_d_err, f_n_err, zeta_err = perr
 
pred = model_amp(freq, *popt)
residuals = (H_meas_amp - pred) / sigma_amp
chi2 = np.sum(residuals**2)
dof = len(H_meas_amp) - len(popt)
 
print("\n" + "=" * 70)
print("RISULTATO DEL FIT")
print("=" * 70)
print(f"  f_d   = {f_d_fit:.4g} +/- {f_d_err:.4g} Hz")
print(f"  f_n   = {f_n_fit:.4g} +/- {f_n_err:.4g} Hz")
print(f"  zeta  = {zeta_fit:.4g} +/- {zeta_err:.4g}")
print(f"\n  chi2 / dof = {chi2:.2f} / {dof} = {chi2/dof:.2f}")
if chi2 / dof > 5:
    print("  ATTENZIONE: chi2/dof >> 1: il modello NON descrive bene i dati.")
    print("  I parametri seguenti vanno quindi interpretati con cautela: il")
    print("  fit ha trovato un minimo locale ma l'accordo numerico e' scarso.")
print("=" * 70)
 
print("\n" + "=" * 70)
print("CONFRONTO DATI vs MODELLO (trasparenza sulla qualita' del fit)")
print("=" * 70)
print(f"{'f [Hz]':>10s}  {'dato |H|':>10s}  {'modello |H|':>12s}  {'residuo [sigma]':>16s}")
for f, d, p, r in zip(freq, H_meas_amp, pred, residuals):
    print(f"{f:>10.0f}  {d:>10.4f}  {p:>12.4f}  {r:>16.2f}")
print("=" * 70)
 
# ==============================================================================
# PROPAGAZIONE MONTE CARLO VERSO LE GRANDEZZE FINALI
# ==============================================================================
 
rng = np.random.default_rng(42)
samples = rng.multivariate_normal(popt, pcov, size=N_MC) if fit_ok else np.tile(popt, (N_MC, 1))
f_d_s, f_n_s, zeta_s = samples[:, 0], samples[:, 1], samples[:, 2]
 
valid = (zeta_s > 0) & (f_n_s > 0) & (f_d_s > 0)
f_d_s, f_n_s, zeta_s = f_d_s[valid], f_n_s[valid], zeta_s[valid]
 
# --- OVERSHOOT % ---
overshoot_s = np.where(
    zeta_s < 1.0,
    100.0 * np.exp(-np.pi * zeta_s / np.sqrt(np.clip(1 - zeta_s**2, 1e-12, None))),
    0.0
)
 
# --- SLEW RATE / TEMPO DI SALITA ---
wn_s = 2 * np.pi * f_n_s
zeta_clip = np.clip(zeta_s, 1e-3, 1.999)
wn_tr_s = 1.76 * zeta_clip**3 - 0.417 * zeta_clip**2 + 1.039 * zeta_clip + 1.0
t_rise_s = np.where(zeta_s < 1.0, wn_tr_s / wn_s, 2.2 / wn_s * zeta_s)
delta_V = 0.8 * V_GEN
slew_rate_s = delta_V / t_rise_s  # [V/s]
 
# --- DROOP %/ms ---
# H_droop(t) ~ exp(-t/tau_droop); droop%/ms = 100*(1-exp(-1ms/tau_droop))
tau_droop_s = 1.0 / (2 * np.pi * f_d_s)
droop_pct_per_ms_s = 100.0 * (1.0 - np.exp(-1e-3 / tau_droop_s))
 
 
def summarize(x, label, unit=""):
    med = np.median(x)
    lo, hi = np.percentile(x, [16, 84])
    err = (hi - lo) / 2.0
    print(f"  {label:28s} = {med:.4g} +/- {err:.4g} {unit}")
    return med, err
 
 
print("\n" + "=" * 70)
print("GRANDEZZE FINALI STIMATE DAL FIT (mediana +/- incertezza, Monte Carlo)")
print("=" * 70)
summarize(overshoot_s, "Overshoot (dal fit)", "%")
summarize(t_rise_s * 1e9, "Tempo di salita 10-90% (dal fit)", "ns")
summarize(slew_rate_s, "Slew Rate equivalente (dal fit)", "V/s")
summarize(droop_pct_per_ms_s, "Droop (dal fit)", "%/ms")
print("-" * 70)
print("  Valori noti di riferimento (dall'onda quadra, usati come p0):")
print(f"  {'Overshoot noto':28s} = {Overshoot_noto_pct:.4g} %")
print(f"  {'Tempo di salita noto':28s} = {SR_noto_tempo_di_salita_ns:.4g} ns")
print(f"  {'Droop noto':28s} = {Droop_noto_pct_per_ms:.4g} %/ms")
print("=" * 70)
 
# ==============================================================================
# GRAFICO DIAGNOSTICO
# ==============================================================================
 
f_smooth = np.logspace(np.log10(freq.min() / 3), np.log10(max(freq.max() * 3, f_n_fit * 3)), 1000)
H_smooth = H_model(f_smooth, *popt)
 
fig, ax = plt.subplots(figsize=(9, 6))
ax.errorbar(freq, H_meas_amp, yerr=sigma_amp, fmt='o', color='tab:blue',
            label='Dati (V_in,misurata / V_GEN)', zorder=3, capsize=3)
ax.plot(f_smooth, np.abs(H_smooth), '-', color='tab:red',
        label=f'Modello fittato (chi2/dof={chi2/dof:.1f})', zorder=2)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Valore ideale (=1)')
ax.set_xscale('log')
ax.set_xlabel('Frequenza [Hz]')
ax.set_ylabel('|H(f)| = V_in,misurata / V_GEN')
ax.set_title('Risposta in ampiezza: dati vs modello droop + 2deg ordine')
ax.legend()
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('fit_ampiezza.png', dpi=150)
print("\nGrafico salvato: fit_ampiezza.png")
 
plt.show()