import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

# ─────────────────────────────────────────────
# FUNZIONI DI TRASFERIMENTO (moduli)
# ─────────────────────────────────────────────
 
def H_R(f, R, L, C):
    omega = 2 * np.pi * f
    return (omega * R * C) / np.sqrt((1 - omega**2 * L * C)**2 + (omega * R * C)**2)
 
def H_L(f, R, L, C):
    omega = 2 * np.pi * f
    return (omega**2 * L * C) / np.sqrt((1 - omega**2 * L * C)**2 + (omega * R * C)**2)
 
def H_C(f, R, L, C):
    omega = 2 * np.pi * f
    return 1 / np.sqrt((1 - omega**2 * L * C)**2 + (omega * R * C)**2)
 
# ─────────────────────────────────────────────
# FUNZIONI DI FASE
# ─────────────────────────────────────────────
 
def fase_R(f, R, L, C):
    # V_R è in fase con la corrente; la corrente è in ritardo di arctan2(X_L - X_C, R) rispetto a V_in
    omega = 2 * np.pi * f
    X = omega * L - 1 / (omega * C)
    return -np.arctan2(X, R)
 
def fase_L(f, R, L, C):
    # V_L anticipa la corrente di π/2, quindi: fase_R + π/2
    omega = 2 * np.pi * f
    X = omega * L - 1 / (omega * C)
    return np.pi / 2 - np.arctan2(X, R)
 
def fase_C(f, R, L, C):
    # V_C è in ritardo sulla corrente di π/2, quindi: fase_R - π/2
    omega = 2 * np.pi * f
    X = omega * L - 1 / (omega * C)
    return -np.pi / 2 - np.arctan2(X, R)


# ─────────────────────────────────────────────
# DATI
# ─────────────────────────────────────────────
 
# Valore noto di R (dal multimetro o dalla taratura)
R_noto = 208   # Ohm  ← (200 + 8 stimati)
sigma_R_noto = 0.01 * R_noto / np.sqrt(3)  # tolleranza 1% produttore
 
# Valori iniziali per il fit
L_start = 10e-3   # H
C_start = 10e-9  # F
 
# Incertezze di risoluzione
sigma_V_risoluzione = 0.01 / np.sqrt(12)   # V
sigma_frequenza = 1 / np.sqrt(12)          # Hz


frequenze = np.array([ 4000.,  5000.,  6000.,  7000.,  8000.,  9000., 10000., 15000., 20000.,
 30000., 50000., 70000.])

V_A = np.array([ 9.986, 10.015,  9.993, 10.025, 10.011, 10.022,  9.947, 10.009, 10.029,
  9.996, 10.021,  9.987]) # V, ai capi del generatore

V_R = np.array([0.544, 0.721, 0.923, 1.091, 1.385, 1.711, 2.129, 8.66 , 4.099, 1.513,
 0.771, 0.467]) # V, ai capi di R

V_L = np.array([ 0.703,  1.099,  1.634,  2.397,  3.371,  4.615,  6.368, 39.33 , 24.818,
 13.787, 11.1  , 10.516]) # V, ai capi di L

V_C = np.array([10.678, 11.068, 11.608, 12.277, 13.217, 14.503, 16.164, 44.325, 15.71 ,
  3.86 ,  1.095,  0.545]) # V, ai capi di C


sigma_V_A = np.array([0.026, 0.027, 0.031, 0.017, 0.012, 0.025, 0.024, 0.035, 0.013, 0.018,
 0.013, 0.017])

sigma_V_R = np.array([0.028, 0.019, 0.018, 0.02 , 0.015, 0.003, 0.018, 0.041, 0.039, 0.021,
 0.021, 0.024])

sigma_V_L = np.array([0.022, 0.044, 0.017, 0.014, 0.036, 0.022, 0.028, 0.025, 0.018, 0.028,
 0.019, 0.029])

sigma_V_C = np.array([0.028, 0.026, 0.02 , 0.026, 0.03 , 0.029, 0.03 , 0.04 , 0.04 , 0.029,
 0.037, 0.025])


phi_A = np.array([ 0.123,  0.21 ,  0.128,  0.208,  0.116, -0.06 , -0.062, -0.098,  0.03 ,
  0.037, -0.023, -0.026])*1e-6

phi_R = np.array([60.336, 47.505, 39.415, 33.272, 28.607, 24.868, 21.785,  5.437, -9.063,
 -7.363, -4.705, -3.364])*1e-6

phi_L = np.array([122.578,  97.775,  80.88 ,  68.951,  59.895,  52.329,  46.516,  22.21 ,
   3.163,   1.038,   0.157,   0.135])*1e-6

phi_C = np.array([ -2.091,  -2.206,  -2.527,  -2.439,  -2.834,  -3.102,  -3.454, -11.028,
 -21.504, -15.974,  -9.575,  -6.96 ])*1e-6


sigma_phi_A = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * 1e-6 # secondi
sigma_phi_R = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * 1e-6 # secondi
sigma_phi_C = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * 1e-6 # secondi
sigma_phi_L = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * 1e-6 # secondi


# ─────────────────────────────────────────────
# PROPAGAZIONE INCERTEZZE SUI RAPPORTI V_x/V_A
# ─────────────────────────────────────────────
 
sigma_V_A = np.sqrt(sigma_V_A**2 + sigma_V_risoluzione**2)
sigma_V_R = np.sqrt(sigma_V_R**2 + sigma_V_risoluzione**2)
sigma_V_L = np.sqrt(sigma_V_L**2 + sigma_V_risoluzione**2)
sigma_V_C = np.sqrt(sigma_V_C**2 + sigma_V_risoluzione**2)
 
y_R = V_R / V_A
y_L = V_L / V_A
y_C = V_C / V_A
 
sigma_y_R = y_R * np.sqrt((sigma_V_R / V_R)**2 + (sigma_V_A / V_A)**2)
sigma_y_L = y_L * np.sqrt((sigma_V_L / V_L)**2 + (sigma_V_A / V_A)**2)
sigma_y_C = y_C * np.sqrt((sigma_V_C / V_C)**2 + (sigma_V_A / V_A)**2)
 
# ─────────────────────────────────────────────
# FASI E LORO INCERTEZZE
# ─────────────────────────────────────────────
 
delta_phi_AR = phi_R - phi_A
delta_phi_AL = phi_L - phi_A
delta_phi_AC = phi_C - phi_A

fase_A_R  = delta_phi_AR * 2 * np.pi * frequenze
fase_A_L  = delta_phi_AL * 2 * np.pi * frequenze
fase_A_C  = delta_phi_AC * 2 * np.pi * frequenze
 
def sigma_fase(delta_phi, sigma_phi1, sigma_phi2, f, sigma_f):
    return 2 * np.pi * np.sqrt(
        f**2 * (sigma_phi1**2 + sigma_phi2**2)
        + delta_phi**2 * sigma_f**2
    )
 
sigma_fase_A_R = sigma_fase(delta_phi_AR, sigma_phi_A, sigma_phi_R, frequenze, sigma_frequenza)
sigma_fase_A_L = sigma_fase(delta_phi_AL, sigma_phi_A, sigma_phi_L, frequenze, sigma_frequenza)
sigma_fase_A_C = sigma_fase(delta_phi_AC, sigma_phi_A, sigma_phi_C, frequenze, sigma_frequenza)
 
# ─────────────────────────────────────────────
# FIT CON VINCOLO SU R
# ─────────────────────────────────────────────
 
def costo_con_vincolo_R(ls):
    def costo(R, L, C):
        return ls(R, L, C) + ((R - R_noto) / sigma_R_noto)**2
    return costo
 
ls_R = LeastSquares(frequenze, y_R, sigma_y_R, H_R)
ls_L = LeastSquares(frequenze, y_L, sigma_y_L, H_L)
ls_C = LeastSquares(frequenze, y_C, sigma_y_C, H_C)

ls_R_fase = LeastSquares(frequenze, fase_A_R, sigma_fase_A_R, fase_R)
ls_L_fase = LeastSquares(frequenze, fase_A_L, sigma_fase_A_L, fase_L)
ls_C_fase = LeastSquares(frequenze, fase_A_C, sigma_fase_A_C, fase_C)
 
def make_minuit(ls):
    m = Minuit(costo_con_vincolo_R(ls), R=R_noto, L=L_start, C=C_start)
    m.errordef = Minuit.LEAST_SQUARES
    m.limits["R"] = (1.0, None)
    m.limits["L"] = (1e-5, 1.0)
    m.limits["C"] = (1e-11, 1e-6)
    m.migrad()
    m.hesse()
    return m
 
m_R = make_minuit(ls_R)
m_L = make_minuit(ls_L)
m_C = make_minuit(ls_C)

m_R_fase = make_minuit(ls_R_fase)
m_L_fase = make_minuit(ls_L_fase)
m_C_fase = make_minuit(ls_C_fase)

ndof = len(frequenze) + 1 - 3  # N punti + vincolo - 3 parametri liberi

p_R = chi2.sf(m_R.fval, ndof)
p_L = chi2.sf(m_L.fval, ndof)
p_C = chi2.sf(m_C.fval, ndof)

p_R_fase = chi2.sf(m_R_fase.fval, ndof)
p_L_fase = chi2.sf(m_L_fase.fval, ndof)
p_C_fase = chi2.sf(m_C_fase.fval, ndof)
 
# ─────────────────────────────────────────────
# FREQUENZA DI RISONANZA E FATTORE DI QUALITÀ
# ─────────────────────────────────────────────
 
def frequenza_risonanza(L, C):
    return 1 / (2 * np.pi * np.sqrt(L * C))
 
def fattore_Q(R, L, C):
    return (1 / R) * np.sqrt(L / C)
 
def errori_f0_Q(m):
    R  = m.values["R"];  L  = m.values["L"];  C  = m.values["C"]
    cov = m.covariance
    f0 = frequenza_risonanza(L, C)
    Q  = fattore_Q(R, L, C)
 
    # df0/dL, df0/dC
    df0_dL = -f0 / (2 * L)
    df0_dC = -f0 / (2 * C)
    var_f0 = (df0_dL**2 * cov["L","L"]
              + df0_dC**2 * cov["C","C"]
              + 2 * df0_dL * df0_dC * cov["L","C"])
 
    # dQ/dR, dQ/dL, dQ/dC
    dQ_dR = -Q / R
    dQ_dL =  Q / (2 * L)
    dQ_dC = -Q / (2 * C)
    var_Q = (dQ_dR**2 * cov["R","R"]
             + dQ_dL**2 * cov["L","L"]
             + dQ_dC**2 * cov["C","C"]
             + 2 * dQ_dR * dQ_dL * cov["R","L"]
             + 2 * dQ_dR * dQ_dC * cov["R","C"]
             + 2 * dQ_dL * dQ_dC * cov["L","C"])
    return np.sqrt(var_f0), np.sqrt(var_Q)
 
# ─────────────────────────────────────────────
# STAMPA RISULTATI
# ─────────────────────────────────────────────
 
print(f"\nR_noto = {R_noto:.3f} ± {sigma_R_noto:.3f} Ohm")
 
for nome, m, p in [("resistenza (H_R)", m_R, p_R),
                   ("induttore  (H_L)", m_L, p_L),
                   ("condensatore (H_C)", m_C, p_C),
                   ("fase resistenza (fase_R)", m_R_fase, p_R_fase),
                   ("fase induttore  (fase_L)", m_L_fase, p_L_fase),
                   ("fase condensatore (fase_C)", m_C_fase, p_C_fase)]:
    R_ = m.values["R"]; sR = m.errors["R"]
    L_ = m.values["L"]; sL = m.errors["L"]
    C_ = m.values["C"]; sC = m.errors["C"]
    f0 = frequenza_risonanza(L_, C_)
    Q  = fattore_Q(R_, L_, C_)
    sf0, sQ = errori_f0_Q(m)
    print(f"\n--- Fit su {nome} ---")
    print(f"  R  = {R_:.2f} ± {sR:.2f} Ohm")
    print(f"  L  = {L_*1e3:.3f} ± {sL*1e3:.3f} mH")
    print(f"  C  = {C_:.3e} ± {sC:.3e} F")
    print(f"  f0 = {f0:.1f} ± {sf0:.1f} Hz")
    print(f"  Q  = {Q:.2f} ± {sQ:.2f}")
    print(f"  Chi2 = {m.fval:.2f},  Ndof = {ndof},  p-value = {p:.4f}")
 
# Compatibilità tra le frequenze di risonanza dei tre fit
f0_vals  = []
sf0_vals = []
for m in [m_R, m_L, m_C]:
    L_ = m.values["L"]; C_ = m.values["C"]
    f0 = frequenza_risonanza(L_, C_)
    sf0, _ = errori_f0_Q(m)
    f0_vals.append(f0); sf0_vals.append(sf0)
 
print(f"\n--- Compatibilità f0 tra i fit ---")
labels = ["H_R", "H_L", "H_C"]
for i in range(3):
    for j in range(i+1, 3):
        z = abs(f0_vals[i] - f0_vals[j]) / np.sqrt(sf0_vals[i]**2 + sf0_vals[j]**2)
        print(f"  {labels[i]} vs {labels[j]}: {z:.2f} sigma")
 
# media pesata di f0
w = [1/s**2 for s in sf0_vals]
f0_media = sum(wi*fi for wi, fi in zip(w, f0_vals)) / sum(w)
sigma_f0_media = 1 / np.sqrt(sum(w))
print(f"\n  f0 (media pesata) = {f0_media:.1f} ± {sigma_f0_media:.1f} Hz")
 
# ─────────────────────────────────────────────
# GRAFICI LINEARI (ampiezza + fase)
# ─────────────────────────────────────────────
 
x_axis = np.linspace(np.min(frequenze), np.max(frequenze), 5000)
colori = {"R": "lightblue", "L": "mediumseagreen", "C": "hotpink"}
 
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
 
ax0, ax1 = axes
ax0.set_ylabel("$V_{out} / V_{in}$")
ax0.set_xlabel("Frequenza (Hz)")

ax0.set_title ("Rapporto delle ampiezze $V_{out} / V_{in}$ - scala lineare")
ax1.set_title ("Differenza tra le fasi $\\Delta \\phi$ - scala lineare")
 
for nome, y, sy, H_func, m, col in [
    ("R", y_R, sigma_y_R, H_R, m_R, colori["R"]),
    ("L", y_L, sigma_y_L, H_L, m_L, colori["L"]),
    ("C", y_C, sigma_y_C, H_C, m_C, colori["C"]),
]:
    ax0.errorbar(frequenze, y, yerr=sy, fmt="o", capsize=4, color=col,
                 label=f"Dati ($V_{{{nome}}} / V_A$)")
    ax0.plot(x_axis, H_func(x_axis, m.values["R"], m.values["L"], m.values["C"]),
             color=col, label=f"Fit $H_{{{nome}}}$")
 
ax0.legend()
ax0.grid(True, alpha=0.4)
 
ax1.set_ylabel("$\\phi_{out} - \\phi_{in}$ (rad)")
ax1.set_xlabel("Frequenza (Hz)")
 
for nome, fase_dati, sigma_fase_dati, fase_func, m, col in [
    ("R", fase_A_R, sigma_fase_A_R, fase_R, m_R, colori["R"]),
    ("L", fase_A_L, sigma_fase_A_L, fase_L, m_L, colori["L"]),
    ("C", fase_A_C, sigma_fase_A_C, fase_C, m_C, colori["C"]),
]:
    ax1.errorbar(frequenze, fase_dati, yerr=sigma_fase_dati, fmt="o", capsize=4,
                 color=col, label=f"Dati ($V_{{{nome}}}$)")
    ax1.plot(x_axis, fase_func(x_axis, m.values["R"], m.values["L"], m.values["C"]),
             color=col, label=f"Fase teorica {nome}")
 
ax1.legend()
ax1.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
 
# ─────────────────────────────────────────────
# GRAFICI LOGARITMICI
# ─────────────────────────────────────────────
 
x_log = np.logspace(np.log10(frequenze.min() * 0.8), np.log10(frequenze.max() * 1.2), 5000)
 
fig, axes = plt.subplots(2, 1, figsize=(9, 7))
 
ax0, ax1 = axes
ax0.set_ylabel("$V_{out} / V_{in}$")
ax0.set_xlabel("Frequenza (Hz)")

ax0.set_title ("Rapporto delle ampiezze $V_{out} / V_{in}$ - scala logaritmica")
ax1.set_title ("Differenza tra le fasi $\\Delta \\phi$ - scala logaritmica")
 
for nome, y, sy, H_func, m, col in [
    ("R", y_R, sigma_y_R, H_R, m_R, colori["R"]),
    ("L", y_L, sigma_y_L, H_L, m_L, colori["L"]),
    ("C", y_C, sigma_y_C, H_C, m_C, colori["C"]),
]:
    ax0.errorbar(frequenze, y, yerr=sy, fmt="o", capsize=4, color=col,
                 label=f"Dati ($V_{{{nome}}} / V_A$)")
    ax0.plot(x_log, H_func(x_log, m.values["R"], m.values["L"], m.values["C"]),
             color=col, label=f"Fit $H_{{{nome}}}$")
 
ax0.set_xscale("log")
ax0.legend()
ax0.grid(True, which="both", alpha=0.4)
 
ax1.set_ylabel("$\\phi_{out} - \\phi_{in}$ (rad)")
ax1.set_xlabel("Frequenza (Hz)")
 
for nome, fase_dati, sigma_fase_dati, fase_func, m, col in [
    ("R", fase_A_R, sigma_fase_A_R, fase_R, m_R, colori["R"]),
    ("L", fase_A_L, sigma_fase_A_L, fase_L, m_L, colori["L"]),
    ("C", fase_A_C, sigma_fase_A_C, fase_C, m_C, colori["C"]),
]:
    ax1.errorbar(frequenze, fase_dati, yerr=sigma_fase_dati, fmt="o", capsize=4,
                 color=col, label=f"Dati ($V_{{{nome}}}$)")
    ax1.plot(x_log, fase_func(x_log, m.values["R"], m.values["L"], m.values["C"]),
             color=col)
 
ax1.set_xscale("log")
ax1.legend()
ax1.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.show()
 
# ─────────────────────────────────────────────
# IMPEDENZE IN FUNZIONE DELLA FREQUENZA
# ─────────────────────────────────────────────
 
# Usiamo i parametri dal fit sulla resistenza (il più diretto)
R_best = m_R.values["R"];  R_err_best = m_R.errors["R"]
L_best = m_R.values["L"];  L_err_best = m_R.errors["L"]
C_best = m_R.values["C"];  C_err_best = m_R.errors["C"]
 
Z_R_teo = np.full_like(x_log, R_best)
Z_L_teo = 2 * np.pi * x_log * L_best
Z_C_teo = 1 / (2 * np.pi * x_log * C_best)
 
sigma_Z_L_teo = Z_L_teo * (L_err_best / L_best)
sigma_Z_C_teo = Z_C_teo * (C_err_best / C_best)
 
# Impedenze sperimentali: dai partitori
#   Z_R = R_noto (costante),  Z_L = R * (V_L / V_R),  Z_C = R * (V_C / V_R)
Z_L_dati = R_noto * (y_L / y_R)
Z_C_dati = R_noto * (y_C / y_R)
 
sigma_Z_L_dati = Z_L_dati * np.sqrt(
    (sigma_y_L / y_L)**2 + (sigma_y_R / y_R)**2 + (sigma_R_noto / R_noto)**2
)
sigma_Z_C_dati = Z_C_dati * np.sqrt(
    (sigma_y_C / y_C)**2 + (sigma_y_R / y_R)**2 + (sigma_R_noto / R_noto)**2
)
 
# Frequenza di risonanza media pesata (già calcolata sopra)
f0 = f0_media
sigma_f0 = sigma_f0_media
 
fig, ax = plt.subplots(figsize=(9, 6))
 
ax.loglog(x_log, Z_R_teo, color=colori["R"], lw=2,
          label=f"$Z_R$ teorica (fit), $R$ = {R_best:.0f} Ω")
ax.loglog(x_log, Z_L_teo, color=colori["L"], lw=2,
          label=f"$Z_L$ teorica (fit), $L$ = {L_best*1e3:.2f} mH")
ax.loglog(x_log, Z_C_teo, color=colori["C"], lw=2,
          label=f"$Z_C$ teorica (fit), $C$ = {C_best:.2e} F")
 
ax.fill_between(x_log, Z_L_teo - sigma_Z_L_teo, Z_L_teo + sigma_Z_L_teo,
                color=colori["L"], alpha=0.1, label="Banda $\\pm 1\\sigma$ su $Z_L$")
ax.fill_between(x_log, Z_C_teo - sigma_Z_C_teo, Z_C_teo + sigma_Z_C_teo,
                color=colori["C"], alpha=0.1, label="Banda $\\pm 1\\sigma$ su $Z_C$")
 
ax.errorbar(frequenze, Z_L_dati, yerr=sigma_Z_L_dati,
            fmt="^", capsize=4, color="navy",
            label="$Z_L$ stimata dai dati ($R \\cdot V_L / V_R$)")
ax.errorbar(frequenze, Z_C_dati, yerr=sigma_Z_C_dati,
            fmt="o", capsize=4, color="navy",
            label="$Z_C$ stimata dai dati ($R \\cdot V_C / V_R$)")
 
ax.axvline(f0, color="crimson", lw=1.5, ls="--",
           label=f"$f_0$ = {f0:.0f} ± {sigma_f0:.0f} Hz")
 
# Punto di incrocio Z_L = Z_C = Z_R a f0
ax.plot(f0, R_best, marker="*", markersize=10, color="gold",
        zorder=5, label=f"Risonanza: $Z_L = Z_C$ a $f_0$")
 
ax.set_xlabel("Frequenza (Hz)")
ax.set_ylabel("Impedenza (Ω)")
ax.set_title("Impedenze $Z_R$, $Z_L$, $Z_C$ in funzione della frequenza")
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.2)
plt.tight_layout()
plt.show()