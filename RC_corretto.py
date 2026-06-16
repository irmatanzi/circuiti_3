import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2


def H_C(f, R_C, C):
    omega = 2 * np.pi * f
    R_tot = R_C + R_load
    return 1 / np.sqrt(1 + (omega * R_tot * C)**2)


def H_R(f, R_C, C):
    omega = 2 * np.pi * f
    R_tot = R_C + R_load
    return (omega * R_tot * C) / np.sqrt(1 + (omega * R_tot * C)**2)


def fase_C(f, R_C, C):
    omega = 2 * np.pi * f
    R_tot = R_C + R_load
    return -np.arctan(omega * R_tot * C)


def fase_R(f, R_C, C):
    omega = 2 * np.pi * f
    R_tot = R_C + R_load
    return np.pi / 2 - np.arctan(omega * R_tot * C)


# -------------------------------------------------------------------------
# Dati
# -------------------------------------------------------------------------

R_load = 3000.  # Ohm, resistenza nota
# R_C incognita (resistenza parassita del condensatore, multimetro in overload)
# R nel fit = R_tot = R_load + R_C  → lasciata libera, senza vincolo

C_misurato = 10e-9  # F, capacità misurata col multimetro

# Incertezza su C: specifica ±(1.2% + 2 digit), risoluzione 1 nF
sigma_C_percent = 0.012 * C_misurato          # contributo percentuale
sigma_C_digit   = 2 * 1e-9                    # 2 digit × 1 nF/digit
sigma_C = np.sqrt(sigma_C_percent**2 + sigma_C_digit**2) / np.sqrt(3)


# Incertezze di risoluzione strumentale
sigma_V_risoluzione = 0.001 / np.sqrt(12)  # V
sigma_frequenza     = 1     / np.sqrt(12)  # Hz

# -------------------------------------------------------------------------
# Misure
# -------------------------------------------------------------------------

frequenza = np.array([  200.,   500.,  1000.,  2000.,  5000.,  7000., 10000.,
                       20000., 30000., 40000., 50000., 60000.])  # Hz

V_A = np.array([ 9.964, 10.038,  9.984, 10.064, 10.027, 10.055,  9.867, 10.023, 10.072,
  9.989, 10.052,  9.967])
V_B = np.array([9.961, 9.951, 9.858, 9.267, 7.275, 6.058, 4.731, 2.494, 1.704, 1.304,
 1.145, 0.803])
V_AB = np.array([0.45 , 0.955, 1.814, 3.561, 6.916, 7.927, 8.815, 9.575, 9.759, 9.991,
 9.95 , 9.921])

sigma_V_A = np.array([0.15, 0.17, 0.12, 0.14, 0.131, 0.137, 0.134, 0.187, 0.068, 0.145,
 0.132, 0.143])
sigma_V_B = np.array([0.021, 0.046, 0.044, 0.026, 0.039, 0.008, 0.045, 0.103, 0.098, 0.028,
 0.027, 0.06 ])
sigma_V_AB = np.array([0.056, 0.109, 0.042, 0.009, 0.089, 0.055, 0.07 , 0.064, 0.044, 0.07 ,
 0.023, 0.072])

phi_A = np.array([ 0.101,  0.012,  0.001, -0.214, -0.173,  0.084,  0.067,  0.036, -0.14 ,
 -0.059, -0.146,  0.005]) * 1e-6
phi_B = np.array([-29.863, -29.701, -29.524, -28.481, -23.942, -21.029, -17.299, -10.535,
  -7.375,  -5.688,  -4.687,  -3.959]) * 1e-6
phi_AB = np.array([1.220e+03, 4.699e+02, 2.205e+02, 9.644e+01, 2.606e+01, 1.486e+01,
 7.933e+00, 2.004e+00, 9.913e-01, 6.863e-01, 3.970e-01, 3.277e-01]) * 1e-6

sigma_phi_A  = np.full(len(frequenza), 0.2e-6)  # s
sigma_phi_B  = np.full(len(frequenza), 0.2e-6)
sigma_phi_AB = np.full(len(frequenza), 0.2e-6)

# -------------------------------------------------------------------------
# Propagazione incertezze
# -------------------------------------------------------------------------

sigma_V_A  = np.sqrt(sigma_V_A**2  + sigma_V_risoluzione**2)
sigma_V_B  = np.sqrt(sigma_V_B**2  + sigma_V_risoluzione**2)
sigma_V_AB = np.sqrt(sigma_V_AB**2 + sigma_V_risoluzione**2)

y_C = V_B  / V_A
y_R = V_AB / V_A

sigma_y_C = y_C * np.sqrt((sigma_V_B  / V_B )**2 + (sigma_V_A / V_A)**2)
sigma_y_R = y_R * np.sqrt((sigma_V_AB / V_AB)**2 + (sigma_V_A / V_A)**2)

delta_phi_A_B  = phi_B  - phi_A
delta_phi_A_AB = phi_AB - phi_A

fase_A_B  = delta_phi_A_B  * 2 * np.pi * frequenza
fase_A_AB = delta_phi_A_AB * 2 * np.pi * frequenza

sigma_fase_A_B = 2*np.pi * np.sqrt(
    frequenza**2 * (sigma_phi_A**2 + sigma_phi_B**2)
    + delta_phi_A_B**2  * sigma_frequenza**2
)
sigma_fase_A_AB = 2*np.pi * np.sqrt(
    frequenza**2 * (sigma_phi_A**2 + sigma_phi_AB**2)
    + delta_phi_A_AB**2 * sigma_frequenza**2
)

# -------------------------------------------------------------------------
# Fit — vincolo gaussiano su C (nota), R_tot libera
# -------------------------------------------------------------------------
# R nel fit è R_tot = R_load + R_C: è incognita, nessun vincolo su di essa.
# Dal risultato del fit si ricava R_C = R_fit - R_load.

def costo_con_vincolo_C(ls):
    def costo(R_C, C):
        return ls(R_C, C) + ((C - C_misurato) / sigma_C)**2
    return costo


def frequenza_taglio(R_C, C):
    R_tot = R_C + R_load
    return 1 / (2 * np.pi * R_tot * C)


def errore_frequenza_taglio(m):
    R_C_fit = m.values["R_C"]
    C_fit = m.values["C"]
    R_tot = R_C_fit + R_load
    f_c = frequenza_taglio(R_C_fit, C_fit)
    cov = m.covariance
    df_dRC = -f_c / R_tot
    df_dC = -f_c / C_fit
    var_f_c = (
        df_dRC**2 * cov["R_C", "R_C"]
        + df_dC**2 * cov["C", "C"]
        + 2 * df_dRC * df_dC * cov["R_C", "C"]
    )
    return np.sqrt(var_f_c)


# ndof = N punti + 1 misura di C (vincolo) - 2 parametri
ndof = len(frequenza) + 1 - 2

# --- Fit ampiezza condensatore ---
ls_C = LeastSquares(frequenza, y_C, sigma_y_C, H_C)
m_C  = Minuit(costo_con_vincolo_C(ls_C), R_C=1000.0, C=C_misurato)
m_C.errordef = Minuit.LEAST_SQUARES
m_C.limits["R_C"] = (1e-9, None)
m_C.limits["C"] = (1e-12, 1e-6)
m_C.migrad()
m_C.hesse()
p_C = chi2.sf(m_C.fval, ndof)

# --- Fit ampiezza resistenza ---
ls_R = LeastSquares(frequenza, y_R, sigma_y_R, H_R)
m_R  = Minuit(costo_con_vincolo_C(ls_R), R_C=10.0, C=C_misurato)
m_R.errordef = Minuit.LEAST_SQUARES
m_R.limits["R_C"] = (0, None)
m_R.limits["C"] = (1e-12, 1e-6)
m_R.migrad()
m_R.hesse()
p_R = chi2.sf(m_R.fval, ndof)

R_C_fit_C = m_C.values["R_C"];  RC_err_C = m_C.errors["R_C"]
C_fit_C = m_C.values["C"];  C_err_C = m_C.errors["C"]
R_C_fit_R = m_R.values["R_C"];  RC_err_R = m_R.errors["R_C"]
C_fit_R = m_R.values["C"];  C_err_R = m_R.errors["C"]

Rtot_fit_C = R_C_fit_C + R_load
Rtot_fit_R = R_C_fit_R + R_load

f_c_fit_C  = frequenza_taglio(R_C_fit_C, C_fit_C)
f_c_err_C  = errore_frequenza_taglio(m_C)
f_c_fit_R  = frequenza_taglio(R_C_fit_R, C_fit_R)
f_c_err_R  = errore_frequenza_taglio(m_R)

# -------------------------------------------------------------------------
# Fit fase
# -------------------------------------------------------------------------

ls_C_fase = LeastSquares(frequenza, fase_A_B,  sigma_fase_A_B,  fase_C)
m_C_fase  = Minuit(costo_con_vincolo_C(ls_C_fase), R_C=10.0, C=C_misurato)
m_C_fase.errordef = Minuit.LEAST_SQUARES
m_C_fase.limits["R_C"] = (0, None)
m_C_fase.limits["C"] = (1e-12, 1e-6)
m_C_fase.migrad()
m_C_fase.hesse()
p_C_fase = chi2.sf(m_C_fase.fval, ndof)

ls_R_fase = LeastSquares(frequenza, fase_A_AB, sigma_fase_A_AB, fase_R)
m_R_fase  = Minuit(costo_con_vincolo_C(ls_R_fase), R_C=10.0, C=C_misurato)
m_R_fase.errordef = Minuit.LEAST_SQUARES
m_R_fase.limits["R_C"] = (0, None)
m_R_fase.limits["C"] = (1e-12, 1e-6)
m_R_fase.migrad()
m_R_fase.hesse()
p_R_fase = chi2.sf(m_R_fase.fval, ndof)

RC_fit_C_fase  = m_C_fase.values["R_C"]
RC_err_C_fase  = m_C_fase.errors["R_C"]
RC_fit_R_fase  = m_R_fase.values["R_C"]
RC_err_R_fase  = m_R_fase.errors["R_C"]

f_c_C_fase     = frequenza_taglio(m_C_fase.values["R_C"], m_C_fase.values["C"])
sigma_fc_C_fase= errore_frequenza_taglio(m_C_fase)
f_c_R_fase     = frequenza_taglio(m_R_fase.values["R_C"], m_R_fase.values["C"])
sigma_fc_R_fase= errore_frequenza_taglio(m_R_fase)

# -------------------------------------------------------------------------
# Stampa risultati
# -------------------------------------------------------------------------

print(f"R_load = {R_load:.3f} Ohm  (nota)")
print(f"C      = {C_misurato:.3e} ± {sigma_C:.3e} F  (nota, usata come vincolo)")

print(f"\n--- Fit ampiezza, condensatore ---")
print(f"R_C   = {R_C_fit_C:.3f} ± {RC_err_C:.3f} Ohm")
print(f"R_tot = {Rtot_fit_C:.3f} Ohm")
print(f"C     = {C_fit_C:.3e} ± {C_err_C:.3e} F")
print(f"f_c   = {f_c_fit_C:.1f} ± {f_c_err_C:.1f} Hz")
print(f"Chi2  = {m_C.fval:.2f},  Ndof = {ndof},  p-value = {p_C:.4f}")

print(f"\n--- Fit ampiezza, resistenza ---")
print(f"R_C   = {R_C_fit_R:.3f} ± {RC_err_R:.3f} Ohm")
print(f"R_tot = {Rtot_fit_R:.3f} Ohm")
print(f"C     = {C_fit_R:.3e} ± {C_err_R:.3e} F")
print(f"f_c   = {f_c_fit_R:.1f} ± {f_c_err_R:.1f} Hz")
print(f"Chi2  = {m_R.fval:.2f},  Ndof = {ndof},  p-value = {p_R:.4f}")

compatibilita = abs(f_c_fit_C - f_c_fit_R) / np.sqrt(f_c_err_C**2 + f_c_err_R**2)
print(f"\nCompatibilità f_c (ampiezza C vs R): {compatibilita:.2f} sigma")

print(f"\n--- Fit fase, condensatore ---")
print(f"R_C   = {RC_fit_C_fase:.3f} ± {RC_err_C_fase:.3f} Ohm")
print(f"R_tot = {RC_fit_C_fase + R_load:.3f} Ohm")
print(f"C     = {m_C_fase.values['C']:.3e} ± {m_C_fase.errors['C']:.3e} F")
print(f"f_c   = {f_c_C_fase:.1f} ± {sigma_fc_C_fase:.1f} Hz")
print(f"Chi2  = {m_C_fase.fval:.2f},  Ndof = {ndof},  p-value = {p_C_fase:.4f}")

print(f"\n--- Fit fase, resistenza ---")
print(f"R_C   = {RC_fit_R_fase:.3f} ± {RC_err_R_fase:.3f} Ohm")
print(f"R_tot = {RC_fit_R_fase + R_load:.3f} Ohm")
print(f"C     = {m_R_fase.values['C']:.3e} ± {m_R_fase.errors['C']:.3e} F")
print(f"f_c   = {f_c_R_fase:.1f} ± {sigma_fc_R_fase:.1f} Hz")
print(f"Chi2  = {m_R_fase.fval:.2f},  Ndof = {ndof},  p-value = {p_R_fase:.4f}")

compatibilita_fase = abs(f_c_C_fase - f_c_R_fase) / np.sqrt(sigma_fc_C_fase**2 + sigma_fc_R_fase**2)
print(f"\nCompatibilità f_c (fase C vs R): {compatibilita_fase:.2f} sigma")

# -------------------------------------------------------------------------
# Grafici — scala lineare
# -------------------------------------------------------------------------

x_axis = np.linspace(np.min(frequenza), np.max(frequenza), 1000)

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 7))

ax[0].set_title("Rapporto delle ampiezze $V_{out} / V_{in}$ — scala lineare")
ax[0].set_xlabel("Frequenza (Hz)")
ax[0].set_ylabel("$V_{out} / V_{in}$")
ax[0].errorbar(frequenza, y_C, yerr=sigma_y_C, fmt="o", capsize=4,
               color="hotpink", label="Dati osservati (condensatore)")
ax[0].plot(x_axis, H_C(x_axis, R_C_fit_C, C_fit_C), color="hotpink",
           label="Funzione di trasferimento $H_C$")
ax[0].errorbar(frequenza, y_R, yerr=sigma_y_R, fmt="o", capsize=4,
               color="lightblue", label="Dati osservati (resistenza)")
ax[0].plot(x_axis, H_R(x_axis, R_C_fit_R, C_fit_R), color="lightblue",
           label="Funzione di trasferimento $H_R$")
ax[0].legend()
ax[0].grid(True, alpha=0.4)

ax[1].set_title("Differenza di fase $\\Delta\\phi$ — scala lineare")
ax[1].set_xlabel("Frequenza (Hz)")
ax[1].set_ylabel("$\\phi_{out} - \\phi_{in}$ (rad)")
ax[1].errorbar(frequenza, fase_A_B, yerr=sigma_fase_A_B, fmt="o", capsize=4,
               color="hotpink", label="Dati osservati (condensatore)")
ax[1].plot(x_axis, fase_C(x_axis, R_C_fit_C, C_fit_C), color="hotpink",
           label="Fase teorica")
ax[1].errorbar(frequenza, fase_A_AB, yerr=sigma_fase_A_AB, fmt="o", capsize=4,
               color="lightblue", label="Dati osservati (resistenza)")
ax[1].plot(x_axis, fase_R(x_axis, R_C_fit_R, C_fit_R), color="lightblue",
           label="Fase teorica")
ax[1].legend()
ax[1].grid(True, alpha=0.4)

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# Grafici — scala logaritmica
# -------------------------------------------------------------------------

fig, ax = plt.subplots(2, 1, figsize=(9, 7))

ax[0].set_title("Rapporto delle ampiezze $V_{out} / V_{in}$ — scala logaritmica")
ax[0].set_xlabel("Frequenza (Hz)")
ax[0].set_ylabel("$V_{out} / V_{in}$")
ax[0].errorbar(frequenza, y_C, yerr=sigma_y_C, fmt="o", capsize=4,
               color="hotpink", label="Dati osservati (condensatore)")
ax[0].plot(x_axis, H_C(x_axis, R_C_fit_C, C_fit_C), color="hotpink",
           label="Funzione di trasferimento $H_C$")
ax[0].errorbar(frequenza, y_R, yerr=sigma_y_R, fmt="o", capsize=4,
               color="lightblue", label="Dati osservati (resistenza)")
ax[0].plot(x_axis, H_R(x_axis, R_C_fit_R, C_fit_R), color="lightblue",
           label="Funzione di trasferimento $H_R$")
ax[0].set_xscale("log")
ax[0].legend()
ax[0].grid(True, alpha=0.4)

ax[1].set_title("Differenza di fase $\\Delta\\phi$ — scala logaritmica")
ax[1].set_xlabel("Frequenza (Hz)")
ax[1].set_ylabel("$\\phi_{out} - \\phi_{in}$ (rad)")
ax[1].errorbar(frequenza, fase_A_B, yerr=sigma_fase_A_B, fmt="o", capsize=4,
               color="hotpink", label="Dati osservati (condensatore)")
ax[1].plot(x_axis, fase_C(x_axis, R_C_fit_C, C_fit_C), color="hotpink")
ax[1].errorbar(frequenza, fase_A_AB, yerr=sigma_fase_A_AB, fmt="o", capsize=4,
               color="lightblue", label="Dati osservati (resistenza)")
ax[1].plot(x_axis, fase_R(x_axis, R_C_fit_R, C_fit_R), color="lightblue")
ax[1].set_xscale("log")
ax[1].legend()
ax[1].grid(True, alpha=0.4)

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# Grafico impedenze
# -------------------------------------------------------------------------

R_best = R_C_fit_C + R_load
C_best = C_fit_C
R_err  = RC_err_C
C_err  = C_err_C

x_log = np.logspace(np.log10(100), np.log10(60000), 2000)

Z_R_teorica = np.full_like(x_log, R_best)
Z_C_teorica = 1 / (2 * np.pi * x_log * C_best)
sigma_Z_C_teorica = Z_C_teorica * (C_err / C_best)

Z_ratio_dati  = y_C / y_R
Z_C_dati      = Z_ratio_dati * R_load
sigma_Z_ratio = Z_ratio_dati * np.sqrt((sigma_y_C / y_C)**2 + (sigma_y_R / y_R)**2)
sigma_Z_C_dati = R_load * sigma_Z_ratio

maschera_impedenze = frequenza != 200

w_C = 1 / f_c_err_C**2
w_R = 1 / f_c_err_R**2
f_c_media      = (w_C * f_c_fit_C + w_R * f_c_fit_R) / (w_C + w_R)
sigma_fc_media = 1 / np.sqrt(w_C + w_R)

fig, ax = plt.subplots(figsize=(8, 5))

ax.loglog(x_log, Z_R_teorica, color="lightblue", lw=2,
          label=f"$Z_R$ teorica (fit), $R_{{tot}}$ = {R_best:.0f} Ω")
ax.loglog(x_log, Z_C_teorica, color="hotpink",   lw=2,
          label=f"$Z_C$ teorica (fit), $C$ = {C_best:.2e} F")
ax.fill_between(x_log,
                Z_C_teorica - sigma_Z_C_teorica,
                Z_C_teorica + sigma_Z_C_teorica,
                color="hotpink", alpha=0.1, label="Banda $\\pm 1\\sigma$ su $Z_C$")
ax.errorbar(frequenza[maschera_impedenze],
            Z_C_dati[maschera_impedenze],
            yerr=sigma_Z_C_dati[maschera_impedenze],
            fmt="o", capsize=4, color="navy",
            label="$Z_C$ stimata dai dati ($R_{load} \\cdot V_C / V_{R}$)")
ax.axvline(f_c_media, color="crimson", lw=1.5, ls="--",
           label=f"$f_c$ = {f_c_media:.0f} ± {sigma_fc_media:.0f} Hz")
ax.plot(f_c_media, R_best, marker="*", markersize=10, color="gold",
        zorder=5, label=f"Incrocio $Z_R = Z_C$ a $f_c$")
ax.set_xlabel("Frequenza (Hz)")
ax.set_ylabel("Impedenza (Ω)")
ax.set_title("Impedenze $Z_R$ e $Z_C$ in funzione della frequenza")
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.2)

plt.tight_layout()
plt.show()