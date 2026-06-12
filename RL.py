import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2


def H_L(f, R, L):
    omega = 2 * np.pi * np.asarray(f)
    return (omega * L) / np.sqrt(R**2 + (omega * L)**2)


def H_R(f, R, L):
    omega = 2 * np.pi * np.asarray(f)
    return R / np.sqrt(R**2 + (omega * L)**2)


def fase_L(f, R, L):
    omega = 2 * np.pi * np.asarray(f)
    return np.arctan(R / (omega * L))


def fase_R(f, R, L):
    omega = 2 * np.pi * np.asarray(f)
    return -np.arctan((omega * L) / R)


# Dati:
R_L = 8.3 # Ohm
R = 50 # Ohm
R_tot = R + R_L

# Incertezze di risoluzione: distribuzione uniforme sull'ultima cifra.
sigma_R = 1 / np.sqrt(12) # Ohm, ultima cifra di R = 1 Ohm
sigma_R_L = 0.1 / np.sqrt(12) # Ohm, ultima cifra di R_L = 0.1 Ohm
sigma_R_tot = np.sqrt(sigma_R**2 + sigma_R_L**2)

sigma_V_risoluzione = 0.001 / np.sqrt(12) # V, ultima cifra letta = 0.001 V
sigma_frequenza = 1 / np.sqrt(12) # Hz, ultima cifra letta = 1 Hz

# i dati veri stanno nella cartella dati

frequenza = np.array([200., 500., 2000., 5000., 10000., 25000., 50000., 100000., 150000.])

V_A = np.array([10.026, 10.018,  9.974,  9.963, 10.02 , 10.049,  9.949, 10.019, 10.019])
V_B = np.array([ 2.104,  4.748,  9.069,  9.839,  9.951, 10.006, 10.004, 10.037,  9.963])
V_AB = np.array([9.743, 8.775, 4.201, 1.842, 0.909, 0.329, 0.164, 0.088, 0.042])

sigma_V_A = np.array([0.02 , 0.013, 0.028, 0.02, 0.016, 0.024, 0.015, 0.014, 0.023])
sigma_V_B = np.array([0.02 , 0.015, 0.021, 0.019, 0.019, 0.02, 0.026, 0.028, 0.036])
sigma_V_AB = np.array([0.032, 0.016, 0.02, 0.02, 0.015, 0.034, 0.016, 0.019, 0.034])

phi_A = np.array([ 0.143, -0.022, -0.132,  0.19 ,  0.14 , -0.092,  0.034, -0.052,  0.094]) * 1e-6 # secondi
phi_B = np.array([ 1.081e+03,  3.427e+02,  3.470e+01,  5.903e+00,  1.274e+00,  2.323e-01,
  7.175e-02,  9.722e-02, -4.936e-02]) * 1e-6 # secondi
phi_AB = np.array([-168.869, -157.443,  -90.362,  -44.067,  -23.429,   -9.843,   -4.865,
   -2.374,   -1.729]) * 1e-6 # secondi

sigma_phi_A = np.array([0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * 1e-6 # secondi
sigma_phi_B = np.array([0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * 1e-6 # secondi
sigma_phi_AB = np.array([0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * 1e-6 # secondi


sigma_V_A = np.sqrt(sigma_V_A**2 + sigma_V_risoluzione**2)
sigma_V_B = np.sqrt(sigma_V_B**2 + sigma_V_risoluzione**2)
sigma_V_AB = np.sqrt(sigma_V_AB**2 + sigma_V_risoluzione**2)


# differenze tra fasi (lo 0 sarà A)
fase_A_B = phi_B - phi_A 
fase_A_AB = phi_AB - phi_A 

y_L = V_B / V_A
y_R = V_AB / V_A

sigma_y_L = y_L * np.sqrt(
    (sigma_V_B / V_B)**2 +
    (sigma_V_A / V_A)**2
)

sigma_y_R = y_R * np.sqrt(
    (sigma_V_AB / V_AB)**2 +
    (sigma_V_A / V_A)**2
)

fase_A_B = (phi_B - phi_A) * 2 * np.pi * frequenza
fase_A_AB = (phi_AB - phi_A) * 2 * np.pi * frequenza

delta_phi_A_B = phi_B - phi_A
delta_phi_A_AB = phi_AB - phi_A

sigma_fase_A_B = (
    2*np.pi
    * np.sqrt(
        (frequenza**2) * (sigma_phi_A**2 + sigma_phi_B**2)
        + (delta_phi_A_B**2) * sigma_frequenza**2
    )
)

sigma_fase_A_AB = (
    2*np.pi
    * np.sqrt(
        (frequenza**2) * (sigma_phi_A**2 + sigma_phi_AB**2)
        + (delta_phi_A_AB**2) * sigma_frequenza**2
    )
)

ls_L = LeastSquares(frequenza, y_L, sigma_y_L, H_L)
ls_R = LeastSquares(frequenza, y_R, sigma_y_R, H_R)


def costo_con_vincolo_R(ls):
    def costo(R, L):
        return ls(R, L) + ((R - R_tot) / sigma_R_tot)**2
    return costo


def frequenza_taglio(R, L):
    return R / (2 * np.pi * L)


def errore_frequenza_taglio(m):
    R_fit = m.values["R"]
    L_fit = m.values["L"]
    f_c = frequenza_taglio(R_fit, L_fit)
    cov = m.covariance

    df_dR = f_c / R_fit
    df_dL = -f_c / L_fit
    var_f_c = (
        df_dR**2 * cov["R", "R"]
        + df_dL**2 * cov["L", "L"]
        + 2 * df_dR * df_dL * cov["R", "L"]
    )
    return np.sqrt(var_f_c)


m_L = Minuit(costo_con_vincolo_R(ls_L), R=R_tot, L=10e-3)
m_L.errordef = Minuit.LEAST_SQUARES
m_L.limits["R"] = (1e-9, None)
m_L.limits["L"] = (1e-10, 1e-1)
m_L.migrad()
m_L.hesse()

ndof_L = len(frequenza) + 1 - 2
p_L = chi2.sf(m_L.fval, ndof_L)

m_R = Minuit(costo_con_vincolo_R(ls_R), R=R_tot, L=10e-3)
m_R.errordef = Minuit.LEAST_SQUARES
m_R.limits["R"] = (1e-9, None)
m_R.limits["L"] = (1e-10, 1e-1)
m_R.migrad()
m_R.hesse()

p_R = chi2.sf(m_R.fval, ndof_L)


R_fit_L = m_L.values["R"]
R_err_L = m_L.errors["R"]
L_fit_L = m_L.values["L"]
L_err_L = m_L.errors["L"]
R_fit_R = m_R.values["R"]
R_err_R = m_R.errors["R"]
L_fit_R = m_R.values["L"]
L_err_R = m_R.errors["L"]
f_c_fit_L = frequenza_taglio(R_fit_L, L_fit_L)
f_c_err_L = errore_frequenza_taglio(m_L)
f_c_fit_R = frequenza_taglio(R_fit_R, L_fit_R)
f_c_err_R = errore_frequenza_taglio(m_R)

print(m_L)
print(m_R)
print(f"\nR_tot = {R_tot:.3f} ± {sigma_R_tot:.3f} Ohm")

print(f"\nFit su V_B / V_A, induttore:")
print(f"R = {R_fit_L:.3f} ± {R_err_L:.3f} Ohm")
print(f"L = {L_fit_L:.3e} ± {L_err_L:.3e} H")
print(f"tau = L/R = {L_fit_L/R_fit_L:.3e} s")
print(f"f_c = R/(2*pi*L) = {f_c_fit_L:.1f} ± {f_c_err_L:.1f} Hz")
print (f"Chi quadro: {m_L.fval}, ndof: {ndof_L}, p-value: {p_L}")

print(f"\nFit su V_AB / V_A, resistenza:")
print(f"R = {R_fit_R:.3f} ± {R_err_R:.3f} Ohm")
print(f"L = {L_fit_R:.3e} ± {L_err_R:.3e} H")
print(f"tau = L/R = {L_fit_R/R_fit_R:.3e} s")
print(f"f_c = R/(2*pi*L) = {f_c_fit_R:.1f} ± {f_c_err_R:.1f} Hz")
print (f"Chi quadro: {m_R.fval}, ndof: {ndof_L}, p-value: {p_R}")

fig, ax = plt.subplots(2, 1, sharex=True)
x_axis = np.linspace(np.min(frequenza), np.max(frequenza), 1000)

ax[0].set_title ("Rapporto delle ampiezze: $V_{out} / V_{in}$")
ax[0].set_xlabel ("Frequenza (Hz)")
ax[0].set_ylabel("$V_{out} / V_{in}$ (V)")

ax[0].errorbar(frequenza, y_L, yerr=sigma_y_L, fmt="o", capsize = 4, color="mediumseagreen", label="Dati osservati (induttore)")
ax[0].plot(x_axis, H_L(x_axis, R_fit_L, L_fit_L), color="mediumseagreen", label="Funzione di trasferimento $H_L$")

ax[0].errorbar(frequenza, y_R, yerr=sigma_y_R, fmt="o", capsize = 4, color="lightblue", label="Dati osservati (resistenza)")
ax[0].plot(x_axis, H_R(x_axis, R_fit_R, L_fit_R), color="lightblue", label="Funzione di trasferimento $H_R$")

ax[0].legend()
ax[0].grid (True, alpha = 0.4)

ax[1].set_title ("Differenza tra le fasi: $\\Delta \\phi$")
ax[1].set_xlabel("Frequenza (Hz)")
ax[1].set_ylabel("$\\phi_{out} - \\phi_{in}$ (rad)")

ax[1].errorbar(frequenza, fase_A_B, yerr=sigma_fase_A_B, fmt="o", capsize = 4, 
               color="mediumseagreen", label="Dati osservati (induttore)")
ax[1].plot(x_axis, fase_L(x_axis, R_fit_L, L_fit_L), color="mediumseagreen",
           label="Fase teorica")

ax[1].errorbar(frequenza, fase_A_AB, yerr=sigma_fase_A_AB, fmt="o", capsize = 4, 
               color="lightblue", label="Dati osservati (resistenza)")
ax[1].plot(x_axis, fase_R(x_axis, R_fit_R, L_fit_R), color="lightblue",
           label="Fase teorica")

ax[1].set_xlabel("Frequenza (Hz)")
ax[1].set_ylabel("Fase (rad)")
ax[1].legend()
ax[1].grid (True)
plt.show ()


# Grafico aggiuntivo: stessa cosa ma con scala logaritmica sull'asse x
fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].set_xscale('log')
ax[0].set_title ("Rapporto delle ampiezze: $V_{out} / V_{in}$")
ax[0].set_xlabel ("Frequenza (Hz)")
ax[0].set_ylabel("$V_{out} / V_{in}$ (V)")

ax[0].errorbar(frequenza, y_L, yerr=sigma_y_L, fmt="o", capsize = 4, color="mediumseagreen", label="Dati osservati (induttore)")
ax[0].plot(x_axis, H_L(x_axis, R_fit_L, L_fit_L), color="mediumseagreen", label="Funzione di trasferimento $H_L$")

ax[0].errorbar(frequenza, y_R, yerr=sigma_y_R, fmt="o", capsize = 4, color="lightblue", label="Dati osservati (resistenza)")
ax[0].plot(x_axis, H_R(x_axis, R_fit_R, L_fit_R), color="lightblue", label="Funzione di trasferimento $H_R$")

ax[0].legend()
ax[0].grid (True, alpha = 0.4)

ax[1].set_xscale('log')
ax[1].set_title ("Differenza tra le fasi: $\\Delta \\phi$")
ax[1].set_xlabel("Frequenza (Hz)")
ax[1].set_ylabel("$\\phi_{out} - \\phi_{in}$ (rad)")

ax[1].errorbar(frequenza, fase_A_B, yerr=sigma_fase_A_B, fmt="o", capsize = 4, 
               color="mediumseagreen", label="Dati osservati (induttore)")
ax[1].plot(x_axis, fase_L(x_axis, R_fit_L, L_fit_L), color="mediumseagreen",
           label="Fase teorica")

ax[1].errorbar(frequenza, fase_A_AB, yerr=sigma_fase_A_AB, fmt="o", capsize = 4, 
               color="lightblue", label="Dati osservati (resistenza)")
ax[1].plot(x_axis, fase_R(x_axis, R_fit_R, L_fit_R), color="lightblue",
           label="Fase teorica")

ax[1].set_xlabel("Frequenza (Hz)")
ax[1].set_ylabel("Fase (rad)")
ax[1].legend()
ax[1].grid (True)
plt.show ()


# CALCOLO DELLE IMPEDENZE

# Usiamo i parametri dal fit sull'induttore (puoi usare anche m_R, sono compatibili)
R_best = R_fit_L
L_best = L_fit_L
R_err = R_err_L
L_err = L_err_L

# Frequenza di taglio dai due fit, con propagazione da matrice di covarianza.
f_c_L = f_c_fit_L
sigma_fc_L = f_c_err_L
f_c_R = f_c_fit_R
sigma_fc_R = f_c_err_R

print(f"\n--- Frequenze di taglio ---")
print(f"Dal fit sull'induttore: f_c = {f_c_L:.1f} ± {sigma_fc_L:.1f} Hz")
print(f"Dal fit sulla resistenza: f_c = {f_c_R:.1f} ± {sigma_fc_R:.1f} Hz")
compatibilita = abs(f_c_L - f_c_R) / np.sqrt(sigma_fc_L**2 + sigma_fc_R**2)
print(f"Compatibilità tra i due fit: {compatibilita:.2f} sigma")

# Asse delle frequenze esteso (log)
x_imp = np.logspace(np.log10(np.min(frequenza) / 2), np.log10(np.max(frequenza) * 2), 2000)

# Impedenze teoriche dai parametri fittati
Z_R_teorica = np.full_like(x_imp, R_best)          # costante
Z_L_teorica = 2 * np.pi * x_imp * L_best           # ~ f

# Propagazione incertezza su Z_L: sigma_ZL / ZL = sigma_L / L
sigma_Z_L_teorica = Z_L_teorica * (L_err / L_best)

# Impedenze sperimentali stimate dai dati misurati:
#   Z_L / Z_R = H_L / H_R = (V_B / V_AB)
Z_ratio_dati = y_L / y_R
Z_L_dati = Z_ratio_dati * R_tot

# Propagazione incertezza su Z_L_dati
sigma_Z_ratio = Z_ratio_dati * np.sqrt(
    (sigma_y_L / y_L)**2 + (sigma_y_R / y_R)**2
)
sigma_Z_L_dati = np.sqrt(
    (R_tot * sigma_Z_ratio)**2 + (Z_ratio_dati * sigma_R_tot)**2
)

# --- Plot ---
fig_imp, ax_imp = plt.subplots(figsize=(8, 5))

ax_imp.loglog(x_imp, Z_R_teorica, color="lightblue", lw=2,
              label=f"$Z_R$ teorica (fit), $R$ = {R_best:.1f} Ohm")
ax_imp.loglog(x_imp, Z_L_teorica, color="mediumseagreen", lw=2,
              label=f"$Z_L$ teorica (fit), $L$ = {L_best:.2e} H")
ax_imp.fill_between(x_imp,
                    Z_L_teorica - sigma_Z_L_teorica,
                    Z_L_teorica + sigma_Z_L_teorica,
                    color="lightcoral", alpha=0.15,
                    label="Banda $\\pm 1\\sigma$ su $Z_L$")

ax_imp.errorbar(frequenza, Z_L_dati, yerr=sigma_Z_L_dati,
                fmt="o", capsize=4, color="navy",
                label="$Z_L$ stimata dai dati ($R \\cdot V_L / V_R$)")

# Linea verticale sulla frequenza di taglio (media pesata dei due fit)
w_L = 1 / sigma_fc_L**2
w_R = 1 / sigma_fc_R**2
f_c_media = (w_L * f_c_L + w_R * f_c_R) / (w_L + w_R)
sigma_fc_media = 1 / np.sqrt(w_L + w_R)

ax_imp.axvline(f_c_media, color="crimson", lw=1.5, ls="--",
               label=f"$f_c$ = {f_c_media:.0f} Hz")

# Punto di incrocio: Z_R = Z_L  =>  f = R/(2*pi*L), cioe' f_c
ax_imp.plot(f_c_media, R_best, marker="*", markersize=10, color="gold",
            zorder=5, label="Incrocio $Z_R = Z_L$ a $f_c$")

ax_imp.set_xlabel("Frequenza (Hz)")
ax_imp.set_ylabel("Impedenza (Ohm)")
ax_imp.set_title("Impedenze $Z_R$ e $Z_L$ in funzione della frequenza")
ax_imp.legend(fontsize=9)
ax_imp.grid(True, which="both", alpha=0.2)

plt.tight_layout()
plt.show()