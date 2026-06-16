import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

# raga nei grafici, vi piace di più "V_out" e "V_in" o "V_C" e "V_g"?


def H_C(f, R, C):
    omega = 2 * np.pi * f
    return 1 / np.sqrt(1 + (omega * R * C)**2)


def H_R(f, R, C):
    omega = 2 * np.pi * f
    return (omega * R * C) / np.sqrt(1 + (omega * R * C)**2)


def fase_C(f, R, C):
    omega = 2 * np.pi * f
    return -np.arctan(omega * R * C)


def fase_R(f, R, C):
    omega = 2 * np.pi * f
    return np.pi / 2 - np.arctan(omega * R * C)


# Dati:

R_load = 3000.2 # Ohm
# R_C ignota (multimetro in overload)
C_start = 1e-9 # F, misurato con massima risoluzione 1 nF e ± (1,2% + 2)

# Incertezze di tipo B: distribuzione uniforme nell'intervallo dichiarato.
sigma_R_load = 0.01 * R_load / np.sqrt(3) # Ohm, tolleranza produttore: +/- 1%

# Incertezze di risoluzione: distribuzione uniforme sull'ultima cifra.
sigma_V_risoluzione = 0.001 / np.sqrt(12) # V, ultima cifra letta = 0.001 V
sigma_frequenza = 1 / np.sqrt(12) # Hz, ultima cifra letta = 1 Hz

frequenza = np.array([  200.,   500.,  1000.,  2000.,  5000.,  7000., 10000., 20000., 30000.,
 40000., 50000., 60000.]) # Hz

V_A = np.array([ 9.964, 10.038,  9.984, 10.064, 10.027, 10.055,  9.867, 10.023, 10.072,
  9.989, 10.052,  9.967]) # V, segnale in ingresso
V_B = np.array([ 9.968,  9.995, 10.028,  9.899,  9.934,  9.895,  9.798,  9.041,  8.238,
  7.404,  6.714,  5.854]) # V, tensione sul condensatore: passa basso
V_AB = np.array([0.119, 0.13 , 0.188, 0.486, 1.181, 1.521, 2.187, 4.031, 5.527, 6.787,
 7.497, 8.01 ]) # V, tensione sulla resistenza: passa alto

sigma_V_A = np.array([0.065, 0.037, 0.052, 0.042, 0.031, 0.037, 0.034, 0.087, 0.028, 0.045,
 0.032, 0.043])
sigma_V_B = np.array([0.041, 0.056, 0.054, 0.046, 0.039, 0.038, 0.065, 0.103, 0.098, 0.038,
 0.047, 0.06 ])
sigma_V_AB = np.array([0.056, 0.109, 0.042, 0.049, 0.089, 0.055, 0.07 , 0.064, 0.044, 0.07 ,
 0.023, 0.072])

phi_A = np.array([ 0.101,  0.012,  0.001, -0.214, -0.173,  0.084,  0.067,  0.036, -0.14 ,
 -0.059, -0.146,  0.005]) * 1e-6 # secondi
phi_B = np.array([-3.477, -3.389, -3.471, -3.389, -3.469, -3.63 , -3.602, -3.479, -3.133,
 -2.889, -2.719, -2.509]) * 1e-6 # secondi
phi_AB = np.array([1246.454,  496.209,  246.566,  121.528,   46.537,   32.262,   21.63 ,
    9.06 ,    5.233,    3.485,    2.365,    1.778]) * 1e-6 # secondi

sigma_phi_A = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])* 1e-6 # secondi
sigma_phi_B = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])* 1e-6 # secondi
sigma_phi_AB = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])* 1e-6 # secondi



sigma_V_A = np.sqrt(sigma_V_A**2 + sigma_V_risoluzione**2)
sigma_V_B = np.sqrt(sigma_V_B**2 + sigma_V_risoluzione**2)
sigma_V_AB = np.sqrt(sigma_V_AB**2 + sigma_V_risoluzione**2)


# V_A è il segnale in entrata.
# V_B è la tensione sul condensatore: passa basso.
# V_AB è la tensione sulla resistenza: passa alto.
y_C = V_B / V_A
y_R = V_AB / V_A

sigma_y_C = y_C * np.sqrt(
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

ls_C = LeastSquares(frequenza, y_C, sigma_y_C, H_C)
ls_R = LeastSquares(frequenza, y_R, sigma_y_R, H_R)


def costo_con_vincolo_R(ls):
    def costo(R, C):
        return ls(R, C) + ((R - R_load) / sigma_R_load)**2
    return costo


def frequenza_taglio(R, C):
    return 1 / (2 * np.pi * R * C)


def errore_frequenza_taglio(m):
    R_fit = m.values["R"]
    C_fit = m.values["C"]
    f_c = frequenza_taglio(R_fit, C_fit)
    cov = m.covariance

    df_dR = -f_c / R_fit
    df_dC = -f_c / C_fit
    var_f_c = (
        df_dR**2 * cov["R", "R"]
        + df_dC**2 * cov["C", "C"]
        + 2 * df_dR * df_dC * cov["R", "C"]
    )
    return np.sqrt(var_f_c)


m_C = Minuit(costo_con_vincolo_R(ls_C), R=R_load, C=C_start)
m_C.errordef = Minuit.LEAST_SQUARES
m_C.limits["R"] = (1e-9, None)
m_C.limits["C"] = (1e-10, 1e-3)
m_C.migrad()
m_C.hesse()

ndof_C = len(frequenza) + 1 - 2
p_C = chi2.sf(m_C.fval, ndof_C)

m_R = Minuit(costo_con_vincolo_R(ls_R), R=R_load, C=C_start)
m_R.errordef = Minuit.LEAST_SQUARES
m_R.limits["R"] = (1e-9, None)
m_R.limits["C"] = (1e-10, 1e-3)
m_R.migrad()
m_R.hesse()

p_R = chi2.sf(m_R.fval, ndof_C)

R_fit_C = m_C.values["R"]
R_err_C = m_C.errors["R"]
C_fit_C = m_C.values["C"]
C_err_C = m_C.errors["C"]
R_fit_R = m_R.values["R"]
R_err_R = m_R.errors["R"]
C_fit_R = m_R.values["C"]
C_err_R = m_R.errors["C"]
f_c_fit_C = frequenza_taglio(R_fit_C, C_fit_C)
f_c_err_C = errore_frequenza_taglio(m_C)
f_c_fit_R = frequenza_taglio(R_fit_R, C_fit_R)
f_c_err_R = errore_frequenza_taglio(m_R)

print(m_C)
print(m_R)
print(f"\nR_load = {R_load:.3f} ± {sigma_R_load:.3f} Ohm")

print(f"\nFit su V_B / V_A, condensatore:")
print(f"R = {R_fit_C:.3f} ± {R_err_C:.3f} Ohm")
print(f"C = {C_fit_C:.3e} ± {C_err_C:.3e} F")
print(f"tau = R*C = {R_fit_C*C_fit_C:.3e} s")
print(f"f_c = 1/(2*pi*R*C) = {f_c_fit_C:.1f} ± {f_c_err_C:.1f} Hz")
print(f"Chi quadro: {m_C.fval}, Ndof: {ndof_C}, p-value: {p_C}")

print(f"\nFit su V_AB / V_A, resistenza:")
print(f"R = {R_fit_R:.3f} ± {R_err_R:.3f} Ohm")
print(f"C = {C_fit_R:.3e} ± {C_err_R:.3e} F")
print(f"tau = R*C = {R_fit_R*C_fit_R:.3e} s")
print(f"f_c = 1/(2*pi*R*C) = {f_c_fit_R:.1f} ± {f_c_err_R:.1f} Hz")
print(f"Chi quadro: {m_R.fval}, Ndof: {ndof_C}, p-value: {p_R}")


ls_C = LeastSquares (frequenza, fase_A_B, sigma_fase_A_B, fase_C)
ls_R = LeastSquares (frequenza, fase_A_AB, sigma_fase_A_AB, fase_R)

m_C_fase = Minuit(costo_con_vincolo_R(ls_C), R=R_load, C=C_start)
m_C_fase.errordef = Minuit.LEAST_SQUARES
m_C_fase.limits["R"] = (1e-9, None)
m_C_fase.limits["C"] = (1e-10, 1e-3)
m_C_fase.migrad()
m_C_fase.hesse()

p_C_fase = chi2.sf(m_C_fase.fval, ndof_C)

m_R_fase = Minuit(costo_con_vincolo_R(ls_R), R=R_load, C=C_start)
m_R_fase.errordef = Minuit.LEAST_SQUARES
m_R_fase.limits["R"] = (1e-9, None)
m_R_fase.limits["C"] = (1e-10, 1e-3)
m_R_fase.migrad()
m_R_fase.hesse()

p_R_fase = chi2.sf(m_R_fase.fval, ndof_C)

print(f"\nFit sulla fase, condensatore:")
print(m_C_fase)
print(f"R = {m_C_fase.values['R']:.3f} ± {m_C_fase.errors['R']:.3f} Ohm")
print(f"C = {m_C_fase.values['C']:.3e} ± {m_C_fase.errors['C']:.3e} F")
print(f"Chi quadro: {m_C_fase.fval}, Ndof: {ndof_C}, p-value: {p_C_fase}")
print(f"\nFit sulla fase, resistenza:")
print(m_R_fase)
print(f"R = {m_R_fase.values['R']:.3f} ± {m_R_fase.errors['R']:.3f} Ohm")
print(f"C = {m_R_fase.values['C']:.3e} ± {m_R_fase.errors['C']:.3e} F")
print(f"Chi quadro: {m_R_fase.fval}, Ndof: {ndof_C}, p-value: {p_R_fase}")


fig, ax = plt.subplots(2, 1, sharex=True)
x_axis = np.linspace(np.min(frequenza), np.max(frequenza), 1000)

ax[0].set_title ("Rapporto delle ampiezze $V_{out} / V_{in}$ - scala lineare")
ax[0].set_xlabel ("Frequenza (Hz)")
ax[0].set_ylabel("$V_{out} / V_{in}$ (V)")

ax[0].errorbar (frequenza, y_C, yerr=sigma_y_C, fmt="o", capsize = 4, color="hotpink", label="Dati osservati (condensatore)")
ax[0].plot(x_axis, H_C(x_axis, R_fit_C, C_fit_C), color="hotpink",
           label="Funzione di trasferimento $H_C$")
ax[0].errorbar(frequenza, y_R, yerr=sigma_y_R, fmt="o", capsize = 4, color="lightblue", label="Dati osservati (resistenza)")
ax[0].plot(x_axis, H_R(x_axis, R_fit_R, C_fit_R), color="lightblue",
           label="Funzione di trasferimento $H_R$")
ax[0].legend()
ax[0].grid(True, alpha = 0.4)

ax[1].set_title ("Differenza tra le fasi $\\Delta \\phi$ - scala lineare")
ax[1].set_xlabel("Frequenza (Hz)")
ax[1].set_ylabel("$\\phi_{out} - \\phi_{in}$ (rad)")

ax[1].errorbar(frequenza, fase_A_B, yerr=sigma_fase_A_B, fmt="o", capsize = 4, 
               color="hotpink", label="Dati osservati (condensatore)")
ax[1].plot(x_axis, fase_C(x_axis, R_fit_C, C_fit_C), color="hotpink",
           label="Fase teorica")
ax[1].errorbar(frequenza, fase_A_AB, yerr=sigma_fase_A_AB, fmt="o", capsize = 4, 
               color="lightblue", label="Dati osservati (resistenza)")
ax[1].plot(x_axis, fase_R(x_axis, R_fit_R, C_fit_R), color="lightblue",
           label="Fase teorica")

ax[1].legend()
ax[1].grid(True, alpha = 0.4)

plt.tight_layout()
plt.show()

# grafico logaritmico

fig, ax = plt.subplots (2,1)

ax[0].set_title ("Rapporto delle ampiezze $V_{out} / V_{in}$ - scala logaritmica")
ax[0].set_xlabel ("Frequenza (Hz)")
ax[0].set_ylabel("$V_{out} / V_{in}$ (V)")

ax[0].errorbar (frequenza, y_C, yerr=sigma_y_C, fmt="o", capsize = 4, color="hotpink", label="Dati osservati (condensatore)")
ax[0].plot(x_axis, H_C(x_axis, R_fit_C, C_fit_C), color="hotpink",
           label="Funzione di trasferimento $H_C$")
ax[0].errorbar(frequenza, y_R, yerr=sigma_y_R, fmt="o", capsize = 4, color="lightblue", label="Dati osservati (resistenza)")
ax[0].plot(x_axis, H_R(x_axis, R_fit_R, C_fit_R), color="lightblue",
           label="Funzione di trasferimento $H_R$")

ax[0].set_xscale('log')
ax[0].legend()
ax[0].grid(True, alpha = 0.4)

ax[1].set_title ("Differenza tra le fasi $\\Delta \\phi$ - scala logaritmica")
ax[1].set_xlabel("Frequenza (Hz)")
ax[1].set_ylabel("$\\phi_{out} - \\phi_{in}$ (rad)")

ax[1].errorbar(frequenza, fase_A_B, yerr=sigma_fase_A_B, fmt="o", capsize = 4, 
               color="hotpink", label="Dati osservati (condensatore)")
ax[1].plot(x_axis, fase_C(x_axis, R_fit_C, C_fit_C), color="hotpink")
ax[1].errorbar(frequenza, fase_A_AB, yerr=sigma_fase_A_AB, fmt="o", capsize = 4, 
               color="lightblue", label="Dati osservati (resistenza)")
ax[1].plot(x_axis, fase_R(x_axis, R_fit_R, C_fit_R), color="lightblue")

ax[1].set_xscale ('log')
ax[1].legend()
ax[1].grid(True, alpha = 0.4)

plt.tight_layout()
plt.show ()

# CALCOLO DELLE IMPEDENZE

# Usiamo i parametri dal fit sul condensatore (puoi usare anche m_R, sono compatibili)
R_best = R_fit_C
C_best = C_fit_C
R_err  = R_err_C
C_err  = C_err_C
 
# Frequenza di taglio dai due fit delle ampiezze, con propagazione da matrice di covarianza.
f_c_C = f_c_fit_C
sigma_fc_C = f_c_err_C
f_c_R = f_c_fit_R
sigma_fc_R = f_c_err_R

# Frequenze di taglio dai due fit della fase, con propagazione da matrice di covarianza.
f_c_C_fase = frequenza_taglio(m_C_fase.values["R"], m_C_fase.values["C"])
sigma_fc_C_fase = errore_frequenza_taglio(m_C_fase)
f_c_R_fase = frequenza_taglio(m_R_fase.values["R"], m_R_fase.values["C"])
sigma_fc_R_fase = errore_frequenza_taglio(m_R_fase)
 
print(f"\n--- Frequenze di taglio ---")
print(f"Dal fit sul condensatore: f_c = {f_c_C:.1f} ± {sigma_fc_C:.1f} Hz")
print(f"Dal fit sulla resistenza: f_c = {f_c_R:.1f} ± {sigma_fc_R:.1f} Hz")
compatibilita = abs(f_c_C - f_c_R) / np.sqrt(sigma_fc_C**2 + sigma_fc_R**2)
print(f"Compatibilità tra i due fit: {compatibilita:.2f} sigma")
print(f"\nDal fit sulla fase, condensatore: f_c = {f_c_C_fase:.1f} ± {sigma_fc_C_fase:.1f} Hz")
print(f"Dal fit sulla fase, resistenza: f_c = {f_c_R_fase:.1f} ± {sigma_fc_R_fase:.1f} Hz")
compatibilita_fase = abs(f_c_C_fase - f_c_R_fase) / np.sqrt(sigma_fc_C_fase**2 + sigma_fc_R_fase**2)
print(f"Compatibilità tra i due fit sulla fase: {compatibilita_fase:.2f} sigma")
 
# Asse delle frequenze esteso (log) per un bel grafico
x_log = np.logspace(np.log10(100), np.log10(50000), 2000)
 
# Impedenze teoriche dai parametri fittati
Z_R_teorica = np.full_like(x_log, R_best)          # costante
Z_C_teorica = 1 / (2 * np.pi * x_log * C_best)     # ~ 1/f
 
# Propagazione incertezza su Z_C: sigma_ZC / ZC = sigma_C / C
sigma_Z_C_teorica = Z_C_teorica * (C_err / C_best)
 
# Impedenze "sperimentali" stimate dai dati misurati:
#   Dal partitore di tensione:  H_C = Z_C / (Z_R + Z_C)  =>  Z_C / Z_R = H_C / (1 - H_C)  ... però
#   più diretto: Z_C = R * (V_B / V_AB), perché V_B/V_A = H_C e V_AB/V_A = H_R
#   quindi Z_C / Z_R = H_C / H_R = (V_B / V_AB)
Z_ratio_dati = y_C / y_R   # = Z_C / Z_R sperimentale
Z_C_dati = Z_ratio_dati * R_load  # Ohm (usiamo R_load come riferimento)
 
# Propagazione incertezza su Z_C_dati
sigma_Z_ratio = Z_ratio_dati * np.sqrt(
    (sigma_y_C / y_C)**2 + (sigma_y_R / y_R)**2
)
sigma_Z_C_dati = np.sqrt(
    (R_load * sigma_Z_ratio)**2 + (Z_ratio_dati * sigma_R_load)**2
)

# Il primo punto (200 Hz) ha V_AB molto piccola e quindi errore relativo > 100%.
# Lo escludiamo solo dal grafico delle impedenze, non dai fit precedenti.
maschera_impedenze = frequenza != 200
 
# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5))
 
ax.loglog(x_log, Z_R_teorica, color="lightblue",  lw=2, label=f"$Z_R$ teorica (fit), $R$ = {R_best:.0f} Ω")
ax.loglog(x_log, Z_C_teorica, color="hotpink",    lw=2, label=f"$Z_C$ teorica (fit), $C$ = {C_best:.2e} F")
ax.fill_between(x_log,
                Z_C_teorica - sigma_Z_C_teorica,
                Z_C_teorica + sigma_Z_C_teorica,
                color="hotpink", alpha=0.1, label="Banda $\\pm 1\\sigma$ su $Z_C$")
 
ax.errorbar(frequenza[maschera_impedenze],
            Z_C_dati[maschera_impedenze],
            yerr=sigma_Z_C_dati[maschera_impedenze],
            fmt="o", capsize=4, color="navy",
            label="$Z_C$ stimata dai dati ($R \\cdot V_C / V_{R}$)")
 
# Linea verticale sulla frequenza di taglio (media pesata dei due fit)
w_C = 1 / sigma_fc_C**2
w_R = 1 / sigma_fc_R**2
f_c_media = (w_C * f_c_C + w_R * f_c_R) / (w_C + w_R)
sigma_fc_media = 1 / np.sqrt(w_C + w_R)
 
ax.axvline(f_c_media, color="crimson", lw=1.5, ls="--",
           label=f"$f_c$ = {f_c_media:.0f} Hz")
 
# Punto di incrocio: Z_R = Z_C  =>  f = 1/(2*pi*R*C), cioè esattamente f_c
ax.plot(f_c_media, R_best, marker="*", markersize=10, color="gold",
        zorder=5, label=f"Incrocio $Z_R = Z_C$ a $f_c$")
 
ax.set_xlabel("Frequenza (Hz)")
ax.set_ylabel("Impedenza (Ω)")
ax.set_title("Impedenze $Z_R$ e $Z_C$ in funzione della frequenza")
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.2)
 
plt.tight_layout()
plt.show()
