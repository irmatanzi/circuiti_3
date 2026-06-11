import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

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

# i dati veri stanno nella cartella dati

frequenza = np.array([200., 500., 2000., 5000., 10000., 25000., 50000., 100000., 150000.])

V_A = np.array([10.053, 10.036,  9.948,  9.926, 10.04 , 10.098,  9.898, 10.038, 10.038])
V_B = np.array([ 2.101,  4.752,  9.066,  9.845,  9.945, 10.018, 10.01 , 10.074,  9.925])
V_AB = np.array([9.711, 8.747, 4.194, 1.86 , 0.894, 0.288, 0.142, 0.083, 0.023])

sigma_V_A = np.array([0.02 , 0.026, 0.057, 0.017, 0.019, 0.048, 0.031, 0.028, 0.045])
sigma_V_B = np.array([0.04 , 0.029, 0.043, 0.039, 0.039, 0.014, 0.051, 0.056, 0.073])
sigma_V_AB = np.array([0.063, 0.032, 0.017, 0.01 , 0.023, 0.067, 0.025, 0.038, 0.067])

phi_A = np.array([ 2.857, -0.448, -2.635,  3.798,  2.8  , -1.833,  0.676, -1.035,  1.873]) * 1e-6 # secondi
phi_B = np.array([ 1.078e+03,  3.435e+02,  3.728e+01,  7.084e+00, -2.489e+00,  1.588e-01, 3.128e-01,  1.664e+00, -1.112e+00]) * 1e-6 # secondi
phi_AB = np.array([-167.476, -159.65 ,  -89.006,  -42.309,  -21.566,  -11.354,   -3.426, -0.257,   -3.032]) * 1e-6 # secondi

sigma_phi_A = np.array([2.084, 1.109, 1.31 , 1.029, 1.852, 0.581, 1.215, 1.977, 2.247]) * 1e-6 # secondi
sigma_phi_B = np.array([2.073, 3.112, 1.287, 1.337, 1.768, 1.214, 2.003, 0.56 , 1.119]) * 1e-6 # secondi
sigma_phi_AB = np.array([2.44 , 1.452, 2.29 , 1.125, 3.562, 1.638, 3.106, 3.011, 2.127]) * 1e-6 # secondi

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

sigma_fase_A_B = (
    2*np.pi*frequenza
    * np.sqrt(sigma_phi_A**2 + sigma_phi_B**2)
)

sigma_fase_A_AB = (
    2*np.pi*frequenza
    * np.sqrt(sigma_phi_A**2 + sigma_phi_AB**2)
)

ls_L = LeastSquares(frequenza, y_L, sigma_y_L, H_L)
ls_R = LeastSquares(frequenza, y_R, sigma_y_R, H_R)

m_L = Minuit(ls_L, R = R_tot, L=10e-3)
m_L.limits["L"] = (1e-10, 1e-1)
m_L.fixed["R"] = True
m_L.migrad()
m_L.hesse()

m_R = Minuit(ls_R, R = R_tot, L=10e-3)
m_R.limits["L"] = (1e-10, 1e-1)
m_R.fixed["R"] = True
m_R.migrad()
m_R.hesse()


L_fit_L = m_L.values["L"]
L_err_L = m_L.errors["L"]
L_fit_R = m_R.values["L"]
L_err_R = m_R.errors["L"]

print(m_L)
print(m_R)
print(f"\nFit su V_B / V_A, condensatore:")
print(f"L = {L_fit_L:.3e} ± {L_err_L:.3e} H")
print(f"tau = R*L = {R_tot*L_fit_L:.3e} s")
print(f"f_c = {1/(2*np.pi*R_tot*L_fit_L):.1f} Hz")

print(f"\nFit su V_AB / V_A, resistenza:")
print(f"L = {L_fit_R:.3e} ± {L_err_R:.3e} H")
print(f"tau = R*L = {R_tot*L_fit_R:.3e} s")
print(f"f_c = {1/(2*np.pi*R_tot*L_fit_R):.1f} Hz")

fig, ax = plt.subplots(2, 1, sharex=True)
x_axis = np.linspace(np.min(frequenza), np.max(frequenza), 1000)

ax[0].errorbar(frequenza, y_L, yerr=sigma_y_L, fmt="o", capsize = 4, color="hotpink", label="V_B / V_A")
ax[0].plot(x_axis, H_L(x_axis, R_tot, L_fit_L), color="hotpink", label=f"fit L: L={L_fit_L:.2e} H")

ax[0].errorbar(frequenza, y_R, yerr=sigma_y_R, fmt="o", capsize = 4, color="lightblue", label="V_AB / V_A")
ax[0].plot(x_axis, H_R(x_axis, R_tot, L_fit_R), color="lightblue", label=f"fit R: L={L_fit_R:.2e} H")

ax[0].set_ylabel("Rapporto di ampiezza")
ax[0].legend()
ax[0].grid (True)

ax[1].errorbar(frequenza, fase_A_B, yerr=sigma_fase_A_B, fmt="o", capsize = 4, color="hotpink", label="fase V_B - V_A")
ax[1].plot(x_axis, fase_L(x_axis, R_tot, L_fit_L), color="hotpink")

ax[1].errorbar(frequenza, fase_A_AB, yerr=sigma_fase_A_AB, fmt="o", capsize = 4, color="lightblue", label="fase V_AB - V_A")
ax[1].plot(x_axis, fase_R(x_axis, R_tot, L_fit_R), color="lightblue")

ax[1].set_xlabel("Frequenza (Hz)")
ax[1].set_ylabel("Fase (rad)")
ax[1].legend()
ax[1].grid (True)
plt.show ()

# Grafico aggiuntivo: stessa cosa ma con scala logaritmica sull'asse x
fig_log, ax_log = plt.subplots(2, 1, sharex=True)
ax_log[0].set_xscale('log')
x_axis_log = np.logspace(np.log10(np.min(frequenza)), np.log10(np.max(frequenza)), 1000)

ax_log[0].errorbar(frequenza, y_L, yerr=sigma_y_L, fmt="o", capsize=4, color="hotpink", label="V_B / V_A")
ax_log[0].plot(x_axis_log, H_L(x_axis_log, R_tot, L_fit_L), color="hotpink", label=f"fit L: L={L_fit_L:.2e} H")

ax_log[0].errorbar(frequenza, y_R, yerr=sigma_y_R, fmt="o", capsize=4, color="lightblue", label="V_AB / V_A")
ax_log[0].plot(x_axis_log, H_R(x_axis_log, R_tot, L_fit_R), color="lightblue", label=f"fit R: L={L_fit_R:.2e} H")

ax_log[0].set_ylabel("Rapporto di ampiezza")
ax_log[0].legend()
ax_log[0].grid(True)
ax_log[1].set_xscale('log')
ax_log[1].errorbar(frequenza, fase_A_B, yerr=sigma_fase_A_B, fmt="o", capsize=4, color="hotpink", label="fase V_B - V_A")
ax_log[1].plot(x_axis_log, fase_L(x_axis_log, R_tot, L_fit_L), color="hotpink")

ax_log[1].errorbar(frequenza, fase_A_AB, yerr=sigma_fase_A_AB, fmt="o", capsize=4, color="lightblue", label="fase V_AB - V_A")
ax_log[1].plot(x_axis_log, fase_R(x_axis_log, R_tot, L_fit_R), color="lightblue")

ax_log[1].set_xlabel("Frequenza (Hz)")
ax_log[1].set_ylabel("Fase (rad)")
ax_log[1].legend()
ax_log[1].grid(True)

plt.show ()