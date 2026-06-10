import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares


def H_C(f, C):
    omega = 2 * np.pi * f
    return 1 / np.sqrt(1 + (omega * R_load * C)**2)


def H_R(f, C):
    omega = 2 * np.pi * f
    return (omega * R_load * C) / np.sqrt(1 + (omega * R_load * C)**2)


def fase_C(f, C):
    omega = 2 * np.pi * f
    return -np.arctan(omega * R_load * C)


def fase_R(f, C):
    omega = 2 * np.pi * f
    return np.pi / 2 - np.arctan(omega * R_load * C)


# Dati:

R_load = 3000 # Ohm
# R_C ignota (multimetro in overload)
C_start = 1e-9 # F, solo valore iniziale per il fit

frequenza = np.array([  200.,   500.,  1000.,  2000.,  5000.,  7000., 10000.]) # Hz

V_A = np.array([10.017, 10.039, 10.035,  9.999,  9.966, 10.022,  9.965]) # V, segnale in ingresso
V_B = np.array([10.02 , 10.005, 10.038,  9.971,  9.959,  9.876,  9.794]) # V, tensione sul condensatore: passa basso
V_AB = np.array([0.008, 0.041, 0.246, 0.474, 1.056, 1.507, 2.24 ]) # V, tensione sulla resistenza: passa alto

sigma_V_A = np.array([0.032, 0.028, 0.032, 0.038, 0.063, 0.027, 0.017])
sigma_V_B = np.array([0.017, 0.043, 0.026, 0.029, 0.048, 0.09 , 0.07 ])
sigma_V_AB = np.array([0.043, 0.042, 0.008, 0.038, 0.022, 0.048, 0.003])

phi_A = np.array([-1.581, -0.712,  0.247, -5.661, -0.475, -1.302,  2.044]) * 1e-6 # secondi
phi_B = np.array([ -7.234,  -1.747,  -1.   , -11.155,  -2.094,   0.288,   0.487]) * 1e-6 # secondi
phi_AB = np.array([1246.141,  494.794,  248.88 ,  121.465,   48.02 ,   33.712,   20.169]) * 1e-6 # secondi

sigma_phi_A = np.array([1.16 , 0.533, 3.15 , 1.834, 2.787, 3.628, 0.786])* 1e-6 # secondi
sigma_phi_B = np.array([2.27 , 2.387, 4.045, 1.63 , 0.797, 3.2  , 2.211])* 1e-6 # secondi
sigma_phi_AB = np.array([2.017, 2.664, 0.916, 0.121, 4.425, 1.715, 1.499])* 1e-6 # secondi

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

sigma_fase_A_B = (
    2*np.pi*frequenza
    * np.sqrt(sigma_phi_A**2 + sigma_phi_B**2)
)

sigma_fase_A_AB = (
    2*np.pi*frequenza
    * np.sqrt(sigma_phi_A**2 + sigma_phi_AB**2)
)

ls_C = LeastSquares(frequenza, y_C, sigma_y_C, H_C)
ls_R = LeastSquares(frequenza, y_R, sigma_y_R, H_R)

m_C = Minuit(ls_C, C=C_start)
m_C.limits["C"] = (1e-10, 1e-3)
m_C.migrad()
m_C.hesse()

m_R = Minuit(ls_R, C=C_start)
m_R.limits["C"] = (1e-10, 1e-3)
m_R.migrad()
m_R.hesse()

C_fit_C = m_C.values["C"]
C_err_C = m_C.errors["C"]
C_fit_R = m_R.values["C"]
C_err_R = m_R.errors["C"]

print(m_C)
print(m_R)
print(f"\nFit su V_B / V_A, condensatore:")
print(f"C = {C_fit_C:.3e} ± {C_err_C:.3e} F")
print(f"tau = R*C = {R_load*C_fit_C:.3e} s")
print(f"f_c = {1/(2*np.pi*R_load*C_fit_C):.1f} Hz")

print(f"\nFit su V_AB / V_A, resistenza:")
print(f"C = {C_fit_R:.3e} ± {C_err_R:.3e} F")
print(f"tau = R*C = {R_load*C_fit_R:.3e} s")
print(f"f_c = {1/(2*np.pi*R_load*C_fit_R):.1f} Hz")

fig, ax = plt.subplots(2, 1, sharex=True)
x_axis = np.linspace(np.min(frequenza), np.max(frequenza), 1000)

ax[0].errorbar(frequenza, y_C, yerr=sigma_y_C, fmt="o", capsize = 4, color="hotpink", label="V_B / V_A")
ax[0].plot(x_axis, H_C(x_axis, C_fit_C), color="hotpink",
           label=f"fit C: C={C_fit_C:.2e} F")
ax[0].errorbar(frequenza, y_R, yerr=sigma_y_R, fmt="o", capsize = 4, color="lightblue", label="V_AB / V_A")
ax[0].plot(x_axis, H_R(x_axis, C_fit_R), color="lightblue",
           label=f"fit R: C={C_fit_R:.2e} F")
ax[0].set_ylabel("Rapporto di ampiezza")
ax[0].legend()

ax[1].errorbar(frequenza, fase_A_B, yerr=sigma_fase_A_B, fmt="o", capsize = 4, color="hotpink", label="fase V_B - V_A")
ax[1].plot(x_axis, fase_C(x_axis, C_fit_C), color="hotpink")
ax[1].errorbar(frequenza, fase_A_AB, yerr=sigma_fase_A_AB, fmt="o", capsize = 4, color="lightblue", label="fase V_AB - V_A")
ax[1].plot(x_axis, fase_R(x_axis, C_fit_R), color="lightblue")
ax[1].set_xlabel("Frequenza (Hz)")
ax[1].set_ylabel("Fase (rad)")
ax[1].legend()

plt.show()