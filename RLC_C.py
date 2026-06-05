import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

# FUNZIONE H
def H_C (f, R, L, C):
    omega = 2 * np.pi * f
    num = 1.0
    den = np.sqrt((1 - omega**2 * L * C)**2 + (omega * R * C)**2)
    return num / den


frequenze = np.array ([5000,
                       ])

V_A = np.array ([10.8,
                 ]) # V

V_L = np.array ([1,
                 ]) # V

phi_A = np.array ([8,
                   ]) # microsecondi

phi_L = np.array ([1,
                   ]) # microsecondi

delta_phi_rad = (phi_A - phi_L) * 1e-6 * 2 * np.pi * frequenze

# Errore stimato — aggiusta se hai una stima migliore
y_data = V_L / V_A
y_err = np.full(len(y_data), 0.02)

ls = LeastSquares (frequenze, y_data, y_err,
                   H_C)

# Minuit
m = Minuit(ls, R=250, L=5e-3, C = 10e-9)
m.limits["R"] = (1, 250)
m.limits["L"] = (1e-4, 1e-1)
m.errordef = 1.0  # chi2
m.migrad()
m.hesse()

print(m)

R_fit = m.values["R"]
L_fit = m.values["L"]
R_err = m.errors["R"]
L_err = m.errors["L"]
C_fit = m.values["C"]

print(f"\nR = {R_fit:.1f} ± {R_err:.1f} Ω")
print(f"L = {L_fit*1000:.2f} ± {L_err*1000:.2f} mH")
print(f"f0 = {1/(2*np.pi*np.sqrt(L_fit*C_fit)):.1f} Hz")
print(f"Q  = {(1/R_fit)*np.sqrt(L_fit/C_fit):.2f}")

# Plot
fig, ax = plt.subplots()
x_axis = np.linspace(3000, 80000, 10000)
ax.plot(frequenze, y_data, 'o', label='dati')
ax.plot(x_axis, H_C (x_axis, R_fit, L_fit, C_fit), color='hotpink', label=f'fit: R={R_fit:.0f} Ω, L={L_fit*1000:.1f} mH')
ax.set_xlabel('Frequenza (Hz)')
ax.set_ylabel('V_R / V_A')
ax.legend()
plt.show()