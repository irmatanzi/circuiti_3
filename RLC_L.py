import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

# FUNZIONE H
def H_L (f, R, L, C):
    omega = 2 * np.pi * f
    num = omega**2 * L * C
    den = np.sqrt((1 - omega**2 * L * C)**2 + (omega * R * C)**2)
    return num / den

def H_L_claudio (f, R, L, C):
    omega = 2 * np.pi * f
    num = omega * L
    den = np.sqrt(R**2 + (omega * L - 1/(omega * C))**2)
    return num / den


frequenze = np.array ([5000,
                       7000,
                       8000,
                       10000,
                       15000,
                       20000,
                       50000,
                       70000
                       ])

V_A = np.array ([10.3,
                 10.0,
                 10.15,
                 10.9, #oscilla tra 10.8 e 11
                 11.9,
                 12.4,
                 12.7,
                 12.8
                 ]) # V

V_L = np.array ([4.88,
                 8.24,
                 9.20,
                 10.2,
                 10.6,
                 10.4,
                 10.2,
                 10.1
                 ]) # V

phi_A = np.array ([86.4,
                   61.4,
                   54.0,
                   43.4,
                   29.4,
                   22.2,
                   8.88,
                   6.32
                   ]) # microsecondi

phi_L = np.array ([11.2,
                   19.6,
                   22.8,
                   25.0,
                   21.8,
                   17.7,
                   7.91,
                   5.72
                   ]) # microsecondi

delta_phi_rad = (phi_A - phi_L) * 1e-6 * 2 * np.pi * frequenze

# Errore stimato — aggiusta se hai una stima migliore
y_data = V_L / V_A
y_err = np.full(len(y_data), 0.02)

ls = LeastSquares (frequenze, y_data, y_err,
                   H_L)

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
ax.plot(x_axis, H_L(x_axis, R_fit, L_fit, C_fit), color='hotpink', label=f'fit: R={R_fit:.0f} Ω, L={L_fit*1000:.1f} mH')
ax.set_xlabel('Frequenza (Hz)')
ax.set_ylabel('V_R / V_A')
ax.legend()
plt.show()