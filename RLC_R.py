import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

# Funzione di trasferimento ai capi di R DA AGGIUNGERE OFFSET
def H_R(f, R, L, C):
    omega = 2 * np.pi * f
    num = omega * R * C
    den = np.sqrt((1 - omega**2 * L * C)**2 + (omega * R * C)**2)
    return num / den

# Dati:

C_misurata = 10e-9 # F
R_load = 200 # Ohm

# DA RIFARE

frequenze = np.array ([4000,
                       5000,
                       6000,
                       7000,
                       8000,
                       9000,
                       10000,
                       20000,
                       30000,
                       50000
                       ]) # Hz

# RESISTENZA

V_A = np.array ([11.1, 
                 10.7,
                 10.4,
                 10.3,
                 10.3,
                 10.6,
                 11.0,
                 12.45,
                 12.7,
                 12.8
                 ]) # V

V_R = np.array ([5.36,
                 6.52,
                 7.44,
                 7.76,
                 7.76,
                 7.28,
                 6.80,
                 3.54,
                 2.32,
                 1.36
                 ]) # V

phi_A = np.array ([109,
                   86.8,
                   72.0,
                   61.4,
                   54.0,
                   48.2,
                   43.6,
                   22.2,
                   14.8,
                   8.88
                  ]) # microsecondi

phi_R = np.array ([78.0,
                   69.2,
                   64.4,
                   61.8,
                   59.2,
                   56.8,
                   54.0,
                   32.0,
                   22.0,
                   13.5
                  ]) # microsecondi

# sensiobilità: 4 microsecondi

delta_phi_rad = (phi_A - phi_R) * 1e-6 * 2 * np.pi * frequenze

# Errore stimato — aggiusta se hai una stima migliore
y_data = V_R / V_A
y_err = np.full(len(y_data), 0.02)

ls = LeastSquares (frequenze, y_data, y_err,
                   H_R)

# Minuit
m = Minuit(ls, R=208, L=25e-3, C = 10e-9)
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
ax.plot(x_axis, H_R(x_axis, R_fit, L_fit, C_fit), color='hotpink', label=f'fit: R={R_fit:.0f} Ω, L={L_fit*1000:.1f} mH')
ax.set_xlabel('Frequenza (Hz)')
ax.set_ylabel('V_R / V_A')
ax.legend()
plt.show()

'''
# DA CONTROLLARE DIO CANE
fig, ax = plt.subplots (2, 1)
x_axis = np.linspace (5000, 70000, 10000)
ax[0].plot (frequenze, [x / y for x, y in zip(V_R, V_A)], 'o-', label = 'V_C / V_A')
ax[0].plot (x_axis, [H (x, R + 8.3, L, C) for x in x_axis], color = 'hotpink')
ax[0].set_xlabel ('Frequenza (Hz)')
ax[0].set_ylabel ('Tensione (V)')
ax[0].legend ()
ax[1].plot (frequenze, delta_phi_rad, 'o-', label = 'delta phi')
ax[1].set_xscale ('log')
ax[1].set_xlabel ('Frequenza (Hz)')
ax[1].set_ylabel ('Fase (microsecondi)')
ax[1].legend ()
plt.show ()
'''