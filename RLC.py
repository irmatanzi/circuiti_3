import numpy as np
import matplotlib.pyplot as plt

# Dati:

R = 3000 # Ohm

# DA RIFARE

frequenze = np.array ([500,
                       1000,
                       2000,
                       5000,
                       7500,
                       10000,
                       25000,
                       50000,
                       ]) # Hz

V_A = np.array ([10.0,
                 8.48,
                 7.76,
                 8.96,
                 8.64,
                 np.mean ([9.92, 10.0]),
                 9.92,
                 9.92,
                 ]) # V

V_L = np.array ([9.92,
                 8.40,
                 7.60,
                 8.80,
                 8.40,
                 np.mean ([9.92, 10.0]),
                 9.52,
                 9.44,
                 ]) # V

phi_A = np.array ([970,
                   484,
                   241,
                   95.2,
                   63.0,
                   46.8,
                   17.8,
                   8.1,
                  ]) # microsecondi

phi_L = np.array ([972,
                   486,
                   243,
                   98.7,
                   64.8,
                   48.6,
                   19.4,
                   9.7,
                  ]) # microsecondi

# sensiobilità: 4 microsecondi

# DA CONTROLLARE DIO CANE
fig, ax = plt.subplots (2, 1)
ax[0].plot (frequenze, [x / y for x, y in zip(V_L, V_A)], 'o-', label = 'V_L / V_A')
ax[0].set_xlabel ('Frequenza (Hz)')
ax[0].set_ylabel ('Tensione (V)')
ax[0].legend ()
ax[1].plot (frequenze, phi_A, 'o-', label = 'phi_A')
ax[1].plot (frequenze, phi_L, 'o-', label = 'phi_L')
ax[1].set_xscale ('log')
ax[1].set_xlabel ('Frequenza (Hz)')
ax[1].set_ylabel ('Fase (microsecondi)')
ax[1].legend ()
plt.show ()