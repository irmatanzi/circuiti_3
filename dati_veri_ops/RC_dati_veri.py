import numpy as np

frequenza = np.array([500, 1000, 2000, 5000, 10000]) # Hz (100000 sus)

V_A = np.array([10.1,
                9.36,
                9.44,
                np.mean ([7.52, 7.44]),
                np.mean ([9.92, 10.0])
                ])
V_B = np.array([6.92,
                6.80,
                8.40,
                np.mean ([6.88, 6.96]),
                np.mean ([9.92, 9.84])
                ])
V_AB = np.array([7.40,
                 4.40,
                 2.24,
                 np.mean ([1.12, 1.20]),
                 0.72
                 ])

# stavolta sono presi "nel centro" --> posizione in orizzontale degli "zeri"
phi_A = np.array([236,
                  60.0,
                  12.0,
                  0.8,
                  -0.8
                  ])
sigma_phi_A = np.array([])
phi_B = np.array([-24.0,
                  -20.0,
                  -9.0,
                  -3.6,
                  -1.6
                  ])
sigma_phi_B = np.array([])
phi_AB = np.array([484,
                   238.0,
                   114.0,
                   44.4,
                   np.mean ([24.0, 25.6])
                   ])
sigma_phi_AB = np.array([])

# sensibilità microsecondi: 4 microsecondi (zoom da 2.00 V)

# V_A è il segnale in entrata.
# V_B è la tensione sul condensatore: passa basso.
# V_AB è la tensione sulla resistenza: passa alto.
y_C = V_B / V_A
y_R = V_AB / V_A
y_err = np.full(len(frequenza), 0.03)

fase_A_B = (phi_B - phi_A) * 1e-6 * 2 * np.pi * frequenza
fase_A_AB = (phi_AB - phi_A) * 1e-6 * 2 * np.pi * frequenza