import numpy as np

from RC_sintetico import genera_dati_rc, stampa_array


frequenza = np.array([200, 500, 1000, 2000, 5000, 7000, 10000, 20000, 30000, 40000, 50000, 60000])

dati = genera_dati_rc(
    frequenze=frequenza,
    R=3000,
    C=10e-9,
    V_in=10.0,
    sigma_V=0.1,
    sigma_t=2e-7,
    n_misure=3,
    seed=7,
)

stampa_array("frequenza", dati["frequenza"], precisione=0)
stampa_array("V_A", dati["V_A"], precisione=3)
stampa_array("V_B", dati["V_B"], precisione=3)
stampa_array("V_AB", dati["V_AB"], precisione=3)
stampa_array("sigma_V_A", dati["sigma_V_A"], precisione=3)
stampa_array("sigma_V_B", dati["sigma_V_B"], precisione=3)
stampa_array("sigma_V_AB", dati["sigma_V_AB"], precisione=3)
stampa_array("phi_A", dati["phi_A"], precisione=3)
stampa_array("phi_B", dati["phi_B"], precisione=3)
stampa_array("phi_AB", dati["phi_AB"], precisione=3)
stampa_array("sigma_phi_A", dati["sigma_phi_A"], precisione=3)
stampa_array("sigma_phi_B", dati["sigma_phi_B"], precisione=3)
stampa_array("sigma_phi_AB", dati["sigma_phi_AB"], precisione=3)
