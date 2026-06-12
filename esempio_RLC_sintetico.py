import numpy as np

from RLC_sintetico import genera_dati_rlc, stampa_array


frequenza = np.array([
    4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 30000, 50000, 70000
])

dati = genera_dati_rlc(
    frequenze=frequenza,
    R=208,
    L=10e-3,
    C=10e-9,
    V_in=10.0,
    sigma_V=0.04,
    sigma_t=2e-7,
    n_misure=3,
    seed=7
    # Esempio per simulare una sonda 10x letta come 1x:
    # scala_sonde={"V_C": 0.1},
)

stampa_array("frequenze", dati["frequenza"], precisione=0)
stampa_array("V_A", dati["V_A"], precisione=3)
stampa_array("V_R", dati["V_R"], precisione=3)
stampa_array("V_L", dati["V_L"], precisione=3)
stampa_array("V_C", dati["V_C"], precisione=3)
stampa_array("sigma_V_A", dati["sigma_V_A"], precisione=3)
stampa_array("sigma_V_R", dati["sigma_V_R"], precisione=3)
stampa_array("sigma_V_L", dati["sigma_V_L"], precisione=3)
stampa_array("sigma_V_C", dati["sigma_V_C"], precisione=3)
stampa_array("phi_A", dati["phi_A"], precisione=3)
stampa_array("phi_R", dati["phi_R"], precisione=3)
stampa_array("phi_L", dati["phi_L"], precisione=3)
stampa_array("phi_C", dati["phi_C"], precisione=3)
stampa_array("sigma_phi_A", dati["sigma_phi_A"], precisione=3)
stampa_array("sigma_phi_R", dati["sigma_phi_R"], precisione=3)
stampa_array("sigma_phi_L", dati["sigma_phi_L"], precisione=3)
stampa_array("sigma_phi_C", dati["sigma_phi_C"], precisione=3)
