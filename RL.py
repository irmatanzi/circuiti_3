import numpy as np
import matplotlib.pyplot as plt

# Dati:
R_L = 8.3 # Ohm
R = 50 # Ohm

frequenza = np.array([200, 500, 2000, 5000, 10000,  25000, 50000, 100000, 150000]) # Hz (200 è sus)
V_A = np.array([5.44, 5.60, 6.88, 8.96, 9.76, 10.05, 10.05, 10.05, 10.0]) # V DA DIVIDERE A META
V_B = np.array([4.80, 4.55, 4.00, 2.64, 1.52, 0.648, 0.328, 0.168, 0.108]) # V
V_AB = np.array([1.16, 1.76, 5.20, 8.40, 9.56, 10.0, 10.05, 10.05, 10.1]) # V (1.16 è [1.12-1.20]) (9.56 tra 52 e 60)

# Posizioni dei picchi nel tempo (media di 2 valori perchè sul picco rimane uguale)

phi_A = np.array([np.mean ([920, 1080]), np.mean ([320, 360]), np.mean ([532, 548]), 
                  np.mean ([195, 204]), np.mean ([98, 102]), np.mean ([39.8, 41.1]), 
                  np.mean ([19.6, 20.2]), np.mean ([9.60, 10.0]), np.mean ([7.20, 7.32])]) # micro secondi

sigma_phi_A = np.array([np.std ([920, 1080])/np.sqrt(2), np.std ([320, 360])/np.sqrt(2), np.std ([532, 548])/np.sqrt(2), 
                        np.std ([195, 204])/np.sqrt(2), np.std ([98, 102])/np.sqrt(2), np.std ([39.8, 41.1])/np.sqrt(2), 
                        np.std ([19.6, 20.2])/np.sqrt(2), np.std ([9.60, 10.0])/np.sqrt(2), np.std ([7.20, 7.32])/np.sqrt(2)]) # microsecondi


phi_B = np.array([np.mean ([920, 1200]), np.mean ([390, 470]), np.mean ([588, 620]), 
                  np.mean ([230, 246]), np.mean ([121, 123]), np.mean ([48.4, 51]), 
                  np.mean ([24.2, 24.8]), np.mean ([12.8, 13.3]), np.mean ([8.68, 9.12])]) # micro secondi (circa -0.25)

sigma_phi_B = np.array([np.std ([920, 1200])/np.sqrt(2), np.std ([390, 470])/np.sqrt(2), np.std ([588, 620])/np.sqrt(2), 
                        np.std ([230, 246])/np.sqrt(2), np.std ([121, 123])/np.sqrt(2), np.std ([48.4, 51])/np.sqrt(2), 
                        np.std ([24.2, 24.8])/np.sqrt(2), np.std ([12.8, 13.3])/np.sqrt(2), np.std ([8.68, 9.12])/np.sqrt(2)]) # micro secondi


phi_AB = np.array([np.mean ([420, 500]), np.mean ([0, 190]), np.mean ([472, 500]),
                   np.mean ([192, 200]), np.mean ([97, 99]),np.mean ([39.6]), 
                   np.mean ([19.6]), np.mean ([10.4, 10.5]), np.mean ([7.16, 7.20])]) # micro secondi (circa -0.25)

sigma_phi_AB = np.array([np.std ([420, 500])/np.sqrt(2), np.std ([0, 190])/np.sqrt(2), np.std ([472, 500])/np.sqrt(2), 
                         np.std ([192, 200])/np.sqrt(2), np.std ([97, 99])/np.sqrt(2), np.std ([39.6])/np.sqrt(2), 
                         np.std ([19.6])/np.sqrt(2), np.std ([10.4, 10.5])/np.sqrt(2), np.std ([7.16, 7.20])/np.sqrt(2)]) # micro secondi


# differenze tra fasi (lo 0 sarà A)
fase_A_B = phi_B - phi_A # micro secondi
fase_A_AB = phi_AB - phi_A # micro secondi




# sensibilità microsecondi: 0.2 microsecondi (zoom da 2.00 V)

delta_phi = np.array([]) # s 

# vaffanculo
# probabilmente RISCALARE (probabile)

fig, ax = plt.subplots()

ax.scatter (frequenza, phi_A, marker = 'o', color = "hotpink", label='phi_A')
ax.scatter (frequenza, phi_B,  marker = 'o', color = "lightblue", label='phi_B')
ax.scatter (frequenza, phi_AB, marker = 'o', color = "lightgreen", label='phi_AB')

ax.scatter (frequenza, fase_A_B, marker = 'o', color = "orange", label='fase_A_B')
ax.scatter (frequenza, fase_A_AB, marker = 'o', color = "cyan", label='fase_A_AB')
ax.plot (frequenza, fase_A_B, color = "orange", linestyle = "--")
ax.plot (frequenza, fase_A_AB, color = "cyan", linestyle = "--")

plt.show ()

# FILTRO PASSA BASSO