import numpy as np


import numpy as np


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


def rumore_clt(shape, sigma, n_contributi=12, rng=None):
    """Rumore quasi-gaussiano ottenuto sommando contributi uniformi.

    La varianza finale e' circa sigma**2. Aumentare n_contributi rende la
    distribuzione piu' simile a una gaussiana, per il teorema centrale.
    """
    rng = np.random.default_rng(rng)
    contributi = rng.uniform(-0.5, 0.5, size=(n_contributi, *np.atleast_1d(shape)))
    z = np.sum(contributi, axis=0)
    z = z / np.sqrt(n_contributi / 12)
    return sigma * z.reshape(shape)


def misure_con_errori(valori_veri, sigma_strumento, n_misure=3,
                      n_contributi=12, rng=None):
    """Simula misure ripetute e restituisce media, errore sulla media e dati grezzi."""
    rng = np.random.default_rng(rng)
    valori_veri = np.asarray(valori_veri)
    letture = []

    for _ in range(n_misure):
        rumore = rumore_clt(valori_veri.shape, sigma_strumento,
                            n_contributi=n_contributi, rng=rng)
        letture.append(valori_veri + rumore)

    letture = np.array(letture)
    media = np.mean(letture, axis=0)

    if n_misure > 1:
        errore_media = np.std(letture, axis=0, ddof=1) / np.sqrt(n_misure)
    else:
        errore_media = np.full(valori_veri.shape, sigma_strumento)

    return media, errore_media, letture


def genera_dati_rl(frequenze, R, L, V_in=10.0, sigma_V=0.08,
                   sigma_t=4e-6, n_misure=3, n_contributi=12, seed=None):
    """Genera un dataset RL sintetico, da dichiarare come simulato.

    Convenzione:
    - V_A: tensione in ingresso
    - V_B: tensione sul induttore
    - V_AB: tensione sulla resistenza
    - phi_A, phi_B, phi_AB: posizioni temporali in microsecondi
    """
    rng = np.random.default_rng(seed)
    frequenze = np.asarray(frequenze, dtype=float)

    V_A_vera = np.full(frequenze.shape, V_in)
    V_B_vera = V_A_vera * H_L(frequenze, R, L)
    V_AB_vera = V_A_vera * H_R(frequenze, R, L)

    V_A, sigma_V_A, letture_V_A = misure_con_errori(
        V_A_vera, sigma_V, n_misure, n_contributi, rng
    )
    V_B, sigma_V_B, letture_V_B = misure_con_errori(
        V_B_vera, sigma_V, n_misure, n_contributi, rng
    )
    V_AB, sigma_V_AB, letture_V_AB = misure_con_errori(
        V_AB_vera, sigma_V, n_misure, n_contributi, rng
    )

    # Scelgo phi_A = 0 e trasformo le fasi teoriche in ritardi temporali.
    phi_A_vera = np.zeros(frequenze.shape)
    phi_B_vera = fase_L(frequenze, R, L) / (2 * np.pi * frequenze)
    phi_AB_vera = fase_R(frequenze, R, L) / (2 * np.pi * frequenze)

    phi_A, sigma_phi_A, letture_phi_A = misure_con_errori(
        phi_A_vera, sigma_t, n_misure, n_contributi, rng
    )
    phi_B, sigma_phi_B, letture_phi_B = misure_con_errori(
        phi_B_vera, sigma_t, n_misure, n_contributi, rng
    )
    phi_AB, sigma_phi_AB, letture_phi_AB = misure_con_errori(
        phi_AB_vera, sigma_t, n_misure, n_contributi, rng
    )

    return {
        "frequenza": frequenze,
        "V_A": V_A,
        "V_B": V_B,
        "V_AB": V_AB,
        "sigma_V_A": sigma_V_A,
        "sigma_V_B": sigma_V_B,
        "sigma_V_AB": sigma_V_AB,
        "phi_A": phi_A * 1e6,
        "phi_B": phi_B * 1e6,
        "phi_AB": phi_AB * 1e6,
        "sigma_phi_A": sigma_phi_A * 1e6,
        "sigma_phi_B": sigma_phi_B * 1e6,
        "sigma_phi_AB": sigma_phi_AB * 1e6,
        "letture": {
            "V_A": letture_V_A,
            "V_B": letture_V_B,
            "V_AB": letture_V_AB,
            "phi_A": letture_phi_A * 1e6,
            "phi_B": letture_phi_B * 1e6,
            "phi_AB": letture_phi_AB * 1e6,
        },
    }


def stampa_array(nome, valori, precisione=3):
    testo = np.array2string(np.asarray(valori), precision=precisione,
                            separator=", ")
    print(f"{nome} = np.array({testo})")
