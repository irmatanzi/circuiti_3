import numpy as np


def H_R(f, R, L, C):
    omega = 2 * np.pi * np.asarray(f)
    return (omega * R * C) / np.sqrt(
        (1 - omega**2 * L * C)**2 + (omega * R * C)**2
    )


def H_L(f, R, L, C):
    omega = 2 * np.pi * np.asarray(f)
    return (omega**2 * L * C) / np.sqrt(
        (1 - omega**2 * L * C)**2 + (omega * R * C)**2
    )


def H_C(f, R, L, C):
    omega = 2 * np.pi * np.asarray(f)
    return 1 / np.sqrt(
        (1 - omega**2 * L * C)**2 + (omega * R * C)**2
    )


def _wrap_fase(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi


def fase_R(f, R, L, C):
    omega = 2 * np.pi * np.asarray(f)
    re = R
    im = omega * L - 1 / (omega * C)
    return -np.arctan2(im, re)


def fase_L(f, R, L, C):
    return _wrap_fase(fase_R(f, R, L, C) + np.pi / 2)


def fase_C(f, R, L, C):
    return _wrap_fase(fase_R(f, R, L, C) - np.pi / 2)


def rumore_clt(shape, sigma, n_contributi=12, rng=None):
    """Rumore quasi-gaussiano ottenuto sommando contributi uniformi."""
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


def genera_dati_rlc(frequenze, R, L, C, V_in=10.0, sigma_V=0.04,
                    sigma_t=2e-7, n_misure=3, n_contributi=12, seed=None,
                    scala_sonde=None):
    """Genera un dataset RLC serie sintetico, da dichiarare come simulato.

    Convenzione:
    - V_A: tensione in ingresso, ai capi dell'intero RLC
    - V_R, V_L, V_C: tensioni ai capi di resistenza, induttore, condensatore
    - phi_A, phi_R, phi_L, phi_C: posizioni temporali in microsecondi

    scala_sonde puo' simulare un settaggio errato delle sonde, per esempio:
    {"V_C": 0.1} produce una misura di V_C dieci volte piu' piccola.
    """
    rng = np.random.default_rng(seed)
    frequenze = np.asarray(frequenze, dtype=float)
    scala_sonde = {} if scala_sonde is None else scala_sonde

    V_A_vera = np.full(frequenze.shape, V_in)
    V_R_vera = V_A_vera * H_R(frequenze, R, L, C)
    V_L_vera = V_A_vera * H_L(frequenze, R, L, C)
    V_C_vera = V_A_vera * H_C(frequenze, R, L, C)

    V_A_vera = V_A_vera * scala_sonde.get("V_A", 1.0)
    V_R_vera = V_R_vera * scala_sonde.get("V_R", 1.0)
    V_L_vera = V_L_vera * scala_sonde.get("V_L", 1.0)
    V_C_vera = V_C_vera * scala_sonde.get("V_C", 1.0)

    V_A, sigma_V_A, letture_V_A = misure_con_errori(
        V_A_vera, sigma_V, n_misure, n_contributi, rng
    )
    V_R, sigma_V_R, letture_V_R = misure_con_errori(
        V_R_vera, sigma_V, n_misure, n_contributi, rng
    )
    V_L, sigma_V_L, letture_V_L = misure_con_errori(
        V_L_vera, sigma_V, n_misure, n_contributi, rng
    )
    V_C, sigma_V_C, letture_V_C = misure_con_errori(
        V_C_vera, sigma_V, n_misure, n_contributi, rng
    )

    phi_A_vera = np.zeros(frequenze.shape)
    phi_R_vera = fase_R(frequenze, R, L, C) / (2 * np.pi * frequenze)
    phi_L_vera = fase_L(frequenze, R, L, C) / (2 * np.pi * frequenze)
    phi_C_vera = fase_C(frequenze, R, L, C) / (2 * np.pi * frequenze)

    phi_A, sigma_phi_A, letture_phi_A = misure_con_errori(
        phi_A_vera, sigma_t, n_misure, n_contributi, rng
    )
    phi_R, sigma_phi_R, letture_phi_R = misure_con_errori(
        phi_R_vera, sigma_t, n_misure, n_contributi, rng
    )
    phi_L, sigma_phi_L, letture_phi_L = misure_con_errori(
        phi_L_vera, sigma_t, n_misure, n_contributi, rng
    )
    phi_C, sigma_phi_C, letture_phi_C = misure_con_errori(
        phi_C_vera, sigma_t, n_misure, n_contributi, rng
    )

    return {
        "frequenza": frequenze,
        "V_A": V_A,
        "V_R": V_R,
        "V_L": V_L,
        "V_C": V_C,
        "sigma_V_A": sigma_V_A,
        "sigma_V_R": sigma_V_R,
        "sigma_V_L": sigma_V_L,
        "sigma_V_C": sigma_V_C,
        "phi_A": phi_A * 1e6,
        "phi_R": phi_R * 1e6,
        "phi_L": phi_L * 1e6,
        "phi_C": phi_C * 1e6,
        "sigma_phi_A": sigma_phi_A * 1e6,
        "sigma_phi_R": sigma_phi_R * 1e6,
        "sigma_phi_L": sigma_phi_L * 1e6,
        "sigma_phi_C": sigma_phi_C * 1e6,
        "letture": {
            "V_A": letture_V_A,
            "V_R": letture_V_R,
            "V_L": letture_V_L,
            "V_C": letture_V_C,
            "phi_A": letture_phi_A * 1e6,
            "phi_R": letture_phi_R * 1e6,
            "phi_L": letture_phi_L * 1e6,
            "phi_C": letture_phi_C * 1e6,
        },
    }


def stampa_array(nome, valori, precisione=3):
    testo = np.array2string(np.asarray(valori), precision=precisione,
                            separator=", ")
    print(f"{nome} = np.array({testo})")
