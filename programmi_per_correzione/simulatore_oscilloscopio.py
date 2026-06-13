"""
Simulatore di misure con oscilloscopio — RC, RL, RLC
======================================================
Genera dati di ampiezza (dB) e fase (gradi) a frequenze note,
includendo le distorsioni tipiche di un oscilloscopio reale
(es. Tektronix TDS2012C, 100 MHz, 2 GS/s).

Uso:
    python simulatore_oscilloscopio.py

Output:
    - Grafici dei diagrammi di Bode (ampiezza e fase)
    - CSV con i dati ideali e disturbati per ogni circuito
    - Stampa a schermo degli errori sistematici alle frequenze chiave
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ============================================================
# PARAMETRI DEI CIRCUITI
# ============================================================

# --- Circuito RC passa-basso ---
RC_PARAMS = {
    "R": 1000,      # Resistenza in Ohm
    "C": 10e-9,    # Capacità in Farad (10 nF)
}

# --- Circuito RL passa-basso ---
RL_PARAMS = {
    "R": 58,        # Resistenza in Ohm
    "L": 10e-3,     # Induttanza in Henry (10 mH)
}

# --- Circuito RLC (filtro passa-banda / risonanza) ---
RLC_PARAMS = {
    "R": 208,       # Resistenza in Ohm
    "L": 10e-3,     # Induttanza in Henry (10 mH)
    "C": 10e-9,     # Capacità in Farad (10 nF)
}

# ============================================================
# PARAMETRI DI DISTORSIONE DELL'OSCILLOSCOPIO
# Questi simulano i difetti visibili nella foto del TDS2012C
# ============================================================

DISTORTION = {
    # Overshoot sui fronti di salita/discesa (in percentuale, 0–100).
    # L'oscilloscopio "rimbalza" oltre il livello finale prima di stabilizzarsi.
    # Effetto: sovrastima dell'ampiezza picco-picco, specialmente a basse frequenze.
    # Valore tipico TDS2012C con sonda non compensata: 10–20%
    "overshoot_pct": 15.0,

    # Slew rate limitato — tempo di salita/discesa del fronte (in nanosecondi).
    # Il fronte reale non è verticale: impiega un certo tempo a passare da 0 a V_max.
    # Effetto: introduce uno sfasamento apparente, proporzionale alla frequenza.
    # Valore tipico per sonda 10x con cavo non compensato: 10–50 ns
    "slew_ns": 20.0,

    # Droop del plateau — inclinazione del livello alto/basso durante un semi-periodo.
    # Causato dall'accoppiamento AC o da una compensazione non corretta della sonda.
    # Unità: percentuale di caduta per millisecondo di durata del plateau.
    # Effetto: sottostima dell'ampiezza a basse frequenze (sotto ~1 kHz).
    "droop_pct_per_ms": 3.0,

    # Rumore di fondo del canale (in Volt rms).
    # Valore tipico TDS2012C a 2 V/div: 5–20 mV rms.
    # Effetto: incertezza casuale sulle misure, peggiora il rapporto segnale/rumore.
    "noise_vrms": 0.005,

    # Banda passante dell'oscilloscopio (in Hz).
    # Oltre questa frequenza il segnale viene attenuato come un filtro passa-basso
    # del primo ordine. TDS2012C: 100 MHz nominali.
    # Effetto: attenuazione aggiuntiva e sfasamento ad alte frequenze.
    "bandwidth_hz": 100e6,

    # Seme per il generatore di numeri casuali (per riproducibilità).
    # Cambia questo valore per ottenere una diversa realizzazione del rumore.
    "random_seed": 42,
}

# ============================================================
# FREQUENZE DI MISURA
# ============================================================

# Scegli le frequenze a cui vuoi simulare le misure.
# Puoi usare una delle due modalità:

# Modalità 1: frequenze logaritmicamente spaziate (utile per Bode completo)
FREQS_LOG = np.logspace(1, 6, 200)   # da 10 Hz a 1 MHz, 200 punti

# Modalità 2: frequenze specifiche (come fareste in laboratorio con il generatore)
FREQS_MANUAL = np.array([
    1e3, 2e3, 5e3,
    10e3, 20e3, 50e3, 70e3,
    100e3, 150e3, 200e3, 500e3
])


# Seleziona quale usare:
USE_LOG_FREQS = True   # True = logaritmiche, False = manuali


# ============================================================
# MODELLI TEORICI DEI CIRCUITI
# ============================================================

def transfer_RC(freqs, R, C):
    """
    Filtro RC passa-basso.
    H(f) = 1 / (1 + j*2π*f*R*C)
    f_-3dB = 1 / (2π*R*C)
    """
    w = 2 * np.pi * freqs
    tau = R * C
    H = 1 / (1 + 1j * w * tau)
    mag_db = 20 * np.log10(np.abs(H))
    phase_deg = np.angle(H, deg=True)
    f3db = 1 / (2 * np.pi * tau)
    return mag_db, phase_deg, {"f_3dB": f3db}


def transfer_RL(freqs, R, L):
    """
    Filtro RL passa-basso (tensione ai capi di R).
    H(f) = 1 / (1 + j*2π*f*L/R)
    f_-3dB = R / (2π*L)
    """
    w = 2 * np.pi * freqs
    tau = L / R
    H = 1 / (1 + 1j * w * tau)
    mag_db = 20 * np.log10(np.abs(H))
    phase_deg = np.angle(H, deg=True)
    f3db = R / (2 * np.pi * L)
    return mag_db, phase_deg, {"f_3dB": f3db}


def transfer_RLC(freqs, R, L, C):
    """
    Filtro RLC (tensione ai capi di R — banda-passante).
    H(f) = (j*2π*f*R*C) / (1 - (f/f0)^2 + j*2π*f*R*C)
    f0 = 1 / (2π*sqrt(L*C))   frequenza di risonanza
    Q  = (1/R) * sqrt(L/C)    fattore di qualità
    """
    w = 2 * np.pi * freqs
    w0 = 1 / np.sqrt(L * C)
    # Funzione di trasferimento passa-banda (V_R / V_in)
    H = (1j * w * R * C) / (1 - (w / w0)**2 + 1j * w * R * C)
    mag_db = 20 * np.log10(np.abs(H))
    phase_deg = np.angle(H, deg=True)
    f0 = w0 / (2 * np.pi)
    Q = (1 / R) * np.sqrt(L / C)
    return mag_db, phase_deg, {"f0": f0, "Q": Q}


# ============================================================
# MODELLO DELLE DISTORSIONI
# ============================================================

def apply_distortions(freqs, mag_db, phase_deg, dist, rng):
    """
    Applica le distorsioni dell'oscilloscopio ai dati ideali.

    Parametri:
        freqs     : array di frequenze (Hz)
        mag_db    : ampiezza ideale (dB)
        phase_deg : fase ideale (gradi)
        dist      : dizionario DISTORTION
        rng       : generatore random numpy

    Ritorna:
        mag_db_dist    : ampiezza con distorsioni (dB)
        phase_deg_dist : fase con distorsioni (gradi)
    """
    mag_lin = 10 ** (mag_db / 20)   # torna in lineare per sommare gli effetti

    # --- 1. Banda passante dell'oscilloscopio (filtro passa-basso 1° ordine) ---
    # Oltre la banda, il canale attenua il segnale come un RC aggiuntivo.
    bw = dist["bandwidth_hz"]
    bw_factor = 1 / np.sqrt(1 + (freqs / bw)**2)
    bw_phase_shift = -np.degrees(np.arctan(freqs / bw))   # gradi
    mag_lin = mag_lin * bw_factor
    phase_deg = phase_deg + bw_phase_shift

    # --- 2. Overshoot sui fronti ---
    # L'overshoot è massimo a basse frequenze (fronte occupa più periodo)
    # e si attenua ad alte frequenze dove il ringing si media su più cicli.
    ov = dist["overshoot_pct"] / 100
    # Funzione di peso: cala con la frequenza (heuristica basata su osservazione)
    ov_weight = np.exp(-freqs / 5000)
    mag_lin = mag_lin * (1 + ov * ov_weight)

    # --- 3. Droop del plateau ---
    # Il droop è un'inclinazione del livello alto durante il semi-periodo T/2.
    # L'effetto sull'ampiezza picco-picco cresce a basse frequenze
    # (plateau lungo → più tempo per "cadere").
    droop_rate = dist["droop_pct_per_ms"] / 100 / 1e-3   # in unità 1/s
    half_period = 1 / (2 * np.maximum(freqs, 1e-3))       # T/2 in secondi
    droop_loss = np.clip(droop_rate * half_period, 0, 0.5)
    mag_lin = mag_lin * (1 - droop_loss)

    # --- 4. Slew rate limitato → sfasamento apparente ---
    # Il fronte di salita impiega t_slew nanosecondi.
    # Questo introduce un ritardo apparente che si traduce in sfasamento
    # proporzionale alla frequenza: Δφ = -360° * f * t_slew
    t_slew = dist["slew_ns"] * 1e-9
    slew_phase_shift = -360 * freqs * t_slew    # gradi
    phase_deg = phase_deg + slew_phase_shift

    # --- 5. Rumore gaussiano ---
    # Il rumore dell'ADC e del canale analogico aggiunge incertezza casuale.
    # Modellato come rumore gaussiano sull'ampiezza (in Volt lineari).
    noise = rng.normal(0, dist["noise_vrms"], size=len(freqs))
    mag_lin = np.abs(mag_lin + noise / np.maximum(mag_lin, 1e-9) * mag_lin * 0.05)

    # Ritorno in dB
    mag_db_dist = 20 * np.log10(np.maximum(mag_lin, 1e-9))
    return mag_db_dist, phase_deg


# ============================================================
# FUNZIONE PRINCIPALE DI SIMULAZIONE
# ============================================================

def simulate_circuit(name, transfer_fn, freqs, dist):
    """
    Calcola risposta ideale + distorta e restituisce un DataFrame.
    """
    rng = np.random.default_rng(dist["random_seed"])

    mag_ideal, ph_ideal, info = transfer_fn(freqs)
    mag_dist, ph_dist = apply_distortions(freqs, mag_ideal, ph_ideal, dist, rng)

    df = pd.DataFrame({
        "Frequenza_Hz":     freqs,
        "Amp_Ideale_dB":    mag_ideal,
        "Amp_Distorta_dB":  mag_dist,
        "Fase_Ideale_deg":  ph_ideal,
        "Fase_Distorta_deg": ph_dist,
        "Errore_Amp_dB":    mag_dist - mag_ideal,
        "Errore_Fase_deg":  ph_dist - ph_ideal,
    })

    return df, info


# ============================================================
# GRAFICA
# ============================================================

def plot_bode(ax_amp, ax_ph, df, name, info, color_dist):
    """Traccia Bode ampiezza + fase su due assi già creati."""
    # Ampiezza
    ax_amp.semilogx(df["Frequenza_Hz"], df["Amp_Ideale_dB"],
                    color="#378ADD", lw=2, label="Ideale")
    ax_amp.semilogx(df["Frequenza_Hz"], df["Amp_Distorta_dB"],
                    color=color_dist, lw=1.5, ls="--", label="Con distorsioni")
    ax_amp.set_ylabel("|H(f)| (dB)")
    ax_amp.set_title(f"Circuito {name}", fontweight="bold")
    ax_amp.grid(True, which="both", alpha=0.3)
    ax_amp.legend(fontsize=9)

    # Linee di riferimento frequenza caratteristica
    if "f_3dB" in info:
        ax_amp.axvline(info["f_3dB"], color="gray", ls=":", alpha=0.6,
                       label=f"f₋₃dB = {info['f_3dB']:.1f} Hz")
        ax_amp.axhline(-3, color="gray", ls=":", alpha=0.4)
    if "f0" in info:
        ax_amp.axvline(info["f0"], color="gray", ls=":", alpha=0.6)

    # Fase
    ax_ph.semilogx(df["Frequenza_Hz"], df["Fase_Ideale_deg"],
                   color="#378ADD", lw=2)
    ax_ph.semilogx(df["Frequenza_Hz"], df["Fase_Distorta_deg"],
                   color=color_dist, lw=1.5, ls="--")
    ax_ph.set_ylabel("Fase (°)")
    ax_ph.set_xlabel("Frequenza (Hz)")
    ax_ph.grid(True, which="both", alpha=0.3)
    if "f_3dB" in info:
        ax_ph.axvline(info["f_3dB"], color="gray", ls=":", alpha=0.6)
    if "f0" in info:
        ax_ph.axvline(info["f0"], color="gray", ls=":", alpha=0.6)


def print_errors(name, df, info):
    """Stampa errori sistematici alle frequenze di interesse."""
    print(f"\n{'='*55}")
    print(f"  Circuito {name}")
    if "f_3dB" in info:
        print(f"  f₋₃dB = {info['f_3dB']:.2f} Hz")
    if "f0" in info:
        print(f"  f₀ = {info['f0']:.2f} Hz   Q = {info['Q']:.3f}")
    print(f"{'='*55}")
    print(f"  {'Frequenza':>12}  {'ΔAmp (dB)':>10}  {'ΔFase (°)':>10}")
    print(f"  {'-'*40}")

    # Frequenze di interesse
    test_freqs = [100, 1e3, 10e3, 100e3, 1e6]
    if "f_3dB" in info:
        test_freqs.append(info["f_3dB"])
    if "f0" in info:
        test_freqs.extend([info["f0"] / 2, info["f0"], info["f0"] * 2])

    for f_test in sorted(set(test_freqs)):
        idx = np.argmin(np.abs(df["Frequenza_Hz"] - f_test))
        row = df.iloc[idx]
        label = f"{row['Frequenza_Hz']:.0f} Hz"
        print(f"  {label:>12}  {row['Errore_Amp_dB']:>+10.2f}  {row['Errore_Fase_deg']:>+10.2f}")
    print()


# ============================================================
# ESECUZIONE
# ============================================================

if __name__ == "__main__":

    freqs = FREQS_LOG if USE_LOG_FREQS else FREQS_MANUAL

    # Costruisce le funzioni di trasferimento con i parametri scelti
    circuits = {
        "RC": lambda f: transfer_RC(f, **RC_PARAMS),
        "RL": lambda f: transfer_RL(f, **RL_PARAMS),
        "RLC": lambda f: transfer_RLC(f, **RLC_PARAMS),
    }
    colors = {"RC": "#D85A30", "RL": "#3B6D11", "RLC": "#993556"}

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    fig.suptitle("Diagrammi di Bode — Ideale vs Con distorsioni oscilloscopio",
                 fontsize=13, fontweight="bold")

    all_dfs = {}
    for i, (name, tfn) in enumerate(circuits.items()):
        df, info = simulate_circuit(name, tfn, freqs, DISTORTION)
        all_dfs[name] = df
        plot_bode(axes[i, 0], axes[i, 1], df, name, info, colors[name])
        print_errors(name, df, info)

    plt.tight_layout()
    out_plot = Path("bode_distorsioni.png")
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    print(f"Grafico salvato in: {out_plot.resolve()}")

    # Salva i CSV
    for name, df in all_dfs.items():
        out_csv = Path(f"dati_{name}.csv")
        df.to_csv(out_csv, index=False, float_format="%.6f", sep=";")
        print(f"CSV salvato in: {out_csv.resolve()}")

    plt.show()
