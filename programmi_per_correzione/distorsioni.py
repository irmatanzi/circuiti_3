"""
Aggiunge distorsioni da oscilloscopio a dati sperimentali reali
================================================================
Input:  tensioni (V) e fasi (s) già "corrette" (prive di distorsioni)
Output: gli stessi dati con le distorsioni tipiche del TDS2012C aggiunte,
        inclusa propagazione degli errori sigma.

Le distorsioni modellate sono:
  1. Banda passante (filtro passa-basso 1° ordine, 100 MHz)
  2. Overshoot sui fronti  → sovrastima dell'ampiezza
  3. Droop del plateau     → sottostima dell'ampiezza a basse frequenze
  4. Slew rate limitato    → sfasamento apparente proporzionale alla frequenza
  5. Rumore gaussiano      → incertezza aggiuntiva su tutte le misure
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# PARAMETRI DI DISTORSIONE  ← modifica questi
# ============================================================

DIST = {
    # Overshoot sui fronti (%).
    # L'oscilloscopio "rimbalza" sul livello finale → sovrastima Vpp.
    # Effetto più forte a basse frequenze dove il fronte occupa più periodo.
    # Tipico TDS2012C sonda non compensata: 10–20 %
    "overshoot_pct": 15.0,

    # Slew rate: tempo di salita del fronte (ns).
    # Il fronte non è verticale → ritardo apparente → errore di fase.
    # Δφ ≈ −360° × f × t_slew  (si somma alla fase misurata in secondi)
    # Tipico sonda 10x: 10–50 ns
    "slew_ns": 20.0,

    # Droop del plateau (%/ms).
    # Il livello alto "scende" durante il semi-periodo (accoppiamento AC,
    # compensazione sonda non perfetta). Effetto max a basse frequenze.
    "droop_pct_per_ms": 3.0,

    # Rumore di fondo (V rms per canale).
    # Si somma in quadratura all'incertezza sigma già presente nei dati.
    # Tipico TDS2012C a 2 V/div: 5–20 mV rms
    "noise_vrms": 0.005,

    # Banda passante oscilloscopio (Hz).
    # TDS2012C nominale: 100 MHz. Abbassa per simulare sonda degradata.
    "bandwidth_hz": 100e6,

    # Seme casuale — fissa per riproducibilità, cambia per altra estrazione
    "seed": 42,
}

# ============================================================
# DATI SPERIMENTALI "CORRETTI"
# ============================================================

# ---------- RL ----------
RL = dict(
    frequenza = np.array([200., 500., 2000., 5000., 10000., 25000., 50000., 100000., 150000.]),

    V_in      = np.array([10.026, 10.018,  9.974,  9.963, 10.02 , 10.049,  9.949, 10.019, 10.019]),
    V_L       = np.array([ 2.104,  4.748,  9.069,  9.839,  9.951, 10.006, 10.004, 10.037,  9.963]),
    V_R       = np.array([ 9.743,  8.775,  4.201,  1.842,  0.909,  0.329,  0.164,  0.088,  0.042]),

    sigma_V_in = np.array([0.02 , 0.013, 0.028, 0.02 , 0.016, 0.024, 0.015, 0.014, 0.023]),
    sigma_V_L  = np.array([0.02 , 0.015, 0.021, 0.019, 0.019, 0.02 , 0.026, 0.028, 0.036]),
    sigma_V_R  = np.array([0.032, 0.016, 0.02 , 0.02 , 0.015, 0.034, 0.016, 0.019, 0.034]),

    # fasi in secondi (ritardo temporale misurato con i cursori)
    phi_in = np.array([ 0.143, -0.022, -0.132,  0.19 ,  0.14 , -0.092,  0.034, -0.052,  0.094]) * 1e-6,
    phi_L  = np.array([ 1.081e+03,  3.427e+02,  3.470e+01,  5.903e+00,  1.274e+00,
                        2.323e-01,  7.175e-02,  9.722e-02, -4.936e-02]) * 1e-6,
    phi_R  = np.array([-168.869, -157.443,  -90.362,  -44.067,  -23.429,
                        -9.843,   -4.865,   -2.374,   -1.729]) * 1e-6,

    sigma_phi_in = np.array([0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * 1e-6,
    sigma_phi_L  = np.array([0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * 1e-6,
    sigma_phi_R  = np.array([0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * 1e-6,
)

# ---------- RC ----------
RC = dict(
    frequenza = np.array([200., 500., 1000., 2000., 5000., 7000., 10000., 20000., 30000., 40000., 50000., 60000.]),

    V_in = np.array([ 9.964, 10.038,  9.984, 10.064, 10.027, 10.055,  9.867, 10.023, 10.072,  9.989, 10.052,  9.967]),
    V_C  = np.array([ 9.968,  9.995, 10.028,  9.899,  9.934,  9.895,  9.798,  9.041,  8.238,  7.404,  6.714,  5.854]),
    V_R  = np.array([ 0.119,  0.13 ,  0.188,  0.486,  1.181,  1.521,  2.187,  4.031,  5.527,  6.787,  7.497,  8.01 ]),

    sigma_V_in = np.array([0.065, 0.037, 0.052, 0.042, 0.031, 0.037, 0.034, 0.087, 0.028, 0.045, 0.032, 0.043]),
    sigma_V_C  = np.array([0.041, 0.056, 0.054, 0.046, 0.039, 0.038, 0.065, 0.103, 0.098, 0.038, 0.047, 0.06 ]),
    sigma_V_R  = np.array([0.056, 0.109, 0.042, 0.049, 0.089, 0.055, 0.07 , 0.064, 0.044, 0.07 , 0.023, 0.072]),

    phi_in = np.array([ 0.101,  0.012,  0.001, -0.214, -0.173,  0.084,  0.067,  0.036, -0.14 , -0.059, -0.146,  0.005]) * 1e-6,
    phi_C  = np.array([-3.477, -3.389, -3.471, -3.389, -3.469, -3.63 , -3.602, -3.479, -3.133, -2.889, -2.719, -2.509]) * 1e-6,
    phi_R  = np.array([1246.454, 496.209, 246.566, 121.528,  46.537,  32.262,  21.63 ,   9.06 ,   5.233,   3.485,   2.365,   1.778]) * 1e-6,

    # RC non ha sigma_phi → usiamo incertezza standard da cursore
    sigma_phi_in = np.full(12, 0.2e-6),
    sigma_phi_C  = np.full(12, 0.2e-6),
    sigma_phi_R  = np.full(12, 0.2e-6),
)

# ---------- RLC ----------
RLC = dict(
    frequenza = np.array([4000., 5000., 6000., 7000., 8000., 9000., 10000., 15000., 20000., 30000., 50000., 70000.]),

    V_in = np.array([ 9.986, 10.015,  9.993, 10.025, 10.011, 10.022,  9.947, 10.009, 10.029,  9.996, 10.021,  9.987]),
    V_R  = np.array([ 0.544,  0.721,  0.923,  1.091,  1.385,  1.711,  2.129,  8.66 ,  4.099,  1.513,  0.771,  0.467]),
    V_L  = np.array([ 0.703,  1.099,  1.634,  2.397,  3.371,  4.615,  6.368, 39.33 , 24.818, 13.787, 11.1  , 10.516]),
    V_C  = np.array([10.678, 11.068, 11.608, 12.277, 13.217, 14.503, 16.164, 44.325, 15.71 ,  3.86 ,  1.095,  0.545]),

    sigma_V_in = np.array([0.026, 0.027, 0.031, 0.017, 0.012, 0.025, 0.024, 0.035, 0.013, 0.018, 0.013, 0.017]),
    sigma_V_R  = np.array([0.028, 0.019, 0.018, 0.02 , 0.015, 0.003, 0.018, 0.041, 0.039, 0.021, 0.021, 0.024]),
    sigma_V_L  = np.array([0.022, 0.044, 0.017, 0.014, 0.036, 0.022, 0.028, 0.025, 0.018, 0.028, 0.019, 0.029]),
    sigma_V_C  = np.array([0.028, 0.026, 0.02 , 0.026, 0.03 , 0.029, 0.03 , 0.04 , 0.04 , 0.029, 0.037, 0.025]),

    phi_in = np.array([ 0.123,  0.21 ,  0.128,  0.208,  0.116, -0.06 , -0.062, -0.098,  0.03 ,  0.037, -0.023, -0.026]) * 1e-6,
    phi_R  = np.array([60.336, 47.505, 39.415, 33.272, 28.607, 24.868, 21.785,  5.437, -9.063, -7.363, -4.705, -3.364]) * 1e-6,
    phi_L  = np.array([122.578, 97.775, 80.88 , 68.951, 59.895, 52.329, 46.516, 22.21 ,  3.163,  1.038,  0.157,  0.135]) * 1e-6,
    phi_C  = np.array([ -2.091,  -2.206,  -2.527,  -2.439,  -2.834,  -3.102,  -3.454, -11.028, -21.504, -15.974,  -9.575,  -6.96 ]) * 1e-6,

    sigma_phi_in = np.full(12, 0.2e-6),
    sigma_phi_R  = np.full(12, 0.2e-6),
    sigma_phi_L  = np.full(12, 0.2e-6),
    sigma_phi_C  = np.full(12, 0.2e-6),
)

# ============================================================
# FUNZIONI DI DISTORSIONE
# ============================================================

def distort_amplitude(freqs, V, sigma_V, dist, rng):
    """
    Applica le distorsioni in ampiezza a un array di tensioni V misurate
    alle frequenze `freqs`.

    Modello fisico:
      V_mis = V_reale × (1 + Δ_overshoot) × (1 − Δ_droop) × BW_factor + rumore
    dove ogni Δ dipende dalla frequenza.

    Propagazione dell'errore: gli effetti sistematici scalano anche sigma.
    Il rumore strumentale si somma in quadratura a sigma_V originale.

    Ritorna: (V_dist, sigma_V_dist)
    """
    f = freqs

    # 1. Banda passante — attenua come RC passa-basso oltre bandwidth_hz
    bw = dist["bandwidth_hz"]
    bw_factor = 1 / np.sqrt(1 + (f / bw) ** 2)

    # 2. Overshoot — massimo a basse frequenze, si attenua salendo
    #    (a frequenze alte i rimbalzi si sovrappongono e si cancellano)
    ov = dist["overshoot_pct"] / 100.0
    ov_weight = np.exp(-f / 5000.0)
    ov_factor = 1 + ov * ov_weight

    # 3. Droop — l'ampiezza cade durante il plateau; effetto inversamente prop. a f
    droop_rate = dist["droop_pct_per_ms"] / 100.0 / 1e-3   # [1/s]
    half_period = 1.0 / (2.0 * np.maximum(f, 1.0))          # T/2 in secondi
    droop_loss = np.clip(droop_rate * half_period, 0.0, 0.5)
    droop_factor = 1.0 - droop_loss

    # Fattore complessivo sistematico
    sys_factor = bw_factor * ov_factor * droop_factor

    # Tensione distorta (sistematica)
    V_dist = V * sys_factor

    # Propagazione errore sistematica (moltiplicativa)
    sigma_sys = sigma_V * sys_factor

    # 4. Rumore strumentale — gaussiano, somma in quadratura
    noise_sample = rng.normal(0.0, dist["noise_vrms"], size=len(f))
    V_dist = V_dist + noise_sample
    sigma_dist = np.sqrt(sigma_sys**2 + dist["noise_vrms"]**2)

    return V_dist, sigma_dist


def distort_phase(freqs, phi, sigma_phi, dist):
    """
    Applica le distorsioni di fase a un array di sfasamenti in secondi.

    Modello:
      phi_mis = phi_reale + Δ_slew + Δ_bandwidth
    dove:
      Δ_slew = −t_slew            (ritardo fisso da fronte non verticale)
      Δ_bandwidth = −1/(2π·f·BW)  (ritardo di gruppo del passa-basso strumentale)

    Entrambi sono espressi in secondi (come phi).

    L'incertezza sigma_phi non cambia perché gli spostamenti sistematici
    non aggiungono incertezza — aggiungono solo bias.

    Ritorna: (phi_dist, sigma_phi)  ← sigma invariata
    """
    f = freqs

    # Slew rate: ritardo fisso (in secondi) — identico per tutte le frequenze
    t_slew = -dist["slew_ns"] * 1e-9   # negativo = ritardo

    # Banda passante: ritardo di gruppo del filtro strumentale
    #   τ_group = 1 / (2π · BW)  per filtro RC 1° ordine
    bw = dist["bandwidth_hz"]
    t_bw = -1.0 / (2.0 * np.pi * bw)  # costante (indip. da f per 1° ordine)

    phi_dist = phi + t_slew + t_bw

    # sigma_phi rimane invariata: il bias sistematico non aumenta l'incertezza
    return phi_dist, sigma_phi


# ============================================================
# APPLICAZIONE AI TRE CIRCUITI
# ============================================================

def apply_to_dataset(data, amp_keys, phi_keys, dist):
    """
    Prende un dizionario di dati e restituisce un nuovo dizionario
    con le stesse chiavi ma i valori disturbati.

    amp_keys : lista di nomi delle tensioni, es. ['V_in', 'V_R', 'V_L']
               Per ognuna ci deve essere 'sigma_<nome>' nel dizionario.
    phi_keys : lista di nomi delle fasi,   es. ['phi_in', 'phi_R', 'phi_L']
               Per ognuna ci deve essere 'sigma_<nome>' nel dizionario.
    """
    rng = np.random.default_rng(dist["seed"])
    freqs = data["frequenza"]
    out = {"frequenza": freqs.copy()}

    for key in amp_keys:
        V_d, s_d = distort_amplitude(freqs, data[key], data[f"sigma_{key}"], dist, rng)
        out[key] = V_d
        out[f"sigma_{key}"] = s_d

    for key in phi_keys:
        p_d, s_d = distort_phase(freqs, data[key], data[f"sigma_{key}"], dist)
        out[key] = p_d
        out[f"sigma_{key}"] = s_d

    return out


# Applica a tutti e tre
RL_dist  = apply_to_dataset(RL,  ["V_in","V_L","V_R"],       ["phi_in","phi_L","phi_R"],       DIST)
RC_dist  = apply_to_dataset(RC,  ["V_in","V_C","V_R"],       ["phi_in","phi_C","phi_R"],       DIST)
RLC_dist = apply_to_dataset(RLC, ["V_in","V_R","V_L","V_C"], ["phi_in","phi_R","phi_L","phi_C"], DIST)


# ============================================================
# STAMPA CONFRONTO
# ============================================================

def print_comparison(name, data_orig, data_dist, amp_keys, phi_keys):
    f = data_orig["frequenza"]
    print(f"\n{'═'*68}")
    print(f"  {name}  —  confronto originale vs distorto")
    print(f"{'═'*68}")
    for key in amp_keys:
        print(f"\n  {key}  (V)")
        print(f"  {'f (Hz)':>10}  {'Orig':>8}  {'Dist':>8}  {'ΔV':>8}  {'σ_orig':>7}  {'σ_dist':>7}")
        print(f"  {'-'*56}")
        for i, fi in enumerate(f):
            dv = data_dist[key][i] - data_orig[key][i]
            print(f"  {fi:>10.0f}  {data_orig[key][i]:>8.4f}  {data_dist[key][i]:>8.4f}"
                  f"  {dv:>+8.4f}  {data_orig['sigma_'+key][i]:>7.4f}  {data_dist['sigma_'+key][i]:>7.4f}")
    for key in phi_keys:
        print(f"\n  {key}  (μs)")
        print(f"  {'f (Hz)':>10}  {'Orig':>10}  {'Dist':>10}  {'Δφ':>10}")
        print(f"  {'-'*46}")
        for i, fi in enumerate(f):
            dp = (data_dist[key][i] - data_orig[key][i]) * 1e6
            print(f"  {fi:>10.0f}  {data_orig[key][i]*1e6:>10.4f}  {data_dist[key][i]*1e6:>10.4f}  {dp:>+10.4f}")

print_comparison("RL",  RL,  RL_dist,  ["V_in","V_L","V_R"], ["phi_in","phi_L","phi_R"])
print_comparison("RC",  RC,  RC_dist,  ["V_in","V_C","V_R"], ["phi_in","phi_C","phi_R"])
print_comparison("RLC", RLC, RLC_dist, ["V_in","V_R","V_L","V_C"], ["phi_in","phi_R","phi_L","phi_C"])


# ============================================================
# GRAFICI
# ============================================================

def make_plot(name, data_orig, data_dist, amp_keys, phi_keys):
    n_amp = len(amp_keys)
    n_phi = len(phi_keys)
    ncols = max(n_amp, n_phi)
    fig, axes = plt.subplots(2, ncols, figsize=(4.5*ncols, 8))
    fig.suptitle(f"Circuito {name} — originale vs con distorsioni", fontweight="bold")

    f = data_orig["frequenza"]

    for col, key in enumerate(amp_keys):
        ax = axes[0, col]
        ax.errorbar(f, data_orig[key], yerr=data_orig[f"sigma_{key}"],
                    fmt="o-", color="#378ADD", lw=1.5, ms=5, capsize=3, label="Originale")
        ax.errorbar(f, data_dist[key], yerr=data_dist[f"sigma_{key}"],
                    fmt="s--", color="#D85A30", lw=1.5, ms=5, capsize=3, label="Con distorsioni")
        ax.set_xscale("log")
        ax.set_xlabel("Frequenza (Hz)")
        ax.set_ylabel("Tensione (V)")
        ax.set_title(key)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    # nascondi eventuali assi vuoti in riga 0
    for col in range(len(amp_keys), ncols):
        axes[0, col].set_visible(False)

    for col, key in enumerate(phi_keys):
        ax = axes[1, col]
        ax.errorbar(f, data_orig[key]*1e6, yerr=data_orig[f"sigma_{key}"]*1e6,
                    fmt="o-", color="#378ADD", lw=1.5, ms=5, capsize=3, label="Originale")
        ax.errorbar(f, data_dist[key]*1e6, yerr=data_dist[f"sigma_{key}"]*1e6,
                    fmt="s--", color="#D85A30", lw=1.5, ms=5, capsize=3, label="Con distorsioni")
        ax.set_xscale("log")
        ax.set_xlabel("Frequenza (Hz)")
        ax.set_ylabel("Sfasamento (μs)")
        ax.set_title(key)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    for col in range(len(phi_keys), ncols):
        axes[1, col].set_visible(False)

    plt.tight_layout()
    out = f"distorsioni_{name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nGrafico salvato: {out}")
    return fig


make_plot("RL",  RL,  RL_dist,  ["V_in","V_L","V_R"], ["phi_in","phi_L","phi_R"])
make_plot("RC",  RC,  RC_dist,  ["V_in","V_C","V_R"], ["phi_in","phi_C","phi_R"])
make_plot("RLC", RLC, RLC_dist, ["V_in","V_R","V_L","V_C"], ["phi_in","phi_R","phi_L","phi_C"])

plt.show()

# ============================================================
# ACCESSO PROGRAMMATICO AI RISULTATI
# ============================================================
# Dopo aver eseguito questo file puoi usare direttamente:
#
#   RL_dist["V_R"]          → tensione V_R distorta (array)
#   RL_dist["sigma_V_R"]    → sua incertezza propagata
#   RL_dist["phi_R"]        → fase phi_R distorta (secondi)
#   RL_dist["sigma_phi_R"]  → sua incertezza (invariata, bias puro)
#
# Stessa struttura per RC_dist e RLC_dist.


# ============================================================
# STAMPA ARRAY GENERATI "PRONTI ALL'USO"
# ============================================================

def print_ready_arrays(name, data_dist, amp_keys, phi_keys):
    print(f"\n{'='*70}")
    print(f"  ARRAY PRONTI PER IL CIRCUITO: {name}")
    print(f"{'='*70}")
    
    # Stampa le frequenze
    freq_str = ", ".join(f"{x:.1f}" for x in data_dist["frequenza"])
    print(f'\n{name}_frequenza = np.array([{freq_str}])')
    
    # Stampa ampiezze e relative incertezze
    for key in amp_keys:
        v_str = ", ".join(f"{x:.4f}" for x in data_dist[key])
        s_str = ", ".join(f"{x:.4f}" for x in data_dist[f"sigma_{key}"])
        print(f'{name}_{key} = np.array([{v_str}])')
        print(f'{name}_sigma_{key} = np.array([{s_str}])')
        
    # Stampa le fasi e relative incertezze
    for key in phi_keys:
        p_str = ", ".join(f"{x:.4e}" for x in data_dist[key])
        s_str = ", ".join(f"{x:.4e}" for x in data_dist[f"sigma_{key}"])
        print(f'{name}_{key} = np.array([{p_str}])')
        print(f'{name}_sigma_{key} = np.array([{s_str}])')

# Esegui la stampa per tutti e tre i circuiti
print_ready_arrays("RL",  RL_dist,  ["V_in","V_L","V_R"], ["phi_in","phi_L","phi_R"])
print_ready_arrays("RC",  RC_dist,  ["V_in","V_C","V_R"], ["phi_in","phi_C","phi_R"])
print_ready_arrays("RLC", RLC_dist, ["V_in","V_R","V_L","V_C"], ["phi_in","phi_R","phi_L","phi_C"])