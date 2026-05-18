# app.py
# ==============================================================================
# APPLICATION STREAMLIT — BILINÉARISATION PUSHOVER
# Eurocode 8 + ASCE 41
# ==============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import trapezoid
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

# ==============================================================================
# CONFIGURATION STREAMLIT
# ==============================================================================

st.set_page_config(
    page_title="Bilinéarisation Pushover",
    layout="wide"
)

st.title("Bilinéarisation d'une Courbe de Capacité Pushover")
st.markdown("""
Application de :
- Bilinéarisation selon Eurocode 8 (Méthode N2)
- Bilinéarisation selon ASCE 41
- Conservation énergétique
""")

# ==============================================================================
# FONCTIONS SCIENTIFIQUES
# ==============================================================================

def lissage_donnees(d, f, window_length=15, polyorder=3):
    """
    Lissage Savitzky-Golay.
    """

    # Sécurisation de la taille de fenêtre
    if window_length >= len(f):
        window_length = len(f) - 1

    if window_length % 2 == 0:
        window_length -= 1

    if window_length < 5:
        window_length = 5

    f_lisse = savgol_filter(
        f,
        window_length=window_length,
        polyorder=polyorder
    )

    f_lisse[0] = 0.0

    return f_lisse


def bilinearisation_eurocode8(d, f_lisse):
    """
    Bilinéarisation Eurocode 8.
    """

    idx_m = np.argmax(f_lisse)

    dm = d[idx_m]
    Vm = f_lisse[idx_m]

    Em = trapezoid(f_lisse[:idx_m + 1], d[:idx_m + 1])

    Vy = Vm

    dy = 2 * (dm - (Em / Vy))

    Ke = Vy / dy

    d_bilin = np.array([0, dy, dm])
    f_bilin = np.array([0, Vy, Vy])

    return {
        "Ke": Ke,
        "Vy": Vy,
        "dy": dy,
        "Em": Em,
        "d_bilin": d_bilin,
        "f_bilin": f_bilin,
        "idx_m": idx_m,
        "dm": dm,
        "Vm": Vm
    }


def bilinearisation_asce41(d, f_lisse):
    """
    Bilinéarisation ASCE 41.
    """

    idx_m = np.argmax(f_lisse)

    dm = d[idx_m]
    Vm = f_lisse[idx_m]

    Em = trapezoid(f_lisse[:idx_m + 1], d[:idx_m + 1])

    d_asc = d[:idx_m + 1]

    f_asc = (
        np.maximum.accumulate(f_lisse[:idx_m + 1])
        + np.linspace(0, 1e-5, idx_m + 1)
    )

    inv_interp = interp1d(
        f_asc,
        d_asc,
        kind='linear',
        fill_value="extrapolate"
    )

    def erreur_energie(Vy_test):

        V_06 = 0.6 * Vy_test

        d_06 = float(inv_interp(V_06))

        Ke = V_06 / d_06

        dy = Vy_test / Ke

        area_bilin = (
            0.5 * Vy_test * dy
            + 0.5 * (Vy_test + Vm) * (dm - dy)
        )

        return area_bilin - Em

    res = root_scalar(
        erreur_energie,
        bracket=[0.5 * Vm, 1.5 * Vm],
        method='brentq'
    )

    Vy = res.root

    V_06 = 0.6 * Vy

    d_06 = float(inv_interp(V_06))

    Ke = V_06 / d_06

    dy = Vy / Ke

    Kp = (Vm - Vy) / (dm - dy)

    Alpha = Kp / Ke

    d_bilin = np.array([0, dy, dm])

    f_bilin = np.array([0, Vy, Vm])

    Em_asce = trapezoid(f_bilin, d_bilin)

    return {
        "Ke": Ke,
        "Vy": Vy,
        "dy": dy,
        "Alpha": Alpha,
        "Em": Em,
        "Em_asce": Em_asce,
        "d_bilin": d_bilin,
        "f_bilin": f_bilin,
        "idx_m": idx_m,
        "dm": dm,
        "Vm": Vm,
        "d06": d_06,
        "v06": V_06
    }


# ==============================================================================
# SIDEBAR
# ==============================================================================

st.sidebar.header("Paramètres")

window_length = st.sidebar.slider(
    "Fenêtre de lissage",
    min_value=5,
    max_value=51,
    value=15,
    step=2
)

polyorder = st.sidebar.slider(
    "Ordre du polynôme",
    min_value=2,
    max_value=5,
    value=3
)

# ==============================================================================
# IMPORT FICHIER
# ==============================================================================

uploaded_file = st.file_uploader(
    "Charger un fichier Excel ou CSV",
    type=["xlsx", "xls", "csv"]
)

# ==============================================================================
# LECTURE DONNÉES
# ==============================================================================

if uploaded_file is not None:

    try:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Aperçu des données")
        st.dataframe(df.head())

        # Détection automatique colonnes
        col_d = [c for c in df.columns if c.lower().startswith("d")][0]
        col_f = [c for c in df.columns if c.lower().startswith("f")][0]

        d_brut = df[col_d].values
        f_brut = df[col_f].values

        # Nettoyage
        mask = np.isfinite(d_brut) & np.isfinite(f_brut)

        d_brut = d_brut[mask]
        f_brut = f_brut[mask]

        # Tri
        sort_idx = np.argsort(d_brut)

        d_brut = d_brut[sort_idx]
        f_brut = f_brut[sort_idx]

        # ======================================================================
        # CALCULS
        # ======================================================================

        f_lisse = lissage_donnees(
            d_brut,
            f_brut,
            window_length=window_length,
            polyorder=polyorder
        )

        result_ec8 = bilinearisation_eurocode8(
            d_brut,
            f_lisse
        )

        result_asce = bilinearisation_asce41(
            d_brut,
            f_lisse
        )

        Em_ec8 = trapezoid(
            result_ec8["f_bilin"],
            result_ec8["d_bilin"]
        )

        idx_m = result_ec8["idx_m"]

        # ======================================================================
        # AFFICHAGE MÉTRIQUES
        # ======================================================================

        st.subheader("Résultats")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Eurocode 8")

            st.metric(
                "Rigidité initiale Ke",
                f"{result_ec8['Ke']:.2f}"
            )

            st.metric(
                "Force de plastification Vy",
                f"{result_ec8['Vy']:.2f}"
            )

            st.metric(
                "Déplacement dy",
                f"{result_ec8['dy']:.2f}"
            )

        with col2:

            st.markdown("### ASCE 41")

            st.metric(
                "Rigidité initiale Ke",
                f"{result_asce['Ke']:.2f}"
            )

            st.metric(
                "Force de plastification Vy",
                f"{result_asce['Vy']:.2f}"
            )

            st.metric(
                "Déplacement dy",
                f"{result_asce['dy']:.2f}"
            )

            st.metric(
                "Coefficient α",
                f"{result_asce['Alpha']:.4f}"
            )

        # ======================================================================
        # FIGURES
        # ======================================================================

        plt.style.use('seaborn-v0_8-whitegrid')

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))

        # ----------------------------------------------------------------------
        # FIGURE 1 — EC8
        # ----------------------------------------------------------------------

        axs[0, 0].plot(
            d_brut[:idx_m + 1],
            f_lisse[:idx_m + 1],
            'k-',
            lw=2,
            label="Courbe lissée"
        )

        axs[0, 0].plot(
            result_ec8["d_bilin"],
            result_ec8["f_bilin"],
            'b--',
            lw=3,
            label="Bilinéaire EC8"
        )

        axs[0, 0].fill_between(
            d_brut[:idx_m + 1],
            0,
            f_lisse[:idx_m + 1],
            alpha=0.2
        )

        axs[0, 0].set_title("Eurocode 8")
        axs[0, 0].set_xlabel("Déplacement")
        axs[0, 0].set_ylabel("Effort")

        axs[0, 0].legend()

        # ----------------------------------------------------------------------
        # FIGURE 2 — ASCE 41
        # ----------------------------------------------------------------------

        axs[0, 1].plot(
            d_brut[:idx_m + 1],
            f_lisse[:idx_m + 1],
            'k-',
            lw=2,
            label="Courbe lissée"
        )

        axs[0, 1].plot(
            result_asce["d_bilin"],
            result_asce["f_bilin"],
            'r--',
            lw=3,
            label="Bilinéaire ASCE 41"
        )

        axs[0, 1].scatter(
            [result_asce["d06"]],
            [result_asce["v06"]],
            color='red',
            s=100,
            label="Point 60% Vy"
        )

        axs[0, 1].set_title("ASCE 41")
        axs[0, 1].set_xlabel("Déplacement")
        axs[0, 1].set_ylabel("Effort")

        axs[0, 1].legend()

        # ----------------------------------------------------------------------
        # FIGURE 3 — ÉNERGIE
        # ----------------------------------------------------------------------

        bars = axs[1, 0].bar(
            ['Réelle', 'EC8', 'ASCE 41'],
            [
                result_ec8["Em"],
                Em_ec8,
                result_asce["Em_asce"]
            ]
        )

        axs[1, 0].set_title("Validation énergétique")

        for bar in bars:

            yval = bar.get_height()

            axs[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f"{yval:.2f}",
                ha='center',
                va='bottom'
            )

        # ----------------------------------------------------------------------
        # FIGURE 4 — RIGIDITÉ INITIALE
        # ----------------------------------------------------------------------

        zoom = int(idx_m * 0.3)

        axs[1, 1].plot(
            d_brut[:zoom],
            f_lisse[:zoom],
            'k-',
            lw=2,
            label="Réel"
        )

        axs[1, 1].plot(
            result_ec8["d_bilin"][:2],
            result_ec8["f_bilin"][:2],
            'b--',
            lw=2,
            label=f"Ke EC8 = {result_ec8['Ke']:.1f}"
        )

        axs[1, 1].plot(
            [0, result_asce["dy"]],
            [0, result_asce["Vy"]],
            'r--',
            lw=2,
            label=f"Ke ASCE = {result_asce['Ke']:.1f}"
        )

        axs[1, 1].set_title("Comparaison des rigidités")

        axs[1, 1].set_xlabel("Déplacement")
        axs[1, 1].set_ylabel("Effort")

        axs[1, 1].legend()

        plt.tight_layout()

        st.pyplot(fig)

        # ======================================================================
        # EXPORT CSV
        # ======================================================================

        resultats = pd.DataFrame({
            "Paramètre": [
                "Ke_EC8",
                "Vy_EC8",
                "dy_EC8",
                "Ke_ASCE41",
                "Vy_ASCE41",
                "dy_ASCE41",
                "Alpha_ASCE41"
            ],
            "Valeur": [
                result_ec8["Ke"],
                result_ec8["Vy"],
                result_ec8["dy"],
                result_asce["Ke"],
                result_asce["Vy"],
                result_asce["dy"],
                result_asce["Alpha"]
            ]
        })

        csv = resultats.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Télécharger les résultats CSV",
            data=csv,
            file_name="resultats_bilinearisation.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(f"Erreur : {e}")

else:

    st.info("Chargez un fichier Excel ou CSV contenant les colonnes déplacement et effort.")

    # Démonstration
    d_demo = np.linspace(0, 100, 300)

    f_demo = (
        2000 * (1 - np.exp(-0.06 * d_demo))
        - 2.5 * d_demo
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(d_demo, f_demo, lw=2)

    ax.set_title("Exemple de courbe Pushover")
    ax.set_xlabel("Déplacement")
    ax.set_ylabel("Effort")

    st.pyplot(fig)
   
        
