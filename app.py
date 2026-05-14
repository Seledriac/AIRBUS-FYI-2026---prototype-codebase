import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import json
import os
import shap


st.set_page_config(page_title="Airbus Federated RUL", layout="wide")


# =========================
# LOAD MODELS & DATA
# =========================

@st.cache_resource
def load_models():
    fed = tf.keras.models.load_model("models/global_model.keras")
    central = (
        tf.keras.models.load_model("models/centralized_model.keras")
        if os.path.exists("models/centralized_model.keras") else None
    )
    local = (
        tf.keras.models.load_model("models/local_model.keras")
        if os.path.exists("models/local_model.keras") else None
    )
    return fed, central, local


fed_model, central_model, local_model = load_models()
scaler = joblib.load("models/scaler.pkl")
df = pd.read_csv("data/generated/engine_data.csv")

FEATURE_NAMES = ["Cycle moteur", "Température (°C)", "Vibration (g)", "Pression (bar)"]

@st.cache_resource
def get_shap_explainer(_model, _background):
    return shap.Explainer(_model, _background)

_bg = pd.DataFrame(
    scaler.transform(df[["cycle", "temperature", "vibration", "pressure"]].sample(100, random_state=42)),
    columns=FEATURE_NAMES,
)
shap_explainer = get_shap_explainer(fed_model, _bg)


# =========================
# SIDEBAR
# =========================

st.sidebar.title("Aircraft Selection")

engine_id = st.sidebar.selectbox(
    "Select Engine",
    df["engine_id"].unique()
)

engine_df = df[df["engine_id"] == engine_id].copy()


# =========================
# PREDICTIONS (3 modèles)
# =========================

X = engine_df[["cycle", "temperature", "vibration", "pressure"]]
X_scaled = scaler.transform(X)

engine_df["RUL_Fed"] = fed_model.predict(X_scaled, verbose=0)

if central_model is not None:
    engine_df["RUL_Central"] = central_model.predict(X_scaled, verbose=0)

if local_model is not None:
    engine_df["RUL_Local"] = local_model.predict(X_scaled, verbose=0)


# =========================
# KPI (basé sur fédéré)
# =========================

latest_rul = int(engine_df["RUL_Fed"].iloc[-1])

if latest_rul > 120:
    risk, color = "LOW", "green"
elif latest_rul > 50:
    risk, color = "MEDIUM", "orange"
else:
    risk, color = "CRITICAL", "red"


# =========================
# HELPER
# =========================

def rul_chart(cycles, rul_real, rul_pred, title, pred_color):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(cycles, rul_real,  label="RUL réel",   color="steelblue", linewidth=1.5)
    ax.plot(cycles, rul_pred,  label="RUL prédit",  color=pred_color,  linestyle="--", linewidth=1.5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# =========================
# LAYOUT
# =========================

tab1, tab2, tab3 = st.tabs(["Dashboard", "Comparaison IA", "Explications"])


with tab1:

    # ── Titre + KPI ──────────────────────────────────────────────────
    title_col, risk_col = st.columns([3, 1])

    with title_col:
        st.title("Federated Predictive Maintenance Dashboard")
        st.caption(
            f"Moteur : Engine {engine_id}  |  "
            f"Profil : {engine_df['profile'].iloc[0]}  |  "
            f"{len(engine_df)} cycles"
        )

    with risk_col:
        st.subheader("Niveau de risque")
        st.markdown(
            f"<h2 style='color:{color}'>{risk}</h2>",
            unsafe_allow_html=True
        )
        st.metric("RUL prédit (Fédéré)", latest_rul)

    # ── 3 graphes RUL ────────────────────────────────────────────────
    st.subheader("RUL prédit vs RUL réel — les 3 approches")

    c_fed, c_cen, c_loc = st.columns(3)

    with c_fed:
        st.pyplot(rul_chart(
            engine_df["cycle"], engine_df["RUL"], engine_df["RUL_Fed"],
            "🟢 Fédéré", "green"
        ))

    with c_cen:
        if central_model is not None:
            st.pyplot(rul_chart(
                engine_df["cycle"], engine_df["RUL"], engine_df["RUL_Central"],
                "🔵 Centralisé", "royalblue"
            ))
        else:
            st.info("Lance `python comparison.py` pour voir ce graphe.")

    with c_loc:
        if local_model is not None:
            st.pyplot(rul_chart(
                engine_df["cycle"], engine_df["RUL"], engine_df["RUL_Local"],
                "🔴 Local (silo)", "tomato"
            ))
        else:
            st.info("Lance `python comparison.py` pour voir ce graphe.")

    # ── SHAP ─────────────────────────────────────────────────────────
    st.subheader("SHAP Explainability")

    X_engine_df = pd.DataFrame(X_scaled, columns=FEATURE_NAMES)
    shap_vals = shap_explainer(X_engine_df.values)
    vals = np.array(shap_vals.values)
    if vals.ndim == 3:
        vals = vals[:, :, 0]
    mean_abs = np.abs(vals).mean(axis=0)
    normalized = mean_abs / mean_abs.sum()

    fig_shap, ax_shap = plt.subplots(figsize=(7, 3))
    bars = ax_shap.barh(FEATURE_NAMES, normalized, color=["#2ecc71", "#e74c3c", "#3498db", "#f39c12"])
    for bar, val in zip(bars, normalized):
        ax_shap.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=10, fontweight="bold"
        )
    ax_shap.set_xlim(0, 1.15)
    ax_shap.set_xlabel("Importance relative (SHAP normalisé)")
    ax_shap.set_title(f"Impact des features sur le RUL prédit — Engine {engine_id}")
    ax_shap.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_shap)


with tab2:
    st.title("Fédéré vs Centralisé vs Local")

    comparison_path = "models/comparison_results.json"

    if not os.path.exists(comparison_path):
        st.warning(
            "Lance `python comparison.py` pour générer les résultats de comparaison.",
            icon="⚠️"
        )
    else:
        with open(comparison_path) as f:
            comp = json.load(f)

        m = comp["metrics"]

        # ── Tableau ──────────────────────────────────────────────────
        st.subheader("Résultats de la comparaison")

        table_data = {
            "Modèle": ["Local (silo)", "Centralisé", "Fédéré (notre approche)"],
            "MAE (cycles)": [m["local"]["mae"], m["centralized"]["mae"], m["federated"]["mae"]],
            "R²": [m["local"]["r2"], m["centralized"]["r2"], m["federated"]["r2"]],
            "Confidentialité": ["totale", "nulle", "totale"],
            "Verdict": ["Trop d'erreurs", "Données exposées", "Recommandé"],
        }

        df_table = pd.DataFrame(table_data)

        st.dataframe(
            df_table.style.highlight_min(subset=["MAE (cycles)"], color="#d4edda")
                          .highlight_max(subset=["R²"], color="#d4edda"),
            hide_index=True,
            use_container_width=True,
        )

        # ── Métriques par round (fédéré) ─────────────────────────────
        st.subheader("Métriques par round — Modèle Fédéré")
        st.caption("À chaque round de fédération, les moteurs partagent leurs modèles — jamais leurs données brutes.")

        if comp.get("round_metrics"):
            df_rounds = pd.DataFrame(comp["round_metrics"])
            df_rounds.columns = ["Round", "MAE Train", "MAE Test", "R² Test"]
            st.dataframe(
                df_rounds.style.format({
                    "MAE Train": "{:.2f}",
                    "MAE Test":  "{:.2f}",
                    "R² Test":   "{:.4f}",
                }),
                hide_index=True,
                use_container_width=True,
            )

        # ── Message clé ───────────────────────────────────────────────
        st.info(
            "**Pourquoi le fédéré ?**  \n"
            "Le modèle fédéré atteint une précision proche du centralisé "
            "sans jamais exposer les données brutes des avions — "
            "seuls les poids agrégés circulent entre les clients et le serveur d'agrégation."
        )


with tab3:
    st.title("Explications")

    st.subheader("1. FedAvg : 3 clients × 5 rounds × 5 epochs")
    st.markdown(
        "3 clients = 3 compagnies aériennes, chacune garde ses données. "
        "À chaque round, elles entraînent localement pendant 5 epochs "
        "(= 5 lectures complètes de leurs données), puis envoient uniquement leurs **poids** au serveur "
        "mais jamais les données brutes. Le serveur fait la moyenne et redistribue un modèle global amélioré. "
        "5 rounds plus tard, le cerveau collectif surpasse chaque modèle individuel."
    )

    st.divider()

    st.subheader("2. Qu'est-ce que la feature cycle moteur dans SHAP ?")
    st.markdown(
        "C'est la variable de temps : plus le cycle est élevé, plus le moteur est usé et moins il reste de vie. "
        "Mais les autres capteurs (température, vibration, pression) apportent la vraie valeur ajoutée : "
        "deux moteurs au même cycle peuvent avoir des RUL très différents si l'un montre des signes de "
        "dégradation accélérée. C'est là que l'IA dépasse le simple comptage."
    )

    st.divider()

    st.subheader("3. Pourquoi le modèle silo prédit un RUL croissant sur un moteur en fin de vie ?")
    st.markdown(
        "Le modèle local n'a été entraîné que sur un seul profil de moteur. "
        "Face à un profil différent avec des températures et vibrations qu'il n'a jamais vues, "
        "il extrapole de façon arbitraire et prédit un RUL qui monte alors que le moteur est déjà "
        "à zéro cycle restant. En aviation ou dans le spatial, ce serait catastrophique : le système dirait "
        "\"pas besoin de révision\" sur un moteur en défaillance. "
        "Le modèle fédéré élimine ce risque car chaque client a, indirectement, appris de tous les profils."
    )
