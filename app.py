"""Interface Streamlit pour explorer et utiliser le modele des notes."""

import pandas as pd
import streamlit as st
from regression_notes import (
    FEATURES,
    NEW_FEATURE,
    NEW_STUDENT,
    build_enriched_dataset,
    calculate_metrics,
    cross_validated_r2,
    load_dataset,
    split_data,
    train_model,
)


st.set_page_config(
    page_title="Prédiction des notes",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Prédiction de la note finale")
st.caption(
    "Régression linéaire multiple entraînée à partir de "
    "`donnees_regression_lineaire_multiple.csv`."
)

try:
    df = load_dataset()
except (FileNotFoundError, ValueError, UnicodeDecodeError, pd.errors.ParserError) as error:
    st.error(f"Impossible de charger le fichier CSV : {error}")
    st.info(
        "Vérifiez que donnees_regression_lineaire_multiple.csv se trouve "
        "dans le même dossier que app.py et contient les colonnes attendues."
    )
    st.stop()

X = df[FEATURES]
X_train, X_test, y_train, y_test = split_data(df)
model = train_model(X_train, y_train)
y_pred = model.predict(X_test)
metrics = calculate_metrics(y_test, y_pred)

st.subheader("Vue d'ensemble des données")
metric_columns = st.columns(4)
metric_columns[0].metric("Élèves", len(df))
metric_columns[1].metric("Matières explicatives", len(FEATURES))
metric_columns[2].metric("Entraînement", len(X_train))
metric_columns[3].metric("Test", len(X_test))
st.dataframe(df, width="stretch", hide_index=True)

st.subheader("Évaluation du modèle")
metric_columns = st.columns(4)
metric_columns[0].metric("MAE", f"{metrics['MAE']:.3f} point")
metric_columns[1].metric("MSE", f"{metrics['MSE']:.3f}")
metric_columns[2].metric("RMSE", f"{metrics['RMSE']:.3f} point")
metric_columns[3].metric("R²", f"{metrics['R2']:.3f}")

predictions = pd.DataFrame(
    {
        "Note réelle": y_test.to_numpy(),
        "Note prédite": y_pred,
    },
    index=y_test.index + 1,
)
predictions.index.name = "Élève"
predictions = predictions.sort_index()
st.line_chart(predictions, x_label="Élève (numéro de ligne)", y_label="Note sur 20")
with st.expander("Afficher les notes réelles et prédites"):
    st.dataframe(
        predictions.style.format("{:.2f}"),
        width="stretch",
    )

st.subheader("Prédire la note d’un élève")
st.write("Saisissez les notes sur 20 pour obtenir une prédiction.")
input_columns = st.columns(3)
student_scores = {
    feature: input_columns[index].number_input(
        feature.capitalize(),
        min_value=0.0,
        max_value=20.0,
        value=float(NEW_STUDENT[feature]),
        step=0.1,
        format="%.2f",
    )
    for index, feature in enumerate(FEATURES)
}
new_student = pd.DataFrame([student_scores], columns=FEATURES)
predicted_grade = float(model.predict(new_student)[0])
st.success(f"Note finale prédite : **{predicted_grade:.2f} / 20**")
st.caption(
    f"Question finale : +2 points en math (autres notes inchangées) "
    f"→ note finale {2 * model.coef_[0]:+.2f} point."
)

st.subheader("Coefficients du modèle")
coefficients = pd.DataFrame(
    {"Matière": FEATURES, "Coefficient": model.coef_}
)
st.dataframe(
    coefficients.style.format({"Coefficient": "{:.4f}"}),
    width="stretch",
    hide_index=True,
)
st.caption(
    f"Constante (intercept) : {model.intercept_:.4f}. "
    "Chaque coefficient estime l’effet d’un point supplémentaire dans "
    "la matière correspondante, à notes constantes dans les autres matières."
)

st.subheader("Travail supplémentaire : ajout des heures de révision")
st.caption(
    "Variable absente du CSV du professeur : valeurs simulées, ajoutées en "
    "mémoire aux données du CSV (avec 5 nouveaux élèves)."
)
enriched = build_enriched_dataset(df, model)
rows = {}
for label, features in {
    "Ancien (3 variables)": FEATURES,
    "Nouveau (4 variables)": [*FEATURES, NEW_FEATURE],
}.items():
    Xe_train, Xe_test, ye_train, ye_test = split_data(enriched, features)
    scores = calculate_metrics(
        ye_test, train_model(Xe_train, ye_train).predict(Xe_test)
    )
    scores["R2 validation croisée"] = cross_validated_r2(enriched, features)
    rows[label] = scores
st.dataframe(pd.DataFrame(rows).style.format("{:.4f}"), width="stretch")
