"""Atelier 3 - Regression lineaire multiple appliquee au CSV fourni par le professeur.

Le sujet decrit une boulangerie, mais le jeu de donnees fourni
(donnees_regression_lineaire_multiple.csv) contient des notes d'eleves.
Chaque partie de l'atelier est donc realisee sur ces notes :

    X1 = note math, X2 = note physique, X3 = note français
    Y  = note finale
    Y  = a*X1 + b*X2 + c*X3 + d

Le CSV a ete ameliore comme le demande le travail supplementaire :
    - nouvelle colonne 'heures de révision' (valeurs simulees) ;
    - 5 nouveaux eleves a la fin (lignes 101 a 105).
Les parties 2 a 7 utilisent les 100 eleves d'origine et les 3 notes ;
le travail supplementaire utilise les 105 eleves et les 4 variables.

Execution : python regression_notes.py
Sorties   : resultats_atelier3.txt, comparaison_notes_reelles_predites.png
"""

import sys
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "donnees_regression_lineaire_multiple.csv"
CHART_PATH = BASE_DIR / "comparaison_notes_reelles_predites.png"
REPORT_PATH = BASE_DIR / "resultats_atelier3.txt"

FEATURES = ["note math", "note physique", "note français"]
TARGET = "note finale"
NEW_FEATURE = "heures de révision"
ALL_FEATURES = [*FEATURES, NEW_FEATURE]
# Nombre d'eleves d'origine fournis par le professeur ; les lignes suivantes
# sont les nouveaux eleves ajoutes pour le travail supplementaire.
ORIGINAL_ROWS = 100

# Partie 7 : nouvel eleve a predire (equivalent de la "nouvelle journee").
NEW_STUDENT = {"note math": 14.0, "note physique": 12.0, "note français": 15.0}
# Question finale : augmentation de la note de math (sans changer les autres).
MATH_INCREASE = 2.0


# ---------------------------------------------------------------------------
# Fonctions reutilisees par app.py
# ---------------------------------------------------------------------------
def load_dataset(path: Path = CSV_PATH, features: list[str] = FEATURES) -> pd.DataFrame:
    """Charge et valide le CSV (colonnes presentes, valeurs numeriques)."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier CSV introuvable : {path}. "
            "Placez le CSV dans le meme dossier que ce script."
        )

    df = pd.read_csv(path, encoding="utf-8-sig")
    columns = [*features, TARGET]
    missing_columns = [column for column in columns if column not in df]
    if missing_columns:
        raise ValueError("Colonnes absentes du CSV : " + ", ".join(missing_columns))

    df = df[columns].apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        details = ", ".join(
            f"{column} ({count})" for column, count in df.isna().sum().items() if count
        )
        raise ValueError(
            "Le CSV contient des valeurs manquantes ou non numeriques : " + details
        )
    if len(df) < 5:
        raise ValueError("Le CSV doit contenir au moins 5 lignes de donnees.")
    return df


def split_data(df: pd.DataFrame, features: list[str] = FEATURES):
    """Separation 80 % entrainement / 20 % test, comme dans l'atelier."""
    return train_test_split(df[features], df[TARGET], test_size=0.2, random_state=42)


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def cross_validated_r2(df: pd.DataFrame, features: list[str]) -> float:
    """R2 moyen en validation croisee 5 plis (plus stable qu'un seul decoupage)."""
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        LinearRegression(), df[features], df[TARGET], cv=folds, scoring="r2"
    )
    return float(scores.mean())


def load_original_and_enriched() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Renvoie (donnees d'origine : 100 eleves, 3 notes ; dataset complet ameliore)."""
    enriched = load_dataset(CSV_PATH, ALL_FEATURES)
    original = enriched.iloc[:ORIGINAL_ROWS][[*FEATURES, TARGET]]
    return original, enriched


# ---------------------------------------------------------------------------
# Programme complet de l'atelier
# ---------------------------------------------------------------------------
def fmt_metrics(metrics: dict[str, float]) -> list[str]:
    return [f"{name:<5}: {value:.4f}" for name, value in metrics.items()]


def main() -> None:
    lines: list[str] = []
    add = lines.extend

    # ---------------- Partie 2 : exploration ----------------
    df, enriched = load_original_and_enriched()
    info_buffer = StringIO()
    df.info(buf=info_buffer)
    add([
        "ATELIER 3 - REGRESSION LINEAIRE MULTIPLE",
        f"Jeu de donnees fourni : {CSV_PATH.name}",
        "Adaptation du sujet : la 'boulangerie' est remplacee par les notes",
        "d'eleves du CSV. Clients/Publicite/Reduction -> notes math/physique/",
        "français, et le chiffre d'affaires -> note finale.",
        f"Parties 2 a 7 : les {ORIGINAL_ROWS} eleves d'origine et les 3 notes.",
        "",
        "=== PARTIE 2 : Exploration du dataset ===",
        "Cinq premieres lignes :",
        df.head().to_string(index=False),
        "",
        f"Dimensions : {df.shape}",
        "Informations generales :",
        info_buffer.getvalue().strip(),
        "",
        "Valeurs manquantes par colonne :",
        df.isna().sum().to_string(),
        "",
        "Statistiques descriptives :",
        df.describe().round(2).to_string(),
        "",
        "Reponses :",
        f"- Le dataset contient {df.shape[0]} lignes (eleves).",
        f"- Il possede {len(FEATURES)} variables independantes : {', '.join(FEATURES)}.",
        f"- La variable a predire est la '{TARGET}'.",
        "",
    ])

    # ---------------- Partie 3 : preparation ----------------
    X_train, X_test, y_train, y_test = split_data(df)
    add([
        "=== PARTIE 3 : Preparation des donnees ===",
        f"X = df[{FEATURES}]",
        f"y = df['{TARGET}']",
        f"Entrainement : {len(X_train)} lignes | Test : {len(X_test)} lignes",
        "",
        "Reponses :",
        "- X contient les variables explicatives (les 3 notes) que le modele",
        "  utilise pour faire sa prediction.",
        "- Y est la valeur cible (note finale) que le modele doit apprendre a predire.",
        "- On separe entrainement et test pour evaluer le modele sur des eleves",
        "  qu'il n'a jamais vus : cela mesure sa capacite a generaliser et",
        "  detecte le sur-apprentissage.",
        "- X_train sert a entrainer le modele (80 %), X_test sert uniquement a",
        "  l'evaluer (20 %).",
        "",
    ])

    # ---------------- Partie 4 : entrainement ----------------
    model = train_model(X_train, y_train)
    a, b, c = model.coef_
    d = model.intercept_
    top_feature = FEATURES[int(np.argmax(model.coef_))]
    add([
        "=== PARTIE 4 : Entrainement du modele ===",
        f"Coefficients : {np.round(model.coef_, 6)}",
        f"Constante    : {d:.6f}",
        "",
        "Equation apprise :",
        f"  note finale = {a:.4f} * math + {b:.4f} * physique + {c:.4f} * français + {d:.4f}",
        "",
        "Reponses :",
        f"- Le modele a calcule {len(model.coef_)} coefficients (un par variable independante).",
        f"- a = {a:.4f} (math), b = {b:.4f} (physique), c = {c:.4f} (français).",
        "- La constante (intercept) est la note finale predite quand les trois",
        "  notes valent 0 ; elle sert surtout a ajuster le niveau de la droite.",
        "- Il y a plusieurs coefficients car il y a plusieurs variables : chacune",
        "  a son propre effet sur la note finale.",
        "",
        "Interpretation (les autres notes restant constantes) :",
        f"- +1 point en math       -> note finale {a:+.2f} point.",
        f"- +1 point en physique   -> note finale {b:+.2f} point.",
        f"- +1 point en français   -> note finale {c:+.2f} point.",
        "- Les trois coefficients sont positifs et proches (~0.3) : la note",
        "  finale ressemble a une moyenne ponderee des trois matieres ;",
        f"  la variable la plus influente est '{top_feature}'.",
        "",
    ])

    # ---------------- Partie 5 : evaluation ----------------
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    results = pd.DataFrame(
        {"Note reelle": y_test.to_numpy(), "Note predite": y_pred},
        index=y_test.index + 1,
    )
    results["Erreur"] = results["Note predite"] - results["Note reelle"]
    results.index.name = "Eleve"
    results = results.sort_index()
    add([
        "=== PARTIE 5 : Evaluation du modele ===",
        results.to_string(float_format=lambda v: f"{v:.2f}"),
        "",
        *fmt_metrics(metrics),
        "",
        "Reponses :",
        f"- MAE = erreur absolue moyenne : en moyenne le modele se trompe de",
        f"  {metrics['MAE']:.2f} point sur la note finale.",
        "- Le MSE est en 'points au carre' ; la racine carree (RMSE) remet",
        "  l'erreur dans la meme unite que la note, donc plus facile a lire.",
        "  Le RMSE penalise davantage les grosses erreurs que le MAE.",
        f"- R2 = {metrics['R2']:.3f} : proche de 1, le modele explique environ",
        f"  {metrics['R2'] * 100:.0f} % de la variation des notes finales.",
        f"- (Equivalent de 'MAE = 30 DH') : ici MAE = {metrics['MAE']:.2f}, soit une",
        f"  erreur moyenne d'environ {metrics['MAE']:.2f} point sur 20 par eleve.",
        "",
    ])

    # ---------------- Partie 6 : visualisation ----------------
    plt.figure(figsize=(8, 5))
    positions = range(len(results))
    plt.plot(positions, results["Note reelle"], marker="o", label="Note réelle")
    plt.plot(positions, results["Note predite"], marker="x", label="Note prédite")
    plt.xticks(positions, results.index)
    plt.xlabel("Élève (numéro de ligne dans le CSV)")
    plt.ylabel("Note finale (/20)")
    plt.title("Comparaison entre note finale réelle et prédite (jeu de test)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    plt.close()
    worst = results["Erreur"].abs().idxmax()
    add([
        "=== PARTIE 6 : Visualisation ===",
        f"Graphique enregistre : {CHART_PATH.name}",
        "Reponse : oui, les courbes reelle et predite se suivent de pres ;",
        f"la plus grande erreur est de {results['Erreur'].abs().max():.2f} point (eleve {worst}).",
        "",
    ])

    # ---------------- Partie 7 : nouvelle prediction ----------------
    new_student = pd.DataFrame([NEW_STUDENT], columns=FEATURES)
    prediction = model.predict(new_student)[0]
    improved = new_student.copy()
    improved["note math"] += MATH_INCREASE
    prediction_improved = model.predict(improved)[0]
    add([
        "=== PARTIE 7 : Nouvelle prediction ===",
        "Nouvel eleve : " + ", ".join(f"{k} = {v:g}" for k, v in NEW_STUDENT.items()),
        f"Note finale prevue : {prediction:.2f} / 20",
        f"Interpretation : avec ces notes, le modele prevoit {prediction:.2f}/20,",
        "coherent avec la moyenne ponderee des trois matieres.",
        "",
        f"Question finale : si la note de math augmente de {MATH_INCREASE:g} points",
        "(physique et français inchanges), la note finale predite varie de",
        f"  {MATH_INCREASE:g} x a = {MATH_INCREASE:g} x {a:.4f} = {MATH_INCREASE * a:+.2f} point",
        f"  ({prediction:.2f} -> {prediction_improved:.2f}).",
        "(Les notes etant sur 20, on utilise +2 points au lieu des '+10 clients'",
        " du sujet ; le raisonnement est identique : variation = coefficient x hausse.)",
        "",
    ])

    # ---------------- Travail supplementaire ----------------
    features_new = ALL_FEATURES
    new_count = len(enriched) - ORIGINAL_ROWS

    Xo_train, Xo_test, yo_train, yo_test = split_data(enriched, FEATURES)
    Xn_train, Xn_test, yn_train, yn_test = split_data(enriched, features_new)
    old_model = train_model(Xo_train, yo_train)
    new_model = train_model(Xn_train, yn_train)
    old_metrics = calculate_metrics(yo_test, old_model.predict(Xo_test))
    new_metrics = calculate_metrics(yn_test, new_model.predict(Xn_test))
    old_cv = cross_validated_r2(enriched, FEATURES)
    new_cv = cross_validated_r2(enriched, features_new)

    comparison = pd.DataFrame(
        {"Ancien (3 variables)": old_metrics, "Nouveau (4 variables)": new_metrics}
    )
    comparison.loc["R2 valid. croisee"] = [old_cv, new_cv]
    gain = new_cv - old_cv
    if gain > 0.01:
        verdict = "l'ajout de la variable AMELIORE les predictions."
    elif gain < -0.01:
        verdict = "l'ajout de la variable DEGRADE les predictions."
    else:
        verdict = "l'ajout de la variable n'ameliore PAS significativement les predictions."
    add([
        "=== TRAVAIL SUPPLEMENTAIRE ===",
        f"1. Nouvelle variable independante : '{NEW_FEATURE}' (par semaine).",
        "   (equivalent du 'nombre de commandes livrees' du sujet)",
        "   Colonne ajoutee au CSV. ATTENTION : le CSV d'origine du professeur",
        "   ne contenait pas cette variable ; ses valeurs sont SIMULEES, liees",
        "   au niveau general de l'eleve (moyenne des 3 notes) plus un bruit.",
        f"2. {new_count} nouveaux eleves ajoutes a la fin du CSV -> {len(enriched)} lignes.",
        "   Nouveaux eleves :",
        enriched.tail(new_count).to_string(index=False),
        "3. Les deux modeles sont reentraines sur le MEME dataset ameliore et le",
        "   MEME decoupage 80/20, pour une comparaison equitable.",
        "   Coefficients du nouveau modele :",
        *[f"     {f} : {coef:.4f}" for f, coef in zip(features_new, new_model.coef_)],
        f"     constante : {new_model.intercept_:.4f}",
        "4. Comparaison des performances :",
        comparison.round(4).to_string(),
        f"5. Conclusion : gain de R2 en validation croisee = {gain:+.4f} ->",
        f"   {verdict}",
        "   Explication : les heures de revision sont deja fortement liees aux",
        "   trois notes ; elles apportent donc peu d'information nouvelle.",
        "   Ajouter une variable n'ameliore un modele que si elle apporte une",
        "   information que les autres variables ne contiennent pas deja.",
        "",
    ])

    # ---------------- Interpretation finale (livrable) ----------------
    add([
        "=== INTERPRETATION DES RESULTATS ===",
        f"Le modele de regression lineaire multiple predit la note finale avec",
        f"un R2 de {metrics['R2']:.2f} et une erreur moyenne (MAE) de {metrics['MAE']:.2f} point.",
        "Les trois matieres contribuent positivement et de facon comparable,",
        f"'{top_feature}' ayant le poids le plus eleve. Le modele est fiable pour",
        "estimer la note finale d'un eleve a partir de ses trois notes.",
    ])

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nGraphique enregistre : {CHART_PATH}")
    print(f"Resultats enregistres : {REPORT_PATH}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
