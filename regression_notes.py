"""Regression lineaire multiple a partir du CSV de notes fourni."""

from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "donnees_regression_lineaire_multiple.csv"
FEATURES = ["note math", "note physique", "note français"]
TARGET = "note finale"


def load_dataset(path: Path = CSV_PATH) -> pd.DataFrame:
    """Charge et valide les notes depuis le fichier CSV."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier CSV introuvable : {path}. "
            "Placez le CSV dans le meme dossier que ce script."
        )

    df = pd.read_csv(path, encoding="utf-8-sig")
    missing_columns = [column for column in [*FEATURES, TARGET] if column not in df]
    if missing_columns:
        raise ValueError(
            "Colonnes absentes du CSV : " + ", ".join(missing_columns)
        )

    df = df[[*FEATURES, TARGET]].apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        missing_values = df.isna().sum()
        details = ", ".join(
            f"{column} ({count})"
            for column, count in missing_values.items()
            if count
        )
        raise ValueError(
            "Le CSV contient des valeurs manquantes ou non numeriques : "
            + details
        )
    if len(df) < 5:
        raise ValueError("Le CSV doit contenir au moins 5 lignes de donnees.")
    return df


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }


def main() -> None:
    df = load_dataset()
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)

    predictions = pd.DataFrame(
        {
            "Note finale reelle": y_test.to_numpy(),
            "Note finale predite": y_pred,
        },
        index=y_test.index + 1,
    )
    predictions.index.name = "Eleve"
    predictions = predictions.sort_index()

    info_buffer = StringIO()
    df.info(buf=info_buffer)
    lines = [
        "ATELIER 3 - REGRESSION LINEAIRE MULTIPLE",
        f"Fichier utilise : {CSV_PATH.name}",
        "",
        "1. Exploration du dataset",
        df.head().to_string(index=False),
        f"Dimensions : {df.shape[0]} lignes et {df.shape[1]} colonnes",
        "Informations generales :",
        info_buffer.getvalue().strip(),
        f"Valeurs manquantes : {int(df.isna().sum().sum())}",
        f"Variables independantes X : {', '.join(FEATURES)}",
        f"Variable a predire Y : {TARGET}",
        "",
        (
            "X contient les notes de mathematiques, physique et francais; "
            "Y est la note finale. Les donnees de test sont mises de cote "
            "pour evaluer le modele sur des eleves non utilises pour son "
            "entrainement."
        ),
        "",
        "2. Entrainement (80 % apprentissage, 20 % test, random_state=42)",
        f"Lignes d'entrainement : {len(X_train)}",
        f"Lignes de test : {len(X_test)}",
        f"Nombre de coefficients : {len(model.coef_)}",
        *[
            f"Coefficient {feature} : {coefficient:.6f}"
            for feature, coefficient in zip(FEATURES, model.coef_)
        ],
        f"Constante (intercept) : {model.intercept_:.6f}",
        (
            "Chaque coefficient estime la variation de la note finale pour "
            "un point supplementaire dans la matiere correspondante, les "
            "autres notes restant constantes. L'intercept est la prediction "
            "lorsque les trois notes valent zero."
        ),
        "",
        "3. Notes reelles et predites sur le jeu de test",
        predictions.to_string(float_format=lambda value: f"{value:.2f}"),
        "",
        "4. Metriques d'evaluation",
        *[f"{name} : {value:.4f}" for name, value in metrics.items()],
        (
            "Le MAE indique l'erreur absolue moyenne en points. Le RMSE, "
            "lui aussi en points, penalise davantage les grandes erreurs. "
            "Un R2 proche de 1 signifie que le modele explique une grande "
            "part de la variation des notes finales."
        ),
        "",
        "5. Exemple de nouvelle prediction",
    ]

    example_student = pd.DataFrame([X.mean().to_dict()], columns=FEATURES)
    example_prediction = model.predict(example_student)[0]
    lines.append(
        "Pour un eleve ayant les notes moyennes du CSV "
        f"({', '.join(f'{name} : {value:.2f}' for name, value in example_student.iloc[0].items())}), "
        f"la note finale predite est {example_prediction:.2f}/20."
    )
    lines.append(
        "Selon le modele, un point supplementaire en mathematiques, "
        "a notes de physique et de francais constantes, modifie la note "
        f"finale predite de {model.coef_[0]:.4f} point."
    )
    lines.extend(
        [
            "",
            (
                "Remarque : le CSV contient des notes scolaires et non les "
                "donnees de ventes de boulangerie du sujet. Le modele, les "
                "predictions et le graphique portent donc sur la note finale."
            ),
            (
                "Le travail supplementaire sur les commandes livrees ne "
                "peut pas etre effectue avec ce CSV : il ne contient ni "
                "donnees de boulangerie ni variable de commandes livrees."
            ),
        ]
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        predictions.index,
        predictions["Note finale reelle"],
        marker="o",
        label="Note finale reelle",
    )
    plt.plot(
        predictions.index,
        predictions["Note finale predite"],
        marker="x",
        label="Note finale predite",
    )
    plt.xlabel("Eleve (numero de ligne dans le CSV)")
    plt.ylabel("Note finale")
    plt.title("Notes finales reelles et predites - jeu de test")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    chart_path = BASE_DIR / "comparaison_notes_reelles_predites.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    report_path = BASE_DIR / "resultats_atelier3.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nGraphique enregistre : {chart_path}")
    print(f"Resultats enregistres : {report_path}")


if __name__ == "__main__":
    main()
