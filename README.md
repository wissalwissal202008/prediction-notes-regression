# Prédiction de la note finale — Régression linéaire multiple

Projet de Machine Learning (Atelier 3) : prédire la **note finale** d'un élève à partir de ses notes en **mathématiques**, **physique** et **français** grâce à une régression linéaire multiple (scikit-learn), avec une interface interactive **Streamlit**.

## Résultats

| Métrique | Valeur |
|----------|--------|
| MAE      | 0.668  |
| MSE      | 0.653  |
| RMSE     | 0.808  |
| R²       | 0.908  |

Le modèle explique environ **91 %** de la variation des notes finales sur le jeu de test (80 % entraînement / 20 % test, `random_state=42`).

![Notes réelles vs prédites](comparaison_notes_reelles_predites.png)

## Structure du projet

```
├── app.py                                    # Interface Streamlit
├── regression_notes.py                                    # Chargement des données, entraînement, rapport et graphique
├── donnees_regression_lineaire_multiple.csv  # Dataset (100 élèves, 4 colonnes)
├── resultats_atelier3.txt                    # Rapport généré par regression_notes.py
├── comparaison_notes_reelles_predites.png    # Graphique généré par regression_notes.py
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/<votre-utilisateur>/prediction-notes-regression.git
cd prediction-notes-regression
pip install -r requirements.txt
```

## Utilisation

Générer le rapport et le graphique :

```bash
python regression_notes.py
```

Lancer l'application web :

```bash
streamlit run app.py
```

L'application permet d'explorer les données, de voir les métriques et les coefficients du modèle, et de saisir des notes pour obtenir une prédiction.

## Technologies

Python · pandas · NumPy · scikit-learn · Matplotlib · Streamlit
