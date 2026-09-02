# Prédiction de la RUL des moteurs turbofan (NASA CMAPSS – FD001)

Ce projet a pour objectif de prédire la **Remaining Useful Life (RUL)** de moteurs d'avion turbofan à partir du jeu de données **CMAPSS de la NASA** (sous-ensemble **FD001**), dans un contexte de **maintenance prédictive**.

Plusieurs approches de Machine Learning et de Deep Learning sont comparées :

- **Random Forest** — modèle tabulaire
- **Autoencoder LSTM + régression**
- **LSTM direct** pour la prédiction de la RUL

## Jeu de données

Le projet utilise les fichiers suivants du dataset CMAPSS (FD001) :

- **`train_FD001.txt`** — historique complet de plusieurs moteurs jusqu'à la panne. Chaque ligne contient :
  - `engine_id` : identifiant du moteur
  - `cycle` : numéro de cycle
  - `os1`, `os2`, `os3` : operating settings
  - `s1` … `s21` : mesures de 21 capteurs
- **`test_FD001.txt`** — historique de nouveaux moteurs, sans RUL explicite (utilisé pour le test).
- **`RUL_FD001.txt`** — pour chaque moteur du test, la RUL vraie au dernier cycle observé.

À partir de `train_FD001`, la RUL est reconstruite pour chaque cycle :

```
RUL = max_cycle_par_moteur − cycle_courant
```

## Approche globale

### 1. Prétraitement & exploration

- Lecture des fichiers `train_FD001` et `test_FD001`
- Suppression des capteurs constants ou quasi constants
- Normalisation des features (moyenne / écart-type calculés sur le train)

### 2. Construction des fenêtres temporelles

- Pour chaque moteur, les données sont découpées en **fenêtres glissantes** de taille fixe (ex. 30 cycles).
- Chaque échantillon = séquence de `WINDOW_SIZE` cycles d'un moteur.
- Cible = RUL au dernier cycle de la fenêtre.

### 3. Modèles entraînés

**Random Forest**
- Entrée : features tabulaires (snapshot à un cycle donné)
- Ne modélise pas explicitement la dynamique temporelle

**Autoencoder LSTM + régression**
- *Autoencoder LSTM* : entrée = séquences de capteurs, sortie = même séquence (reconstruction) ; le goulot (latent) sert de représentation compressée.
- *Régresseur* : entrée = vecteur latent, sortie = RUL.
- Apprentissage en deux temps : autoencoder (non supervisé) puis régression (supervisé).

**LSTM direct**
- Entrée : fenêtre de `WINDOW_SIZE` cycles (séquence)
- Sortie : RUL au dernier cycle
- Modèle séquentiel de bout en bout, sans étape de compression séparée

### 4. Évaluation

Pour chaque moteur du test, on prend la prédiction de RUL au dernier cycle disponible et on la compare à la RUL vraie de `RUL_FD001.txt`.

**Métriques :**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Pourcentage de moteurs prédits dans ±5, ±10, ±20 cycles

**Visualisations :**
- Nuages de points RUL vraie vs RUL prédite
- Histogrammes d'erreurs
- Comparaison par moteur (barres True vs Pred)

## Structure du dépôt

```
.
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_random_forest.ipynb
│   ├── 03_autoencoder_lstm.ipynb
│   └── 04_direct_lstm.ipynb
├── visualize.py          # fonctions utilitaires de visualisation (plots)
├── requirements.txt
└── README.md
```

## Installation & exécution

### 1. Création de l'environnement

```bash
# Création d'un environnement virtuel (Python 3.10)
python -m venv .venv

# Activation
source .venv/bin/activate        # Linux / Mac
.venv\Scripts\activate.bat       # Windows

# Installation des dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

Le fichier `requirements.txt` contient typiquement :

```
numpy
pandas
matplotlib
scikit-learn
tensorflow
```

### 2. Lancement des notebooks

```bash
jupyter notebook
```

Ouvrir ensuite les notebooks du dossier `notebooks/` et exécuter les cellules dans l'ordre.

## Résultats principaux (résumé qualitatif)

**Random Forest** — fournit un premier benchmark raisonnable, mais les erreurs sont plus dispersées et le modèle a tendance à surestimer la RUL (biais optimiste).

**Autoencoder LSTM + régression** — profite de la modélisation temporelle, mais la compression dans un latent space induit une certaine perte d'information ; les erreurs restent plus élevées et plus instables que pour le LSTM direct.

**LSTM direct** — le modèle le plus performant :
- nuage de points RUL vraie vs RUL prédite plus serré autour de la diagonale,
- distribution d'erreurs plus centrée, avec moins de valeurs extrêmes,
- meilleures valeurs de MAE / RMSE sur le jeu de test.

**Conclusion :** la prise en compte explicite de la dimension temporelle via un LSTM de bout en bout améliore significativement la prédiction de la RUL, par rapport à une approche tabulaire (Random Forest) ou à une architecture Autoencoder plus complexe.

## Pistes d'amélioration

- Tester d'autres architectures séquentielles : GRU, CNN 1D, modèles hybrides CNN–LSTM.
- Affiner la taille des fenêtres temporelles et éventuellement le clipping de la RUL.
- Améliorer la sélection de capteurs (feature selection plus poussée).
- Étendre l'évaluation aux autres sous-ensembles CMAPSS (FD002, FD003, FD004).

## Références

- NASA CMAPSS Turbofan Engine Degradation Simulation Data Set
- Littérature sur la prédiction de la RUL et la maintenance prédictive basée sur l'apprentissage profond
