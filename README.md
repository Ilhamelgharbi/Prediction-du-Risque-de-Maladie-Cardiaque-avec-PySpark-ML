
#  Prédiction du Risque de Maladie Cardiaque avec PySpark ML

Ce projet implémente un **pipeline complet de Machine Learning** pour prédire la présence de maladies cardiovasculaires à partir de données médicales. Il est construit avec **PySpark ML** et inclut toutes les étapes : nettoyage, EDA, feature engineering, entraînement, évaluation et interprétation du modèle.

---

## 🔹 Table des matières

1. [Objectif](#objectif)
2. [Technologies et librairies](#technologies-et-librairies)
3. [Préparation de l'environnement](#préparation-de-lenvironnement)
4. [Chargement et exploration des données](#chargement-et-exploration-des-données)
5. [Nettoyage et recodage](#nettoyage-et-recodage)
6. [Analyse exploratoire (EDA)](#analyse-exploratoire-eda)
7. [Feature engineering et preprocessing](#feature-engineering-et-preprocessing)
8. [Construction des pipelines ML](#construction-des-pipelines-ml)
9. [Entraînement et évaluation](#entraînement-et-évaluation)
10. [Interprétation des modèles](#interprétation-des-modèles)
11. [Sauvegarde et utilisation du modèle](#sauvegarde-et-utilisation-du-modèle)
12. [Conclusions](#conclusions)

---

## 🎯 Objectif

Prédire la présence ou l’absence d’une **maladie cardiovasculaire** à partir de caractéristiques cliniques telles que :

* Âge, Sexe
* Pression artérielle, Cholestérol, Glycémie
* Fréquence cardiaque, ECG, Angine après effort
* Fluoroscopie, Thalassémie

Le modèle final permettra d’aider les médecins à **évaluer le risque de maladie cardiaque** rapidement.

---

## 🛠 Technologies et librairies

* **Python 3.12**
* **PySpark 3.5.1** (DataFrame, MLlib)
* **Pandas / Matplotlib / Seaborn** pour la visualisation
* Modèles de classification PySpark : Logistic Regression, Random Forest, Gradient Boosted Trees

---

## ⚙️ Préparation de l'environnement

```bash
# Installer PySpark si nécessaire
pip install pyspark
```

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Créer une session Spark
spark = SparkSession.builder.appName("Heart Disease Prediction").getOrCreate()
print("Spark Version:", spark.version)
```

---

## 📂 Chargement et exploration des données

```python
df = spark.read.csv("heart-disease.csv", header=True, inferSchema=True)
df.printSchema()
df.show(5)
print(f"Nombre total de lignes: {df.count()}")
df.describe().show()
```

* Dataset : 303 patients, 15 colonnes
* Variables mixtes : numériques et catégorielles
* Valeurs manquantes identifiées (`?`) et supprimées

---

## 🧹 Nettoyage et recodage

* Recodage des variables catégorielles pour plus de lisibilité (ex : `Sex: 0 → Female, 1 → Male`)
* Conversion de la variable cible en binaire (`Disease: 0 → no, 1-4 → yes`)
* Suppression des valeurs manquantes

```python
df_clean.show(5)
```

---

## 📊 Analyse exploratoire (EDA)

* Statistiques descriptives pour variables numériques et catégorielles
* Visualisation des distributions (histogrammes, boxplots)
* Vérification de l’équilibre de la variable cible (`no ≈ 160`, `yes ≈ 137`)

---

## ⚡ Feature Engineering et Preprocessing

* **StringIndexer + OneHotEncoder** pour variables catégorielles
* **VectorAssembler + StandardScaler** pour variables numériques
* Création de nouvelles features :

  * `Cholesterol_Blood_Pressure_Ratio`
  * `Age_BP_Interaction`
  * `BP_HeartRate_Interaction`

---

## 🏗 Construction des pipelines ML

* Pipeline complet pour **Logistic Regression, Random Forest, GBT**
* Label indexé avec `StringIndexer`
* Assemblage de toutes les features en vecteur final

---

## 🚀 Entraînement et évaluation

* Split train/test : 70% / 30%
* Modèles entraînés : Logistic Regression, Random Forest, GBT
* Validation croisée (3 folds) et tuning d’hyperparamètres

### Exemple de métriques sur test set

| Modèle              | Accuracy | F1-score | Precision | Recall |
| ------------------- | -------- | -------- | --------- | ------ |
| Logistic Regression | 0.844    | 0.845    | 0.865     | 0.844  |
| Random Forest       | 0.818    | 0.819    | 0.848     | 0.818  |
| GBT                 | 0.805    | 0.806    | 0.830     | 0.805  |

* Matrices de confusion visualisées avec **Seaborn heatmap**

---

## 🌟 Interprétation des modèles

* Importance des features calculée pour Random Forest et GBT
* Variables les plus influentes : `Glycemia`, `BP_HeartRate_Interaction`, `Sex_encoded`

---

## 💾 Sauvegarde et utilisation du modèle

```python
# Sauvegarde des pipelines
model_lr.save("pipeline_lr_model.")
model_rf.save("pipeline_rf_model.")
model_gbt.save("pipeline_gbt_model.")

# Chargement et prédiction sur de nouvelles observations
from pyspark.ml.pipeline import PipelineModel
loaded_pipeline_lr = PipelineModel.load("pipeline_lr_model")
predictions_new = loaded_pipeline_lr.transform(new_df)
predictions_new.select("features", "prediction", "probability").show(truncate=False)
```

* Permet de prédire rapidement le risque pour de **nouveaux patients**

---

## 📌 Conclusions

* Pipeline PySpark complet pour prédiction de maladies cardiovasculaires
* Modèles précis, robustes et interprétables
* Prêt pour intégration dans une application **de support à la décision médicale**

---

## 🔗 Liens utiles

* [Documentation PySpark ML](https://spark.apache.org/docs/latest/ml-classification-regression.html)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)


