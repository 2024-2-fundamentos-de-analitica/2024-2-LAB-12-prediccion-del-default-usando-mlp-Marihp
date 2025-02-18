import pandas as pd
import zipfile
import pickle
import gzip
import os
import json
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier


def clean_dataset(path):
    """Carga y limpia los datasets."""
    with zipfile.ZipFile(path, "r") as z:
        csv_file = z.namelist()[0]
        with z.open(csv_file) as f:
            df = pd.read_csv(f)

    # Renombrar la columna "default payment next month" a "default"
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    # Remover la columna "ID"
    df.drop(columns=["ID"], inplace=True)
    # Eliminar registros con información no disponible
    df.dropna(inplace=True)
    # Agrupar valores de EDUCATION > 4 como "others" (se reemplaza por 4)
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x <= 4 else 4)

    return df


# Cargar datasets
df_test = clean_dataset("./files/input/test_data.csv.zip")
df_train = clean_dataset("./files/input/train_data.csv.zip")

# Separar variables
x_train = df_train.drop(columns=["default"])
y_train = df_train["default"]

x_test = df_test.drop(columns=["default"])
y_test = df_test["default"]


def build_pipeline():
    """Construye un pipeline con preprocesamiento, PCA, selección de características y MLP."""
    # Definir columnas categóricas y numéricas
    categorical_features = ["EDUCATION", "MARRIAGE", "SEX"]
    numeric_features = [
        col for col in x_train.columns if col not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    # Ajustamos el preprocesador para conocer el número de características resultantes
    preprocessor.fit(x_train)
    x_train_transformed = preprocessor.transform(x_train)
    num_features_after_preprocessing = x_train_transformed.shape[1]
    print(
        f"Número de características después del preprocesamiento: {num_features_after_preprocessing}"
    )

    # Configuración inicial de SelectKBest (el valor k se sobreescribirá en GridSearchCV)
    k_best = SelectKBest(f_classif, k=min(10, num_features_after_preprocessing))

    # Modelo MLP con random_state para reproducibilidad, early stopping y más iteraciones
    model = MLPClassifier(random_state=21, max_iter=1500, early_stopping=True)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("k_best", k_best),
            ("pca", PCA()),  # Se parametriza en GridSearchCV
            ("classifier", model),
        ]
    )

    return pipeline


def optimize_pipeline(pipeline, x_train, y_train):
    """Optimiza el pipeline usando GridSearchCV con 10-fold cross-validation."""
    param_grid = {
        "pca__n_components": [None],
        "k_best__k": [20],
        "classifier__hidden_layer_sizes": [(50, 30, 40, 60), (100,)],
        "classifier__alpha": [0.26],
        "classifier__learning_rate_init": [0.001],
        "classifier__activation": ["relu"],
        "classifier__solver": ["adam"],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
        verbose=2,
    )

    print("Optimizando hiperparámetros con GridSearchCV...")
    grid_search.fit(x_train, y_train)
    print("Optimización finalizada.")
    print("Mejores parámetros:", grid_search.best_params_)
    print("Mejor balanced_accuracy:", grid_search.best_score_)

    return grid_search


def save_model(model, file_path="files/models/model.pkl.gz"):
    """Guarda el modelo entrenado en un archivo comprimido."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with gzip.open(file_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en {file_path}")


def evaluate_model(
    model, x_train, y_train, x_test, y_test, file_path="files/output/metrics.json"
):
    """Evalúa el modelo en los conjuntos de entrenamiento y prueba y guarda las métricas."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        for dataset, (x, y) in zip(
            ["train", "test"], [(x_train, y_train), (x_test, y_test)]
        ):
            y_pred = model.predict(x)
            metrics = {
                "type": "metrics",
                "dataset": dataset,
                "precision": precision_score(y, y_pred, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y, y_pred),
                "recall": recall_score(y, y_pred, zero_division=0),
                "f1_score": f1_score(y, y_pred, zero_division=0),
            }
            f.write(json.dumps(metrics) + "\n")

            cm = confusion_matrix(y, y_pred)
            cm_data = {
                "type": "cm_matrix",
                "dataset": dataset,
                "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
                "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
            }
            f.write(json.dumps(cm_data) + "\n")

    print(f"Métricas guardadas en {file_path}")


# Flujo principal
print("Construcción del pipeline...")
pipeline = build_pipeline()

print("Optimización del modelo...")
best_pipeline = optimize_pipeline(pipeline, x_train, y_train)

print("Guardando el modelo...")
save_model(best_pipeline)

print("Evaluando el modelo y guardando métricas...")
evaluate_model(best_pipeline, x_train, y_train, x_test, y_test)

print("¡Proceso completado con éxito!")
