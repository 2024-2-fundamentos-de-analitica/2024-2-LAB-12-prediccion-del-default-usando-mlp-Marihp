# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
import pandas as pd
import zipfile
import pickle
import gzip
import os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

import json
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".


def clean_dataset(path):
    """Carga y limpia los datasets."""
    with zipfile.ZipFile(path, "r") as z:
        csv_file = z.namelist()[0]
        with z.open(csv_file) as f:
            df = pd.read_csv(f)

    # 1. Renombrar la columna "default payment next month" a "default"
    df.rename(columns={"default payment next month": "default"}, inplace=True)

    # 2. Remover la columna "ID"
    df.drop(columns=["ID"], inplace=True)

    # 3. Eliminar registros con información no disponible (NaN)
    df.dropna(inplace=True)

    # 4. Agrupar valores de EDUCATION > 4 como "others"
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x <= 4 else 4)

    return df


#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.

# Cargar datasets
df_test = clean_dataset("./files/input/test_data.csv.zip")
df_train = clean_dataset("./files/input/train_data.csv.zip")

# Separar variables
x_train = df_train.drop(columns=["default"])
y_train = df_train["default"]

x_test = df_test.drop(columns=["default"])
y_test = df_test["default"]


#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
def build_pipeline():
    """Construye un pipeline con preprocesamiento, PCA, selección de características y MLP."""
    from sklearn.neural_network import MLPClassifier

    # Definir columnas categóricas y numéricas
    categorical_features = ["EDUCATION", "MARRIAGE", "SEX"]
    numeric_features = [
        col for col in x_train.columns if col not in categorical_features
    ]

    # One-Hot Encoding para variables categóricas y escalado para numéricas
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    # Aplicamos la transformación a los datos
    x_train_transformed = preprocessor.fit_transform(x_train)

    # Contamos el número de características después del preprocesamiento
    num_features_after_preprocessing = x_train_transformed.shape[1]
    print(
        f"Número de características después del preprocesamiento: {num_features_after_preprocessing}"
    )

    # Ajustar SelectKBest dinámicamente
    k_best = SelectKBest(f_classif, k=min(10, num_features_after_preprocessing))

    # Modelo MLP
    model = MLPClassifier()

    # Construcción del pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("pca", PCA()),
            ("k_best", k_best),
            ("classifier", model),
        ]
    )

    return pipeline


# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.


def optimize_pipeline(pipeline, x_train, y_train):
    """Optimiza el pipeline usando GridSearchCV con 10-fold cross-validation."""

    param_grid = {
        "pca__n_components": [31],
        "k_best__k": [22],
        "classifier__hidden_layer_sizes": [(50, 30, 40, 60)],
        "classifier__alpha": [0.33],
        "classifier__learning_rate_init": [0.001],
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
    print(grid_search.best_params_)

    return grid_search


def save_model(model, file_path="files/models/model.pkl.gz"):
    """Guarda el modelo entrenado en un archivo comprimido."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with gzip.open(file_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en {file_path}")


# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#


# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


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

            # Matriz de confusión
            cm = confusion_matrix(y, y_pred)
            cm_data = {
                "type": "cm_matrix",
                "dataset": dataset,
                "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
                "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
            }
            f.write(json.dumps(cm_data) + "\n")

    print(f"Métricas guardadas en {file_path}")


print("Construcción del pipeline...")
pipeline = build_pipeline()

print("Optimización del modelo...")
best_pipeline = optimize_pipeline(pipeline, x_train, y_train)

print("Guardando el modelo...")
save_model(best_pipeline)

print("Evaluando el modelo y guardando métricas...")
evaluate_model(best_pipeline, x_train, y_train, x_test, y_test)

print("¡Proceso completado con éxito!")
