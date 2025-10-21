#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
optimize.py
================================================================================
Propósito
---------
Entrenar y **optimizar** un modelo de clasificación **XGBoost** para el dataset
de potabilidad de agua, **registrando todo el proceso en MLflow** y dejando
artefactos y resultados en carpetas estándar del proyecto.

Este archivo cumple *textualmente* los puntos del enunciado y trae encabezados
`###` colocados EXACTAMENTE en el lugar donde se realiza cada acción pedida.

Qué hace (visión ejecutiva)
---------------------------
1) Carga ``water_potability.csv`` y separa **train/valid/test**.
2) Crea un **experimento nuevo** en MLflow con nombre interpretable (no "Default").
3) Usa **Optuna** para buscar hiperparámetros de **XGBoost**:
   - Cada *trial* (intento) se ejecuta como **run anidado** en MLflow,
     guardando (a) parámetros, (b) métrica **valid_f1**, (c) el modelo del trial.
4) Genera **gráficos** de Optuna y **feature importance** y los guarda en ``/plots``.
5) Selecciona el **mejor run por valid_f1**, **carga su modelo** y lo serializa
   como **``models/best_model.pkl``** (pipeline listo para `predict`).
6) Registra **versiones de librerías** y **reportes/configs** como artefactos.

Salidas esperadas (carpetas)
----------------------------
- ``mlruns/``  → lo crea **MLflow** automáticamente (historial de experimentos/runs).
- ``plots/``   → figuras de Optuna & feature importance (también subidas a MLflow).
- ``models/``  → **best_model.pkl** (pipeline imputador + XGBoost ganador).

Entradas esperadas
------------------
- ``water_potability.csv`` con columna **Potability** (0/1) y features:
  ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon,
  Trihalomethanes, Turbidity.

Cómo ejecutarlo
---------------
Entrenar/optimizar desde terminal/VS Code:

    python optimize.py --csv-path water_potability.csv --n-trials 40 \
        --test-size 0.2 --val-size 0.2 --seed 42

Ver UI de MLflow (opcional):

    mlflow ui
    # abre http://127.0.0.1:5000

Notas de reproducibilidad
-------------------------
- Guardamos ``env/versions.json`` como artefacto del run para rastrear versiones.
- El mejor pipeline se guarda con ``pickle`` para **serving** (FastAPI usará ese archivo).
- Los nombres de experimento/run son **interpretables** para inspección.

Seguridad y errores comunes
---------------------------
- Si el CSV no contiene ``Potability`` se lanzará un error claro.
- Si no hay runs (ej. Optuna no ejecutó nada), se explicita el problema.
- Si faltan librerías, usa `pip install -r requirements.txt`.

================================================================================
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

import optuna
from optuna.visualization.matplotlib import (
    plot_param_importances,
    plot_optimization_history,
)

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# Estructuras y utilidades de apoyo (pequeñas, puramente utilitarias)
# ------------------------------------------------------------------------------

@dataclass
class Splits:
    X_tr: pd.DataFrame
    X_val: pd.DataFrame
    X_te: pd.DataFrame
    y_tr: pd.Series
    y_val: pd.Series
    y_te: pd.Series


def _ensure_dirs() -> None:
    """
    Garantiza que existan las carpetas locales donde guardaremos artefactos
    fuera del store de MLflow (además de subirlos como artifacts).
    """
    Path("plots").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)


def _make_experiment_name(base: str = "xgboost_optuna_waterpotability") -> str:
    """
    Genera un nombre interpretable con timestamp para el **experimento** en MLflow.
    Ej.: xgboost_optuna_waterpotability_20251021-012233
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{base}_{ts}"


def _load_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carga el CSV y devuelve (X, y). Lanza error informativo si falta 'Potability'.
    """
    df = pd.read_csv(csv_path)
    if "Potability" not in df.columns:
        missing = "Potability"
        raise ValueError(
            f"El CSV no contiene la columna '{missing}'. "
            f"Columnas detectadas: {list(df.columns)}"
        )
    y = df["Potability"].astype(int)
    X = df.drop(columns=["Potability"])
    return X, y


def _split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    seed: int,
) -> Splits:
    """
    Realiza un split estable y estratificado en: train / valid / test.
    val_size y test_size son proporciones relativas al total.
    """
    X_tr, X_temp, y_tr, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=seed, stratify=y
    )
    rel_val = val_size / (test_size + val_size)
    X_val, X_te, y_val, y_te = train_test_split(
        X_temp, y_temp, test_size=(1 - rel_val), random_state=seed, stratify=y_temp
    )
    return Splits(X_tr, X_val, X_te, y_tr, y_val, y_te)


# ### Guardar las versiones de las librerías utilizadas en el desarrollo.
def _log_library_versions() -> None:
    """
    Deja constancia de versiones (Python y libs) como artefacto del run.
    También añade tags para facilitar búsquedas en la UI de MLflow.
    """
    import sys
    import sklearn
    import xgboost
    import optuna as _optuna
    import mlflow as _mlflow

    versions = {
        "python": sys.version.replace("\n", " "),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "optuna": _optuna.__version__,
        "mlflow": _mlflow.__version__,
    }
    mlflow.log_dict(versions, "env/versions.json")
    for k, v in versions.items():
        mlflow.set_tag(f"lib.{k}", v)


def get_best_model(experiment_id: str):
    """
    Devuelve el **pipeline** ganador (imputador + XGBoost) cargado desde MLflow,
    seleccionándolo por **mayor `metrics.valid_f1`**.

    NOTA: Se usa `mlflow.sklearn.load_model(...)` porque el pipeline se registró
    como artifact sklearn en cada trial.
    """
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    if runs.empty:
        raise RuntimeError(
            "No hay runs en el experimento. ¿Se ejecutó Optuna/study.optimize?"
        )
    best_model_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model(f"runs:/{best_model_id}/model")
    return best_model


# ### Optimizar los hiperparámetros del modelo `XGBoost` usando `Optuna`.
def _build_objective(X_train, y_train, X_val, y_val, seed: int):
    """
    Devuelve la función objetivo para Optuna. Cada trial:
      1) Construye pipeline (imputer + XGBClassifier con parámetros del trial).
      2) Entrena en train y evalúa en valid.
      3) **Registra** parámetros, métrica 'valid_f1' y el modelo del trial en MLflow.
    """
    def objective(trial: optuna.Trial) -> float:
        # Espacio de búsqueda sensato para un XGBClassifier tabular
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": seed,
            "n_jobs": -1,
            "tree_method": "hist",
            "eval_metric": "logloss",
        }

        pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(**params)),
        ])

        # ### Registrar cada entrenamiento en un experimento nuevo, asegurándose
        # ### de que la métrica f1-score se registre como "valid_f1".
        # - Cada trial es un **run anidado** con nombre interpretable.
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.log_params(params)

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            f1 = f1_score(y_val, y_pred, average="binary")

            mlflow.log_metric("valid_f1", f1)
            # Guardamos el pipeline entrenado de este trial como artifact
            mlflow.sklearn.log_model(pipe, artifact_path="model")

            return f1

    return objective


def _save_and_log_optuna_plots(study: optuna.Study) -> None:
    _ensure_dirs()
    # ### Guardar los gráficos de Optuna dentro de una carpeta de artefactos
    # ### de Mlflow llamada `/plots`.
    fig1 = plot_optimization_history(study)
    fig1.figure.tight_layout()
    fig1.figure.savefig("plots/optuna_optimization_history.png", dpi=150)
    plt.close(fig1.figure)

    fig2 = plot_param_importances(study)
    fig2.figure.tight_layout()
    fig2.figure.savefig("plots/optuna_param_importances.png", dpi=150)
    plt.close(fig2.figure)

    # Subimos la carpeta completa como artifacts del run padre
    mlflow.log_artifacts("plots", artifact_path="plots")


def _train_and_log_best(
    splits: Splits,
    study: optuna.Study,
    experiment_id: str,
    seed: int,
) -> None:
    """
    - Carga el mejor pipeline con `get_best_model`.
    - Evalúa en valid/test.
    - Registra reportes/configs.
    - Genera gráfico de **feature importance** y lo guarda en `/plots`.
    - **Serializa** el pipeline ganador a `/models/best_model.pkl`.
    """
    # ### Devolver el mejor modelo usando la función `get_best_model` y
    # ### serializarlo en el disco con `pickle.dump`. Luego, guardar el modelo
    # ### en la carpeta `/models`.
    best_model = get_best_model(experiment_id)

    y_val_pred = best_model.predict(splits.X_val)
    y_te_pred = best_model.predict(splits.X_te)

    valid_f1 = f1_score(splits.y_val, y_val_pred, average="binary")
    test_f1 = f1_score(splits.y_te, y_te_pred, average="binary")

    # ### Respalde las configuraciones del modelo final y la importancia de las
    # ### variables en un gráfico dentro de la carpeta `/plots` creada anteriormente.
    report = {
        "validation_f1": float(valid_f1),
        "test_f1": float(test_f1),
        "validation_report": classification_report(splits.y_val, y_val_pred, output_dict=True),
        "test_report": classification_report(splits.y_te, y_te_pred, output_dict=True),
        "best_params_from_study": study.best_params,
        "best_value_from_study": float(study.best_value),
        "random_state": seed,
    }
    mlflow.log_dict(report, "reports/summary.json")
    mlflow.log_text(json.dumps(study.best_params, indent=2), "configs/best_params.json")

    # Feature importance (del XGB del pipeline)
    try:
        model = best_model.named_steps["model"]
        importances = model.feature_importances_
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(len(importances)), importances)
        ax.set_title("Feature Importances (XGBoost)")
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Importance")
        fig.tight_layout()
        fpath = "plots/feature_importances.png"
        fig.savefig(fpath, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(fpath, artifact_path="plots")
    except Exception as e:
        # Si por cualquier motivo falla (p.ej., modelo sin atributo), deja constancia.
        mlflow.log_text(str(e), "reports/feature_importances_error.txt")

    _ensure_dirs()
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    mlflow.log_artifact("models/best_model.pkl", artifact_path="models")


# ### Guardar el código en `optimize.py`. La ejecución de `python optimize.py`
# ### debería ejecutar la función `optimize_model`.
def optimize_model(
    csv_path: str = "water_potability.csv",
    n_trials: int = 40,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
):
    """
    Función orquestadora que **cumple el enunciado punto por punto**.

    Parámetros:
    - csv_path: ruta al CSV con columna 'Potability' (0/1) y features numéricas.
    - n_trials: cuántos intentos probará Optuna.
    - test_size, val_size: proporciones globales para test y validación.
    - seed: aleatoriedad controlada para reproducibilidad.

    Retorna:
    - El **pipeline ganador** (imputer + XGBClassifier) ya entrenado.
    """
    _ensure_dirs()

    # 1) Datos
    X, y = _load_dataset(csv_path)
    splits = _split_data(X, y, test_size=test_size, val_size=val_size, seed=seed)

    # 2) Experimento MLflow (NO usar "Default")
    # ### Registrar cada entrenamiento en un experimento nuevo, con nombres interpretables.
    experiment_name = _make_experiment_name()
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    assert experiment is not None  # por tipado y tranquilidad

    # 3) Estudio Optuna
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Run padre (engloba todos los trials como runs anidados)
    with mlflow.start_run(run_name="optuna_study_parent"):
        # Tags informativos (se ven en la UI)
        mlflow.set_tag("problem", "water_potability_binary_classification")
        mlflow.set_tag("framework", "xgboost")
        mlflow.set_tag("optimizer", "optuna")

        # ### Guardar las versiones de las librerías utilizadas en el desarrollo.
        _log_library_versions()

        # ### Optimizar los hiperparámetros del modelo `XGBoost` usando `Optuna`.
        objective = _build_objective(
            splits.X_tr, splits.y_tr, splits.X_val, splits.y_val, seed
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # ### Guardar los gráficos de Optuna dentro de `/plots`.
        _save_and_log_optuna_plots(study)

        # ### Devolver el mejor modelo, serializarlo y guardarlo en `/models`.
        _train_and_log_best(
            splits=splits,
            study=study,
            experiment_id=experiment.experiment_id,
            seed=seed,
        )

        # Retorno de conveniencia
        return get_best_model(experiment.experiment_id)


# ------------------------------------------------------------------------------
# CLI para que correr `python optimize.py` funcione de inmediato
# ------------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Optuna + MLflow + XGBoost (Water Potability).")
    p.add_argument("--csv-path", type=str, default="water_potability.csv")
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _ = optimize_model(
        csv_path=args.csv_path,
        n_trials=args.n_trials,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    print("✔ Mejor modelo serializado en models/best_model.pkl")
