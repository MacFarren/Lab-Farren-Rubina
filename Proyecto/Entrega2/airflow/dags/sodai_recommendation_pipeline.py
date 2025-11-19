"""SodAI recommendation pipeline DAG that relies on local artifacts.

The workflow follows these stages:
1. Extract and validate raw data files.
2. Build engineered features for the ranking model.
3. Detect data drift and branch into ``retrain`` or ``load existing`` paths.
4. Train the model with Optuna when drift is detected or no model exists.
5. Evaluate, interpret, and generate demo predictions from the model.
6. Persist pipeline metrics as JSON files inside ``/opt/airflow/models``.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.getenv("SODAI_DATA_PATH", "/opt/airflow/data"))
MODELS_PATH = Path(os.getenv("SODAI_MODELS_PATH", "/opt/airflow/models"))
PIPELINE_RUNS_PATH = MODELS_PATH / "pipeline_runs"
SCRIPTS_PATH = PROJECT_ROOT / "scripts"

if str(SCRIPTS_PATH) not in sys.path:
	sys.path.append(str(SCRIPTS_PATH))

from scripts.data_extraction import DataExtractor
from scripts.feature_engineering import FeatureEngineer
from scripts.drift_detection import DriftDetector
from scripts.model_training import ModelTrainer
from scripts.model_evaluation import ModelEvaluator
from scripts.interpretability import InterpretabilityAnalyzer
from scripts.prediction_generator import PredictionGenerator

PIPELINE_RUNS_PATH.mkdir(parents=True, exist_ok=True)


def extract_and_validate_data(**context: Any) -> Dict[str, Any]:
	extractor = DataExtractor(
		data_path=str(DATA_PATH),
		validation_rules={
			"transacciones": {
				"required_columns": [
					"customer_id",
					"product_id",
					"order_id",
					"purchase_date",
					"items",
				],
				"date_column": "purchase_date",
				"min_records": 100,
			},
			"clientes": {
				"required_columns": ["customer_id", "region_id", "zone_id"],
				"unique_key": "customer_id",
			},
			"productos": {
				"required_columns": ["product_id", "brand", "category", "segment"],
				"unique_key": "product_id",
			},
		},
	)

	result = extractor.extract_and_validate()
	return result


def create_features(**context: Any) -> Dict[str, Any]:
	engineer = FeatureEngineer(data_path=str(DATA_PATH))
	result = engineer.create_recommendation_features()
	return result


def detect_data_drift(**context: Any) -> str:
	ti = context["task_instance"]
	model_path = MODELS_PATH / "recommendation_model.pkl"

	if not model_path.exists():
		fallback = {
			"detection_timestamp": datetime.utcnow().isoformat(),
			"drift_detected": True,
			"drift_score": 1.0,
			"reason": "model_missing",
		}
		ti.xcom_push(key="drift_results", value=fallback)
		return "retrain_model"

	detector = DriftDetector(
		reference_data_path=str(DATA_PATH / "features" / "reference_features.parquet"),
		current_data_path=str(DATA_PATH / "features" / "current_features.parquet"),
		drift_threshold=float(Variable.get("DRIFT_THRESHOLD", default_var="0.1")),
	)

	drift_results = detector.detect_drift()
	ti.xcom_push(key="drift_results", value=drift_results)
	return "retrain_model" if drift_results.get("drift_detected") else "load_existing_model"


def retrain_model(**context: Any) -> Dict[str, Any]:
	trials = int(Variable.get("OPTIMIZATION_TRIALS", default_var="10"))
	cv_folds = int(Variable.get("CV_FOLDS", default_var="3"))

	trainer = ModelTrainer(
		data_path=str(DATA_PATH),
		models_path=str(MODELS_PATH),
		optimization_trials=trials,
		cv_folds=cv_folds,
	)

	result = trainer.train_recommendation_model()
	return result


def load_existing_model(**context: Any) -> Dict[str, Any]:
	model_path = MODELS_PATH / "recommendation_model.pkl"
	metadata_path = MODELS_PATH / "model_metadata.json"

	if not model_path.exists():
		raise FileNotFoundError(f"Model not found at {model_path}")

	metadata: Dict[str, Any] = {}
	if metadata_path.exists():
		with metadata_path.open("r", encoding="utf-8") as handle:
			metadata = json.load(handle)

	result = {
		"model_uri": model_path.as_posix(),
		"retrained": False,
		"model_metadata": metadata,
		"feature_names": metadata.get("feature_names"),
		"training_timestamp": metadata.get("training_timestamp"),
	}
	return result


def evaluate_model(**context: Any) -> Dict[str, Any]:
	ti = context["task_instance"]
	training_results = ti.xcom_pull(task_ids="retrain_model")
	model_info = training_results or ti.xcom_pull(task_ids="load_existing_model")

	if not model_info:
		raise ValueError("Model information not found in XCom")

	feature_names = None
	if training_results:
		feature_names = training_results.get("feature_names")
	elif isinstance(model_info, dict):
		feature_names = model_info.get("feature_names") or None

	evaluator = ModelEvaluator(
		data_path=str(DATA_PATH),
		model_uri=model_info["model_uri"],
		feature_names=feature_names,
	)

	evaluation_results = evaluator.evaluate_recommendation_model()
	evaluation_results["model_uri"] = model_info["model_uri"]
	evaluation_results["is_retrained"] = bool(training_results)
	return evaluation_results


def analyze_interpretability(**context: Any) -> Dict[str, Any]:
	ti = context["task_instance"]
	training_results = ti.xcom_pull(task_ids="retrain_model")
	model_info = training_results or ti.xcom_pull(task_ids="load_existing_model")

	if not model_info:
		raise ValueError("Model information not available for interpretability analysis")

	analyzer = InterpretabilityAnalyzer(
		data_path=str(DATA_PATH),
		model_uri=model_info["model_uri"],
		models_path=str(MODELS_PATH),
	)

	shap_results = analyzer.generate_shap_analysis()
	shap_results["model_uri"] = model_info["model_uri"]
	return shap_results


def generate_predictions(**context: Any) -> Dict[str, Any]:
	ti = context["task_instance"]
	training_results = ti.xcom_pull(task_ids="retrain_model")
	model_info = training_results or ti.xcom_pull(task_ids="load_existing_model")

	if not model_info:
		raise ValueError("Model information not available for predictions")

	output_path = DATA_PATH / "predictions"

	generator = PredictionGenerator(
		data_path=str(DATA_PATH),
		model_uri=model_info["model_uri"],
		output_path=str(output_path),
	)

	prediction_results = generator.generate_weekly_predictions()
	prediction_results["model_uri"] = model_info["model_uri"]
	return prediction_results


def log_pipeline_metrics(**context: Any) -> Dict[str, Any]:
	ti = context["task_instance"]

	data_stats = ti.xcom_pull(task_ids="extract_and_validate_data")
	features_stats = ti.xcom_pull(task_ids="create_features")
	drift_results = ti.xcom_pull(task_ids="detect_data_drift", key="drift_results")
	training_results = ti.xcom_pull(task_ids="retrain_model")
	model_info = ti.xcom_pull(task_ids="load_existing_model")
	evaluation_results = ti.xcom_pull(task_ids="evaluate_model")
	shap_results = ti.xcom_pull(task_ids="analyze_interpretability")
	prediction_results = ti.xcom_pull(task_ids="generate_predictions")

	run_payload = {
		"run_id": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
		"data_stats": data_stats,
		"features_stats": features_stats,
		"drift_results": drift_results,
		"training_results": training_results,
		"model_info": model_info,
		"evaluation_results": evaluation_results,
		"interpretability": shap_results,
		"prediction_results": prediction_results,
		"logged_at": datetime.utcnow().isoformat(),
	}

	PIPELINE_RUNS_PATH.mkdir(parents=True, exist_ok=True)
	output_file = PIPELINE_RUNS_PATH / f"pipeline_run_{run_payload['run_id']}.json"
	with output_file.open("w", encoding="utf-8") as handle:
		json.dump(run_payload, handle, indent=2, default=str)

	return {"metrics_file": output_file.as_posix()}


default_args = {
	"owner": "sodai-mlops-team",
	"depends_on_past": False,
	"start_date": datetime(2025, 1, 1),
	"email": ["mlops@sodai-drinks.com"],
	"email_on_failure": True,
	"email_on_retry": False,
	"retries": 1,
	"retry_delay": timedelta(minutes=10),
}


dag = DAG(
	dag_id="sodai_recommendation_pipeline",
	default_args=default_args,
	description="Weekly MLOps pipeline for SodAI recommendation system",
	schedule_interval="@weekly",
	max_active_runs=1,
	catchup=False,
	tags=["sodai", "recommendation", "local-artifacts"],
	doc_md=__doc__,
)


start_task = DummyOperator(task_id="start_pipeline", dag=dag)

extract_data_task = PythonOperator(
	task_id="extract_and_validate_data",
	python_callable=extract_and_validate_data,
	dag=dag,
)

create_features_task = PythonOperator(
	task_id="create_features",
	python_callable=create_features,
	dag=dag,
)

drift_detection_task = BranchPythonOperator(
	task_id="detect_data_drift",
	python_callable=detect_data_drift,
	dag=dag,
)

retrain_model_task = PythonOperator(
	task_id="retrain_model",
	python_callable=retrain_model,
	dag=dag,
)

load_model_task = PythonOperator(
	task_id="load_existing_model",
	python_callable=load_existing_model,
	dag=dag,
)

join_task = DummyOperator(
	task_id="join_model_paths",
	dag=dag,
	trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
)

evaluate_model_task = PythonOperator(
	task_id="evaluate_model",
	python_callable=evaluate_model,
	dag=dag,
)

interpretability_task = PythonOperator(
	task_id="analyze_interpretability",
	python_callable=analyze_interpretability,
	dag=dag,
)

predictions_task = PythonOperator(
	task_id="generate_predictions",
	python_callable=generate_predictions,
	dag=dag,
)

log_metrics_task = PythonOperator(
	task_id="log_pipeline_metrics",
	python_callable=log_pipeline_metrics,
	dag=dag,
)

end_task = DummyOperator(task_id="pipeline_complete", dag=dag)


start_task >> extract_data_task >> create_features_task >> drift_detection_task
drift_detection_task >> retrain_model_task
drift_detection_task >> load_model_task
[retrain_model_task, load_model_task] >> join_task
join_task >> evaluate_model_task >> interpretability_task >> predictions_task
predictions_task >> log_metrics_task >> end_task


def setup_airflow_variables() -> None:
	required_defaults = {
		"SODAI_DATA_PATH": str(DATA_PATH),
		"SODAI_MODELS_PATH": str(MODELS_PATH),
		"DRIFT_THRESHOLD": "0.1",
		"OPTIMIZATION_TRIALS": "10",
		"CV_FOLDS": "3",
	}

	for key, value in required_defaults.items():
		try:
			Variable.get(key)
		except KeyError:
			Variable.set(key, value)


if __name__ != "__main__":
	try:
		setup_airflow_variables()
	except Exception as exc:  # pragma: no cover
		print(f"Failed to ensure Airflow variables: {exc}")

