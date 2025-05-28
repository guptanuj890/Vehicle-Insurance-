import os
import sys
import json
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from sklearn.metrics import f1_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact
)
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
from shutil import copy


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys)

    def _map_gender_column(self, df):
        logging.info("Mapping 'Gender' column to binary values")
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        return df

    def _create_dummy_columns(self, df):
        logging.info("Creating dummy variables for categorical features")
        return pd.get_dummies(df, drop_first=True)

    def _rename_columns(self, df):
        logging.info("Renaming specific columns and casting to int")
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype(int)
        return df

    def _drop_id_column(self, df):
        logging.info("Dropping 'id' column if it exists")
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        return df

    def get_best_model(self) -> Optional[object]:
        try:
            best_model_path = os.path.join("artifact", "model_trainer", "best_model", "best_model.pkl")
            if os.path.exists(best_model_path):
                logging.info(f"Loading best model from path: {best_model_path}")
                return load_object(best_model_path)
            logging.info("No existing best model found.")
            return None
        except Exception as e:
            raise MyException(e, sys)

    def _save_evaluation_metrics(self, evaluation_response: EvaluateModelResponse):
        try:
            eval_metrics_dir = os.path.join("artifact", "evaluation_metrics")
            os.makedirs(eval_metrics_dir, exist_ok=True)
            eval_metrics_file_path = os.path.join(eval_metrics_dir, "evaluation_report.json")

            with open(eval_metrics_file_path, "w") as f:
                json.dump(evaluation_response.__dict__, f, indent=4)

            logging.info(f"Saved evaluation metrics to {eval_metrics_file_path}")
        except Exception as e:
            raise MyException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            x = self._map_gender_column(x)
            x = self._drop_id_column(x)
            x = self._create_dummy_columns(x)
            x = self._rename_columns(x)

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score

            best_model = self.get_best_model()
            best_model_f1_score = None

            if best_model:
                y_pred_best = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_pred_best)

            tmp_best_score = 0 if best_model_f1_score is None else best_model_f1_score
            is_model_accepted = trained_model_f1_score > tmp_best_score

            evaluation_response = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=is_model_accepted,
                difference=trained_model_f1_score - tmp_best_score
            )

            return evaluation_response

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Initialized Model Evaluation Component.")
            evaluation_response = self.evaluate_model()

            self._save_evaluation_metrics(evaluation_response)

            # Promote model if accepted
            if evaluation_response.is_model_accepted:
                dest_dir = os.path.join("artifacts", "best_model")
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, "best_model.pkl")
                copy(self.model_trainer_artifact.trained_model_file_path, dest_path)
                logging.info(f"Promoted new trained model to best model at: {dest_path}")

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluation_response.is_model_accepted,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluation_response.difference,
                # s3_model_path=""  # Optional: leave blank or remove if unused
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise MyException(e, sys)
