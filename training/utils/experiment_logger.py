# =========================================================
# experiment_logger.py
# ---------------------------------------------------------
# 팀 프로젝트에서 모델 실험 결과를 일관된 형식으로 저장하기 위한 유틸 파일
#
# [주요 기능]
# 1. 실행(run)마다 고유 ID 생성
# 2. 모든 실험 결과를 experiment_log.csv에 누적 저장
# 3. 모델별 최고 성능만 best_results.csv에 별도 저장
# 4. 분류 리포트(txt), confusion matrix(csv), 하이퍼파라미터(json) 저장
# 5. 회귀 / 분류 모두 지원
#
# [권장 사용 방식]
# - 모델 학습/평가가 끝난 뒤 이 logger를 호출
# - 팀원마다 runner 이름을 넣어 누가 돌린 실험인지 기록
# - model_name, model_version, data_version, feature_version 등을 함께 저장
# - 나중에 실험 비교, 재현, 발표 자료 정리에 활용
# =========================================================

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class ExperimentLogger:
    """
    실험 결과 저장 전용 클래스

    이 클래스는 프로젝트 내에서 모델 결과를 체계적으로 저장하기 위해 사용한다.
    팀원 여러 명이 같은 모델/다른 모델을 여러 번 돌릴 수 있으므로,
    모든 실행을 누적 기록하면서도 최고 성능은 별도로 관리한다.
    """

    def __init__(self, artifact_dir: str):
        """
        Parameters
        ----------
        artifact_dir : str
            실험 결과를 저장할 최상위 폴더 경로
            예: training/artifacts
        """
        self.artifact_dir = artifact_dir

        # 하위 폴더 구조 생성
        self.logs_dir = os.path.join(self.artifact_dir, "logs")
        self.reports_dir = os.path.join(self.artifact_dir, "reports")
        self.confusion_dir = os.path.join(self.artifact_dir, "confusion_matrices")
        self.params_dir = os.path.join(self.artifact_dir, "params")
        self.summaries_dir = os.path.join(self.artifact_dir, "summaries")

        for path in [
            self.artifact_dir,
            self.logs_dir,
            self.reports_dir,
            self.confusion_dir,
            self.params_dir,
            self.summaries_dir,
        ]:
            os.makedirs(path, exist_ok=True)

        # 전체 실행 이력 저장 파일
        self.experiment_log_path = os.path.join(self.logs_dir, "experiment_log.csv")

        # 최고 결과만 저장하는 파일
        self.best_results_path = os.path.join(self.summaries_dir, "best_results.csv")

    # =========================================================
    # 1. 실행 ID 생성
    # =========================================================
    def generate_run_id(self, runner: str, model_name: str) -> str:
        """
        실행마다 고유한 run_id를 생성한다.

        예:
        eunbyeol_lgbm_peak_cls_20260415_173012
        """
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_runner = str(runner).replace(" ", "_")
        safe_model = str(model_name).replace(" ", "_")
        return f"{safe_runner}_{safe_model}_{now_str}"

    # =========================================================
    # 2. JSON 저장용 보조 함수
    # =========================================================
    @staticmethod
    def _safe_json_dumps(data: Dict[str, Any]) -> str:
        """
        dict를 json 문자열로 안전하게 변환한다.
        csv 한 칸에 저장할 때 사용한다.
        """
        try:
            return json.dumps(data, ensure_ascii=False, sort_keys=True)
        except Exception:
            safe_data = {k: str(v) for k, v in data.items()}
            return json.dumps(safe_data, ensure_ascii=False, sort_keys=True)

    # =========================================================
    # 3. 파일 저장 공통 함수
    # =========================================================
    def _save_json(self, data: Dict[str, Any], save_path: str) -> None:
        """
        dict 형태 데이터를 json 파일로 저장한다.
        """
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _append_csv_row(self, row_dict: Dict[str, Any], csv_path: str) -> None:
        """
        csv 파일에 한 줄(row)을 누적 저장한다.
        dtype 충돌을 피하기 위해 기존 파일도 문자열 기반으로 읽는다.
        """
        safe_row_dict = {k: ("" if v is None else v) for k, v in row_dict.items()}
        new_df = pd.DataFrame([safe_row_dict]).astype(object)

        if os.path.exists(csv_path):
            old_df = pd.read_csv(
                csv_path,
                encoding="utf-8-sig",
                dtype=str,
                keep_default_na=False
            ).astype(object)

            # 새 row에만 있는 컬럼 추가
            for col in new_df.columns:
                if col not in old_df.columns:
                    old_df[col] = ""

            # 기존 csv에만 있는 컬럼 추가
            for col in old_df.columns:
                if col not in new_df.columns:
                    new_df[col] = ""

            # 컬럼 순서 맞추기
            new_df = new_df[old_df.columns]

            merged_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            merged_df = new_df

        merged_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # =========================================================
    # 4. best 결과 판별 규칙
    # =========================================================
    @staticmethod
    def _is_better_result(task_type: str, new_row: Dict[str, Any], old_row: pd.Series) -> bool:
        """
        새 결과가 기존 best보다 더 좋은지 판단한다.

        [분류 기준]
        1. macro_f1 높을수록 좋음
        2. weighted_f1 높을수록 좋음
        3. acc 높을수록 좋음

        [회귀 기준]
        1. mae 낮을수록 좋음
        2. rmse 낮을수록 좋음
        3. r2 높을수록 좋음
        """
        if task_type == "classification":
            new_key = (
                float(new_row.get("macro_f1", -1)),
                float(new_row.get("weighted_f1", -1)),
                float(new_row.get("acc", -1)),
            )
            old_key = (
                float(old_row.get("macro_f1", -1)),
                float(old_row.get("weighted_f1", -1)),
                float(old_row.get("acc", -1)),
            )
            return new_key > old_key

        if task_type == "regression":
            new_key = (
                -float(new_row.get("mae", 1e18)),
                -float(new_row.get("rmse", 1e18)),
                float(new_row.get("r2", -1e18)),
            )
            old_key = (
                -float(old_row.get("mae", 1e18)),
                -float(old_row.get("rmse", 1e18)),
                float(old_row.get("r2", -1e18)),
            )
            return new_key > old_key

        return False

    def _update_best_results(self, row_dict: Dict[str, Any]) -> None:
        """
        모델별 최고 성능(best result)을 저장/갱신한다.

        best를 구분할 때는 아래 조합을 기준으로 묶는다.
        - task_type
        - model_name
        - model_version
        - data_version
        - feature_version
        - label_definition_name
        """
        key_cols = [
            "task_type",
            "model_name",
            "model_version",
            "data_version",
            "feature_version",
            "label_definition_name",
        ]

        safe_row_dict = {k: ("" if v is None else v) for k, v in row_dict.items()}
        row_df = pd.DataFrame([safe_row_dict]).astype(object)

        # best_results.csv가 없으면 바로 생성
        if not os.path.exists(self.best_results_path):
            row_df.to_csv(self.best_results_path, index=False, encoding="utf-8-sig")
            return

        # 기존 best 파일은 전부 문자열 기반으로 안전하게 읽기
        best_df = pd.read_csv(
            self.best_results_path,
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False
        ).astype(object)

        # 새 row에 있는 컬럼이 기존 best_df에 없으면 추가
        for col in row_df.columns:
            if col not in best_df.columns:
                best_df[col] = ""

        # 기존 best_df에만 있는 컬럼이 있으면 row_df에도 추가
        for col in best_df.columns:
            if col not in row_df.columns:
                row_df[col] = ""

        row_df = row_df[best_df.columns]

        # key가 같은 기존 best 행 찾기
        mask = pd.Series([True] * len(best_df))
        for col in key_cols:
            left = best_df[col].fillna("").astype(str)
            right = str(safe_row_dict.get(col, ""))
            mask &= (left == right)

        matched_idx = best_df.index[mask]

        if len(matched_idx) == 0:
            # 같은 key 조합이 없으면 새로 추가
            best_df = pd.concat([best_df, row_df], ignore_index=True)
        else:
            # 기존 best와 비교해서 더 좋으면 교체
            idx = matched_idx[0]
            old_row = best_df.loc[idx]

            if self._is_better_result(safe_row_dict["task_type"], safe_row_dict, old_row):
                for col, value in safe_row_dict.items():
                    if col not in best_df.columns:
                        best_df[col] = ""
                    best_df[col] = best_df[col].astype(object)
                    best_df.loc[idx, col] = "" if value is None else value

        best_df.to_csv(self.best_results_path, index=False, encoding="utf-8-sig")

    # =========================================================
    # 5. 공통 메타 정보 정리
    # =========================================================
    def _build_base_row(
        self,
        run_id: str,
        runner: str,
        task_type: str,
        model_name: str,
        model_version: str,
        dataset_name: str,
        data_version: str,
        split_version: str,
        feature_version: str,
        label_definition_name: Optional[str],
        label_definition_detail: Optional[Dict[str, Any]],
        hyperparams: Optional[Dict[str, Any]],
        notes: Optional[str],
        report_txt_path: Optional[str] = None,
        confusion_csv_path: Optional[str] = None,
        params_json_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        회귀/분류 공통으로 저장할 메타 정보를 하나의 dict로 정리한다.
        """
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_id,
            "runner": runner,
            "task_type": task_type,
            "model_name": model_name,
            "model_version": model_version,
            "dataset_name": dataset_name,
            "data_version": data_version,
            "split_version": split_version,
            "feature_version": feature_version,
            "label_definition_name": label_definition_name if label_definition_name else "",
            "label_definition_detail": self._safe_json_dumps(label_definition_detail) if label_definition_detail else "",
            "hyperparams": self._safe_json_dumps(hyperparams) if hyperparams else "",
            "notes": notes if notes else "",
            "report_txt_path": report_txt_path if report_txt_path else "",
            "confusion_csv_path": confusion_csv_path if confusion_csv_path else "",
            "params_json_path": params_json_path if params_json_path else "",
        }
        return row

    # =========================================================
    # 6. 분류 결과 저장
    # =========================================================
    def log_classification_result(
        self,
        y_true,
        y_pred,
        runner: str,
        model_name: str,
        model_version: str = "v1",
        dataset_name: str = "",
        data_version: str = "",
        split_version: str = "",
        feature_version: str = "",
        label_definition_name: str = "",
        label_definition_detail: Optional[Dict[str, Any]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        class_labels: Optional[List[str]] = None,
        notes: str = "",
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        분류 모델 결과를 저장한다.

        저장 내용:
        - 분류 metric 계산
        - labeled confusion matrix csv 저장
        - classification report txt 저장
        - hyperparams json 저장
        - experiment_log.csv 누적
        - best_results.csv 갱신
        """
        if run_id is None:
            run_id = self.generate_run_id(runner, model_name)

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))

        if class_labels is None:
            class_labels = [str(c) for c in unique_classes]

        if len(class_labels) != len(unique_classes):
            raise ValueError("class_labels 길이는 실제 클래스 개수와 같아야 합니다.")

        # metric 계산
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        weighted_f1 = f1_score(y_true, y_pred, average="weighted")

        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

        report_dict = classification_report(
            y_true,
            y_pred,
            labels=unique_classes,
            target_names=class_labels,
            digits=4,
            output_dict=True,
            zero_division=0
        )

        report_text = classification_report(
            y_true,
            y_pred,
            labels=unique_classes,
            target_names=class_labels,
            digits=4,
            zero_division=0
        )

        # confusion matrix를 라벨 붙은 DataFrame으로 정리
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{label}" for label in class_labels],
            columns=[f"pred_{label}" for label in class_labels]
        )

        report_txt_path = os.path.join(self.reports_dir, f"{run_id}_classification_report.txt")
        confusion_csv_path = os.path.join(self.confusion_dir, f"{run_id}_confusion_matrix.csv")
        params_json_path = os.path.join(self.params_dir, f"{run_id}_hyperparams.json")

        # confusion matrix csv 저장
        cm_df.to_csv(confusion_csv_path, encoding="utf-8-sig")

        # 하이퍼파라미터 json 저장
        self._save_json(hyperparams if hyperparams else {}, params_json_path)

        # txt 리포트 저장
        with open(report_txt_path, "w", encoding="utf-8") as f:
            f.write("[Classification Result]\n")
            f.write(f"run_id               : {run_id}\n")
            f.write(f"runner               : {runner}\n")
            f.write(f"model_name           : {model_name}\n")
            f.write(f"model_version        : {model_version}\n")
            f.write(f"dataset_name         : {dataset_name}\n")
            f.write(f"data_version         : {data_version}\n")
            f.write(f"split_version        : {split_version}\n")
            f.write(f"feature_version      : {feature_version}\n")
            f.write(f"label_definition     : {label_definition_name}\n")
            f.write(f"ACC                  : {acc:.4f}\n")
            f.write(f"Macro F1             : {macro_f1:.4f}\n")
            f.write(f"Weighted F1          : {weighted_f1:.4f}\n\n")

            f.write("[Label Definition Detail]\n")
            if label_definition_detail:
                f.write(json.dumps(label_definition_detail, ensure_ascii=False, indent=2))
            else:
                f.write("N/A")
            f.write("\n\n")

            f.write("[Confusion Matrix - labeled]\n")
            f.write(cm_df.to_string())
            f.write("\n\n")

            f.write("[Classification Report]\n")
            f.write(report_text)
            f.write("\n")

            f.write("[Hyperparameters]\n")
            if hyperparams:
                f.write(json.dumps(hyperparams, ensure_ascii=False, indent=2))
            else:
                f.write("{}")
            f.write("\n\n")

            f.write("[Notes]\n")
            f.write(notes if notes else "")
            f.write("\n")

        row_dict = self._build_base_row(
            run_id=run_id,
            runner=runner,
            task_type="classification",
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            data_version=data_version,
            split_version=split_version,
            feature_version=feature_version,
            label_definition_name=label_definition_name,
            label_definition_detail=label_definition_detail,
            hyperparams=hyperparams,
            notes=notes,
            report_txt_path=report_txt_path,
            confusion_csv_path=confusion_csv_path,
            params_json_path=params_json_path,
        )

        row_dict["acc"] = round(acc, 4)
        row_dict["macro_f1"] = round(macro_f1, 4)
        row_dict["weighted_f1"] = round(weighted_f1, 4)

        # 클래스별 metric 저장
        for label in class_labels:
            row_dict[f"{label}_precision"] = round(report_dict[label]["precision"], 4)
            row_dict[f"{label}_recall"] = round(report_dict[label]["recall"], 4)
            row_dict[f"{label}_f1"] = round(report_dict[label]["f1-score"], 4)
            row_dict[f"{label}_support"] = int(report_dict[label]["support"])

        # 전체 로그 저장
        self._append_csv_row(row_dict, self.experiment_log_path)

        # best 결과 갱신
        self._update_best_results(row_dict)

        return {
            "run_id": run_id,
            "acc": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "report_txt_path": report_txt_path,
            "confusion_csv_path": confusion_csv_path,
            "params_json_path": params_json_path,
        }

    # =========================================================
    # 7. 회귀 결과 저장
    # =========================================================
    def log_regression_result(
        self,
        y_true,
        y_pred,
        runner: str,
        model_name: str,
        model_version: str = "v1",
        dataset_name: str = "",
        data_version: str = "",
        split_version: str = "",
        feature_version: str = "",
        hyperparams: Optional[Dict[str, Any]] = None,
        notes: str = "",
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        회귀 모델 결과를 저장한다.

        저장 내용:
        - 회귀 metric 계산
        - txt 리포트 저장
        - hyperparams json 저장
        - experiment_log.csv 누적
        - best_results.csv 갱신
        """
        if run_id is None:
            run_id = self.generate_run_id(runner, model_name)

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        report_txt_path = os.path.join(self.reports_dir, f"{run_id}_regression_report.txt")
        params_json_path = os.path.join(self.params_dir, f"{run_id}_hyperparams.json")

        # 하이퍼파라미터 저장
        self._save_json(hyperparams if hyperparams else {}, params_json_path)

        # txt 저장
        with open(report_txt_path, "w", encoding="utf-8") as f:
            f.write("[Regression Result]\n")
            f.write(f"run_id               : {run_id}\n")
            f.write(f"runner               : {runner}\n")
            f.write(f"model_name           : {model_name}\n")
            f.write(f"model_version        : {model_version}\n")
            f.write(f"dataset_name         : {dataset_name}\n")
            f.write(f"data_version         : {data_version}\n")
            f.write(f"split_version        : {split_version}\n")
            f.write(f"feature_version      : {feature_version}\n")
            f.write(f"MAE                  : {mae:.4f}\n")
            f.write(f"RMSE                 : {rmse:.4f}\n")
            f.write(f"R2                   : {r2:.4f}\n\n")

            f.write("[Hyperparameters]\n")
            if hyperparams:
                f.write(json.dumps(hyperparams, ensure_ascii=False, indent=2))
            else:
                f.write("{}")
            f.write("\n\n")

            f.write("[Notes]\n")
            f.write(notes if notes else "")
            f.write("\n")

        row_dict = self._build_base_row(
            run_id=run_id,
            runner=runner,
            task_type="regression",
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            data_version=data_version,
            split_version=split_version,
            feature_version=feature_version,
            label_definition_name="",
            label_definition_detail=None,
            hyperparams=hyperparams,
            notes=notes,
            report_txt_path=report_txt_path,
            confusion_csv_path="",
            params_json_path=params_json_path,
        )

        row_dict["mae"] = round(mae, 4)
        row_dict["rmse"] = round(rmse, 4)
        row_dict["r2"] = round(r2, 4)

        # 전체 로그 저장
        self._append_csv_row(row_dict, self.experiment_log_path)

        # best 결과 갱신
        self._update_best_results(row_dict)

        return {
            "run_id": run_id,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "report_txt_path": report_txt_path,
            "params_json_path": params_json_path,
        }