import json
import os
import joblib


def save_model_artifacts(
    model_dir: str,
    reg_model,
    peak_congestion_model,
    full_model,
    feature_cols: list,
    peak_thresholds: list,
    full_binary_threshold: float,
    label_definition_detail: dict,
):
    os.makedirs(model_dir, exist_ok=True)

    # ---------- 모델 저장 ----------
    joblib.dump(reg_model, os.path.join(model_dir, "reg.pkl"))
    joblib.dump(peak_congestion_model, os.path.join(model_dir, "peak_congestion_cls.pkl"))
    joblib.dump(full_model, os.path.join(model_dir, "full_cls.pkl"))

    # ---------- feature 목록 저장 ----------
    with open(os.path.join(model_dir, "feature_cols.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    # ---------- threshold 저장 ----------
    with open(os.path.join(model_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "peak_congestion_thresholds": peak_thresholds,
                "full_binary_threshold": full_binary_threshold,
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    # ---------- 라벨 정의 저장 ----------
    with open(os.path.join(model_dir, "label_definition.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "peak_congestion_4class": label_definition_detail,
                "full_binary": {
                    "0": "여석있음",
                    "1": "만차",
                }
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n모델 저장 완료")
    print(f"- {os.path.join(model_dir, 'reg.pkl')}")
    print(f"- {os.path.join(model_dir, 'peak_congestion_cls.pkl')}")
    print(f"- {os.path.join(model_dir, 'full_cls.pkl')}")
    print(f"- {os.path.join(model_dir, 'feature_cols.json')}")
    print(f"- {os.path.join(model_dir, 'thresholds.json')}")
    print(f"- {os.path.join(model_dir, 'label_definition.json')}")