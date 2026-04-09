import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime
import glob

# =========================================================
# 1. 여러 파일 불러오기 (4~5일치)
# =========================================================
file_list = glob.glob("bus_data_2026_03_*.csv")  # 패턴으로 여러 파일 불러오기

df_list = []

for file in file_list:
    temp_df = pd.read_csv(
        file,
        dtype={
            "busRouteId": str,
            "stId": str,
            "arsId": str
        }
    )
    df_list.append(temp_df)

# 하나로 합치기
df = pd.concat(df_list, ignore_index=True)

# 시간 컬럼 변환
df["mkTm"] = pd.to_datetime(df["mkTm"])

# 시간순 정렬 (중요)
df = df.sort_values("mkTm").reset_index(drop=True)

print("총 데이터 수:", len(df))

# =========================================================
# 2. 필요한 컬럼 확인 및 정리
# =========================================================
# 실제 파일 기준 컬럼명
# ['mkTm', 'busRouteId', 'stId', 'arsId', 'staOrd', 'vehId1', 'exps1',
#  'arrmsg1', 'remaining_seat', 'full_flag', 'year', 'month', 'day',
#  'hour', 'minute', 'dayofweek', 'is_weekend', 'is_holiday', 'is_peak',
#  'month_sin', 'month_cos', 'day_sin', 'day_cos',
#  'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'dow_sin', 'dow_cos']

# 결측치 제거
df = df.dropna(subset=["mkTm", "busRouteId", "stId", "arsId", "remaining_seat"])

# 자료형 통일
df["busRouteId"] = df["busRouteId"].astype(str).str.strip()
df["stId"] = df["stId"].astype(str).str.strip()
df["arsId"] = df["arsId"].astype(str).str.strip()

# =========================================================
# 3. 인코딩 컬럼 생성
# =========================================================
route_le = LabelEncoder()
stid_le = LabelEncoder()
arsid_le = LabelEncoder()

df["route_enc"] = route_le.fit_transform(df["busRouteId"])
df["stid_enc"] = stid_le.fit_transform(df["stId"])
df["arsid_enc"] = arsid_le.fit_transform(df["arsId"])

# =========================================================
# 4. 미래 타깃(y) 생성
# =========================================================
HORIZON_MINUTES = 10

# 타입 안전하게 한 번 더 맞추기
df["busRouteId"] = df["busRouteId"].astype(str).str.strip()
df["stId"] = df["stId"].astype(str).str.strip()
df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")

# null 제거
df = df.dropna(subset=["mkTm", "busRouteId", "stId", "remaining_seat"]).copy()

# target_time 생성
df["target_time"] = df["mkTm"] + pd.Timedelta(minutes=HORIZON_MINUTES)

# 오른쪽용 데이터 준비
right_df = df[["busRouteId", "stId", "mkTm", "remaining_seat"]].copy()
right_df = right_df.rename(columns={
    "mkTm": "future_mkTm",
    "remaining_seat": "target_remaining_seat"
})

# merge_asof 직전 null 제거
left_df = df.dropna(subset=["target_time"]).copy()
right_df = right_df.dropna(subset=["future_mkTm"]).copy()

# ★ 중요: 시간키를 먼저 기준으로 정렬
left_df = left_df.sort_values(["target_time", "busRouteId", "stId"]).reset_index(drop=True)
right_df = right_df.sort_values(["future_mkTm", "busRouteId", "stId"]).reset_index(drop=True)

# 디버깅 확인
print("left sorted? ", left_df["target_time"].is_monotonic_increasing)
print("right sorted?", right_df["future_mkTm"].is_monotonic_increasing)

merged = pd.merge_asof(
    left_df,
    right_df,
    left_on="target_time",
    right_on="future_mkTm",
    by=["busRouteId", "stId"],
    direction="forward",
    tolerance=pd.Timedelta(minutes=20)
)

# 타깃이 없는 행 제거
merged = merged.dropna(subset=["target_remaining_seat"]).copy()

# =========================================================
# 5. 입력 feature 정의
# =========================================================
FEATURE_COLS = [
    "route_enc",
    "stid_enc",
    "arsid_enc",
    "year", "month", "day",
    "hour", "minute",
    "dayofweek",
    "is_weekend",
    "is_holiday",
    "is_peak",
    "month_sin", "month_cos",
    "day_sin", "day_cos",
    "hour_sin", "hour_cos",
    "minute_sin", "minute_cos",
    "dow_sin", "dow_cos",
]

TARGET_COL = "target_remaining_seat"

X = merged[FEATURE_COLS].copy()
y = merged[TARGET_COL].copy()

# =========================================================
# 6. 시간순 train / valid / test 분리
# =========================================================
merged = merged.sort_values("mkTm").reset_index(drop=True)

n = len(merged)
train_end = int(n * 0.7)
valid_end = int(n * 0.85)

train_df = merged.iloc[:train_end]
valid_df = merged.iloc[train_end:valid_end]
test_df  = merged.iloc[valid_end:]

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET_COL]

X_valid = valid_df[FEATURE_COLS]
y_valid = valid_df[TARGET_COL]

X_test = test_df[FEATURE_COLS]
y_test = test_df[TARGET_COL]

print("train:", X_train.shape, y_train.shape)
print("valid:", X_valid.shape, y_valid.shape)
print("test :", X_test.shape, y_test.shape)

# =========================================================
# 7. MLP 모델 정의 및 학습
# =========================================================
# StandardScaler + MLPRegressor 파이프라인
mlp_model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    ))
])

mlp_model.fit(X_train, y_train)

# =========================================================
# 8. 검증 / 테스트 평가
# =========================================================
def evaluate_regression(model, X, y, name="dataset"):
    pred = model.predict(X)

    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    within_2 = np.mean(np.abs(y - pred) <= 2)
    within_3 = np.mean(np.abs(y - pred) <= 3)

    print(f"\n[{name}]")
    print(f"MAE       : {mae:.4f}")
    print(f"RMSE      : {rmse:.4f}")
    print(f"R2        : {r2:.4f}")
    print(f"±2석 적중률: {within_2:.4%}")
    print(f"±3석 적중률: {within_3:.4%}")

    return pred

valid_pred = evaluate_regression(mlp_model, X_valid, y_valid, "VALID")
test_pred = evaluate_regression(mlp_model, X_test, y_test, "TEST")

# =========================================================
# 9. 모델 저장
# =========================================================
joblib.dump(mlp_model, "mlp_remaining_seat_model.pkl")
joblib.dump(route_le, "route_label_encoder.pkl")
joblib.dump(stid_le, "stid_label_encoder.pkl")
joblib.dump(arsid_le, "arsid_label_encoder.pkl")

print("\n모델과 인코더 저장 완료")

# =========================================================
# 10. 미래 일시 입력용 feature 생성 함수
# =========================================================
def make_time_features(dt: pd.Timestamp):
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    dayofweek = dt.dayofweek

    # 간단 버전
    is_weekend = 1 if dayofweek >= 5 else 0
    is_holiday = 0   # 필요하면 한국 공휴일 라이브러리로 확장 가능
    is_peak = 1 if ((7 <= hour <= 9) or (17 <= hour <= 19)) else 0

    # 주기성 인코딩
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    day_sin = np.sin(2 * np.pi * day / 31)
    day_cos = np.cos(2 * np.pi * day / 31)

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    minute_sin = np.sin(2 * np.pi * minute / 60)
    minute_cos = np.cos(2 * np.pi * minute / 60)

    dow_sin = np.sin(2 * np.pi * dayofweek / 7)
    dow_cos = np.cos(2 * np.pi * dayofweek / 7)

    return {
        "year": dt.year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "dayofweek": dayofweek,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "is_peak": is_peak,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "minute_sin": minute_sin,
        "minute_cos": minute_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
    }

# =========================================================
# 11. 예측 함수
# =========================================================
def predict_remaining_seat(route_id, st_id, ars_id, future_datetime_str):
    """
    route_id: 예) "100100389"
    st_id   : 예) "100000169"
    ars_id  : 예) "01001"
    future_datetime_str: 예) "2026-03-10 08:30:00"
    """

    dt = pd.Timestamp(future_datetime_str)

    # 학습에 없던 값이면 예측 불가
    if route_id not in route_le.classes_:
        raise ValueError(f"학습 데이터에 없는 busRouteId: {route_id}")
    if st_id not in stid_le.classes_:
        raise ValueError(f"학습 데이터에 없는 stId: {st_id}")
    if ars_id not in arsid_le.classes_:
        raise ValueError(f"학습 데이터에 없는 arsId: {ars_id}")

    row = {
        "route_enc": route_le.transform([route_id])[0],
        "stid_enc": stid_le.transform([st_id])[0],
        "arsid_enc": arsid_le.transform([ars_id])[0],
    }

    row.update(make_time_features(dt))

    X_pred = pd.DataFrame([row])[FEATURE_COLS]
    pred = mlp_model.predict(X_pred)[0]

    # 좌석수는 음수가 되면 안 되므로 보정
    pred = max(0, round(pred))

    return pred

# =========================================================
# 12. 예측 예시
# =========================================================
sample_route = merged.iloc[-1]["busRouteId"]
sample_stid  = merged.iloc[-1]["stId"]
sample_arsid = merged.iloc[-1]["arsId"]

pred_seat = predict_remaining_seat(
    route_id=sample_route,
    st_id=sample_stid,
    ars_id=sample_arsid,
    future_datetime_str="2026-03-10 18:10:00"
)

print(f"\n예측 남은 좌석수: {pred_seat}석")