# 📏 Experiment Logger 팀 규칙

## 1️⃣ 모든 실험은 반드시 저장

* 테스트라도 무조건 logger 사용
* 결과 삭제 금지

---

## 2️⃣ 메타 정보 필수 입력

* RUNNER 반드시 작성
* MODEL_VERSION 반드시 작성

---

## 3️⃣ 버전 관리 규칙

### MODEL_VERSION

* 모델 구조/파라미터 변경 시 증가

예:

* v1: 기본 모델
* v2: learning rate 변경
* v3: feature 추가

---

### DATA_VERSION

* 데이터 변경 시 업데이트

예:

* 20260415
* weather_added_v1

---

### FEATURE_VERSION

* feature 변경 시 업데이트

예:

* basic_v1
* pattern_v2
* pattern_weather_v1

---

## 4️⃣ best 결과는 자동 관리

* 직접 수정 금지
* logger가 자동 갱신

---

## 5️⃣ 파일 삭제 금지

* logs / reports / params
* 실험 기록은 자산

---

## 6️⃣ 동일 조건 비교 기준

다음이 같을 때만 성능 비교 가능:

* dataset
* data_version
* split_version
* feature_version
* label_definition

---

## 7️⃣ 모델 이름 규칙

* lgbm_reg
* lgbm_peak_cls
* xgb_reg
* rf_cls

통일해서 사용

```
```
