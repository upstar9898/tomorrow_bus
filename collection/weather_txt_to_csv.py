import os
import glob
import pandas as pd


FILE_NAME_FORMAT = "weather_raw_*.txt"

# 강제 고정 컬럼
COLS = [
    "TM",
    "STN",
    "WD",
    "WS",
    "GST_WD",
    "GST_WS",
    "GST_TM",
    "PA",
    "PS",
    "PT",
    "PR",
    "TA",
    "TD",
    "HM",
    "PV",
    "RN",
    "RN_DAY",
    "RN_JUN",
    "RN_INT",
    "SD_HR3",
    "SD_DAY",
    "SD_TOT",
    "WC",
    "WP",
    "WW",
    "CA_TOT",
    "CA_MID",
    "CH_MIN",
    "CT",
    "CT_TOP",
    "CT_MID",
    "CT_LOW",
    "VS",
    "SS",
    "SI",
    "ST_GD",
    "TS",
    "TE_005",
    "TE_01",
    "TE_02",
    "TE_03",
    "ST_SEA",
    "WH",
    "BF",
    "IR",
    "IX",
]


def main():
    # 현재 파일 위치: 프로젝트루트/collection/파일명.py
    current_file = os.path.abspath(__file__)
    collection_dir = os.path.dirname(current_file)
    base_dir = os.path.dirname(collection_dir)   # 프로젝트 루트
    data_dir = os.path.join(base_dir, "data")

    os.makedirs(data_dir, exist_ok=True)

    print("현재 파일 위치:", current_file)
    print("프로젝트 루트:", base_dir)
    print("data 폴더:", data_dir)

    # 최신 txt 파일 찾기
    pattern = os.path.join(data_dir, FILE_NAME_FORMAT)
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"{pattern} 파일이 없습니다.")

    txt_path = max(files, key=os.path.getmtime)
    csv_path = txt_path.replace(".txt", ".csv")

    print("사용할 파일:", txt_path)
    print("출력 파일:", csv_path)

    data_lines = []

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # 빈 줄 제외
            if not line:
                continue

            # 주석/설명 줄 제외
            if line.startswith("#"):
                continue

            # 데이터 라인만 수집
            if line[0].isdigit():
                row = line.split()

                # 컬럼 수 맞추기
                if len(row) < len(COLS):
                    row += [None] * (len(COLS) - len(row))
                elif len(row) > len(COLS):
                    row = row[:len(COLS)]

                data_lines.append(row)

    if not data_lines:
        raise ValueError("데이터 라인을 찾지 못했습니다.")

    # DataFrame 생성
    df = pd.DataFrame(data_lines, columns=COLS)

    # 숫자형 변환
    string_cols = ["TM", "STN", "WW"]
    numeric_cols = [col for col in COLS if col not in string_cols]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df[string_cols] = df[string_cols].astype(str)

    # CSV 저장
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n[완료]")
    print(f"CSV 변환 완료: {csv_path}")
    print(f"행 수: {len(df)}")
    print(f"컬럼 수: {len(df.columns)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]")
        print(type(e).__name__, ":", e)
        raise