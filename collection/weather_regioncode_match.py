import os
import pandas as pd


def main():
    # 현재 파일 위치: 프로젝트루트/collection/파일.py
    current_file = os.path.abspath(__file__)
    collection_dir = os.path.dirname(current_file)
    base_dir = os.path.dirname(collection_dir)   # 프로젝트 루트
    data_dir = os.path.join(base_dir, "data")

    os.makedirs(data_dir, exist_ok=True)

    input_path = os.path.join(data_dir, "weather_regioncode.txt")
    output_path = os.path.join(data_dir, "weather_regioncode.csv")

    print("현재 파일 위치:", current_file)
    print("프로젝트 루트:", base_dir)
    print("data 폴더:", data_dir)
    print("입력 파일:", input_path)
    print("출력 파일:", output_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"입력 파일이 없습니다: {input_path}")

    rows = []

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip()

            # 주석 제거
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # 최소 길이 체크
            if len(parts) < 13:
                continue

            try:
                row = {
                    "STN": int(parts[0]),
                    "LON": float(parts[1]),
                    "LAT": float(parts[2]),
                    "STN_KO": parts[10],
                    "STN_EN": parts[11],
                    "FCT_ID": parts[12],
                    "LAW_ID": parts[13] if len(parts) > 13 else None,
                    "LAW_ADDR": " ".join(parts[15:]) if len(parts) > 15 else None,
                }
                rows.append(row)

            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(rows)

    print("\n[변환 결과]")
    print("행 개수:", len(df))

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n[완료]")
    print(f"CSV 변환 완료: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]")
        print(type(e).__name__, ":", e)
        raise