import os
import pandas as pd


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    file_path = os.path.join(data_dir, "route_stations.csv")

    df = pd.read_csv(file_path)

    # -----------------------
    # 👇 여기만 바꿔서 사용
    # -----------------------
    TARGET_ROUTE_ID = 100100400
    INSERT_STAORD = 45
    NEW_STATION_ID = 218000209
    # -----------------------

    # 1️⃣ 해당 노선만 분리
    route_df = df[df["routeId"] == TARGET_ROUTE_ID].copy()
    other_df = df[df["routeId"] != TARGET_ROUTE_ID].copy()

    # 2️⃣ staOrd 밀기 (뒤로)
    route_df.loc[
        route_df["staOrd"] >= INSERT_STAORD, "staOrd"
    ] += 1

    # 3️⃣ 신규 row 생성
    new_row = pd.DataFrame(
        [{
            "routeId": TARGET_ROUTE_ID,
            "stationId": NEW_STATION_ID,
            "staOrd": INSERT_STAORD,
        }]
    )

    # 4️⃣ 합치기
    route_df = pd.concat([route_df, new_row], ignore_index=True)

    # 5️⃣ 정렬
    route_df = route_df.sort_values(by="staOrd")

    # 6️⃣ 전체 합치기
    result_df = pd.concat([other_df, route_df], ignore_index=True)

    # 7️⃣ 최종 정렬
    result_df = result_df.sort_values(by=["routeId", "staOrd"])

    # 8️⃣ 저장
    result_df.to_csv(file_path, index=False, encoding="utf-8-sig")

    print("\n[완료]")
    print(f"노선 {TARGET_ROUTE_ID}에 정류소 추가 완료")
    print(f"삽입 위치: {INSERT_STAORD}")
    print(f"정류소 ID: {NEW_STATION_ID}")


if __name__ == "__main__":
    main()