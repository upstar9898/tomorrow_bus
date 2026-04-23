import os
import csv
import requests
import xml.etree.ElementTree as ET


def main():
    # 현재 파일 위치 기준
    current_file = os.path.abspath(__file__)
    collection_dir = os.path.dirname(current_file)
    base_dir = os.path.dirname(collection_dir)   # 프로젝트 루트
    data_dir = os.path.join(base_dir, "data")

    os.makedirs(data_dir, exist_ok=True)

    input_txt_path = os.path.join(data_dir, "reg_id_list.txt")
    output_csv_path = os.path.join(data_dir, "fcst_zone_regid_regname.csv")

    print("현재 파일 위치:", current_file)
    print("프로젝트 루트:", base_dir)
    print("data 폴더:", data_dir)
    print("입력 파일:", input_txt_path)
    print("출력 파일:", output_csv_path)

    if not os.path.exists(input_txt_path):
        raise FileNotFoundError(f"입력 txt 파일이 없습니다: {input_txt_path}")

    service_key = os.environ.get("WEATHER_ZONE_API_KEY")
    if not service_key:
        raise EnvironmentError("환경변수 WEATHER_ZONE_API_KEY가 설정되지 않았습니다.")

    base_url = "https://apis.data.go.kr/1360000/FcstZoneInfoService/getFcstZoneCd"

    # REG_ID 목록 읽기
    with open(input_txt_path, "r", encoding="utf-8") as f:
        reg_ids = [line.strip() for line in f if line.strip()]

    print(f"총 REG_ID 수: {len(reg_ids)}")

    result_rows = []

    for idx, reg_id in enumerate(reg_ids):
        params = {
            "serviceKey": service_key,
            "pageNo": 1,
            "numOfRows": 10,
            "dataType": "XML",
            "regId": reg_id,
        }

        try:
            response = requests.get(base_url, params=params, timeout=20)
            response.raise_for_status()

            root = ET.fromstring(response.text)

            result_code = root.findtext(".//header/resultCode")
            result_msg = root.findtext(".//header/resultMsg")

            if result_code != "00":
                print(f"[건너뜀] regId={reg_id}, code={result_code}, msg={result_msg}")
                continue

            items = root.findall(".//items/item")

            if not items:
                print(f"[데이터 없음] regId={reg_id}")
                continue

            for item in items:
                api_reg_id = item.findtext("regId", default="").strip()
                reg_name = item.findtext("regName", default="").strip()

                result_rows.append({
                    "regId": api_reg_id,
                    "regName": reg_name,
                })

            print(f"[완료] {idx+1}/{len(reg_ids)} regId={reg_id}")

        except Exception as e:
            print(f"[오류] regId={reg_id}, error={e}")

    # CSV 저장
    with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["regId", "regName"])
        writer.writeheader()
        writer.writerows(result_rows)

    print("\n[완료]")
    print(f"CSV 저장: {output_csv_path}")
    print(f"총 {len(result_rows)}건")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]")
        print(type(e).__name__, ":", e)
        raise