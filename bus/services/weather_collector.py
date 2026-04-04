import requests
import os

def download_file(file_url, save_path):
    response = requests.get(file_url)

    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print("다운로드 성공")
    else:
        print("다운로드 실패:", response.status_code)

def collect_weather():
    api_key = os.environ.get("WEATHER_API_KEY")
    url = f"https://apihub.kma.go.kr/api/typ01/url/fct_afs_dl.php?reg=&tmfc1=2026031106&tmfc2=2026040400&disp=0&help=1&authKey={api_key}"
    save_path = "output_file.txt"

    download_file(url, save_path)