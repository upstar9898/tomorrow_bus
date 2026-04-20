import os
import sys

# Django 환경 세팅 (중요!!)
def setup_django():
    current_file = os.path.abspath(__file__)
    collection_dir = os.path.dirname(current_file)
    base_dir = os.path.dirname(collection_dir)

    sys.path.append(base_dir)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tomorrow_bus.settings")

    import django
    django.setup()


def main():
    setup_django()

    from bus.services.weather_collector import collect_weather

    print("[날씨 수집 시작]")
    collect_weather()
    print("[날씨 수집 완료]")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e)
        raise