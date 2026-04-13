from django.core.management.base import BaseCommand
from bus.services.weather_collector import collect_weather

class Command(BaseCommand):
    help = "날씨 데이터 1회 수집"

    def handle(self, *args, **options):
        collect_weather()
        self.stdout.write(self.style.SUCCESS("수집 완료"))