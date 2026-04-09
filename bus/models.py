from django.db import models


class Bus_route(models.Model):
    # 노선 아이디 (NOT NULL)
    routeId = models.CharField(
        max_length=50, primary_key=True, verbose_name="노선아이디"
    )

    # 노선 이름 (NOT NULL)
    routeName = models.CharField(
        max_length=20, null=False, blank=False, verbose_name="노선명"
    )

    def __str__(self):
        return self.routeName

    class Meta:
        db_table = "bus_route"
        verbose_name = "노선"
        verbose_name_plural = "노선 목록"
        ordering = ["routeId"]  # 노선아이디 기준 정렬


# bus_station


class Bus_station(models.Model):
    # 정류소 고유 ID
    stationId = models.CharField(
        max_length=50, primary_key=True, verbose_name="정류소고유아이디"
    )

    # 정류소 이름
    stationName = models.CharField(
        max_length=100, null=False, blank=False, verbose_name="정류소명"
    )

    stn = models.CharField(
        max_length=50, null=False, blank=False, verbose_name="기상지점번호"
    )

    # 경도 (Longitude)
    locationX = models.FloatField(null=False, verbose_name="경도")

    # 위도 (Latitude)
    locationY = models.FloatField(null=False, verbose_name="위도")

    address = models.CharField(
        max_length=255, null=False, blank=False, verbose_name="주소지")
    
    isVirtual = models.CharField(
        max_length=20, null=False, blank=False, verbose_name="가상정류소 여부")
    

    # 사용자가 보는 정류소 ID
    arsId = models.CharField(
        max_length=100, null=False, blank=False, verbose_name="사용자가 보는 정류소"
    )

    def __str__(self):
        return self.stationName

    class Meta:
        db_table = "bus_station"  # 테이블명 명시
        verbose_name = "정류소"
        verbose_name_plural = "정류소 목록"
        ordering = ["stationId"]  # 정류소 ID 기준 정렬


class Route_station(models.Model):

    # 노선 FK
    route = models.ForeignKey(
        "Bus_route",  # 너가 만든 노선 모델 이름
        on_delete=models.CASCADE,
        to_field="routeId",
        db_column="routeId",
        related_name="route_stations",
        verbose_name="노선",
    )

    # 정류소 FK
    station = models.ForeignKey(
        "Bus_station",
        on_delete=models.CASCADE,
        to_field="stationId",
        db_column="stationId",
        related_name="station_routes",
        verbose_name="정류소",
    )

    # 정류소 순서
    staOrd = models.IntegerField(null=False, verbose_name="정류소 순서")

    def __str__(self):
        return f"{self.id} - {self.route_id} - {self.station_id} ({self.staOrd})"

    class Meta:
        db_table = "route_station"
        verbose_name = "노선-정류소"
        verbose_name_plural = "노선-정류소 목록"
        ordering = ["route", "staOrd"]  # 노선별 + 순서 정렬

        # 같은 노선에서 같은 정류소 중복 방지
        unique_together = ("route", "station")
