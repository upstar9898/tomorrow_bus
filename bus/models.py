from django.db import models


class Bus_route(models.Model):
    # 노선 아이디를 직접 PK로 사용
    routeId = models.CharField(
        max_length=50,
        primary_key=True,
        null=False,
        blank=False,
        verbose_name="노선아이디",
    )

    # 노선 이름
    routeName = models.CharField(
        max_length=100,
        null=False,
        blank=False,
        verbose_name="노선명",
    )

    def __str__(self):
        return f"{self.routeName} ({self.routeId})"

    class Meta:
        db_table = "bus_route"
        verbose_name = "노선"
        verbose_name_plural = "노선 목록"
        ordering = ["routeId"]


class Bus_station(models.Model):
    # 정류소 ID를 직접 PK로 사용
    stationId = models.CharField(
        max_length=50,
        primary_key=True,
        null=False,
        blank=False,
        verbose_name="정류소아이디",
    )

    # 정류소명
    stationName = models.CharField(
        max_length=255,
        null=False,
        blank=False,
        verbose_name="정류소명",
    )

    # 기상 관측소 번호
    stn = models.CharField(
        max_length=50,
        null=False,
        blank=True,
        verbose_name="기상지점번호",
    )

    # 경도
    locationX = models.FloatField(
        null=False,
        verbose_name="경도",
    )

    # 위도
    locationY = models.FloatField(
        null=False,
        verbose_name="위도",
    )

    # 주소
    address = models.CharField(
        max_length=255,
        null=False,
        blank=True,
        verbose_name="주소",
    )

    # 가상 정류소 여부
    isVirtual = models.CharField(
        max_length=20,
        null=False,
        blank=True,
        verbose_name="가상정류소여부",
    )

    # 정류소 번호
    arsId = models.CharField(
        max_length=50,
        null=True,
        blank=True,
        verbose_name="정류소번호",
    )

    def __str__(self):
        return f"{self.stationName} ({self.stationId})"

    class Meta:
        db_table = "bus_station"
        verbose_name = "정류소"
        verbose_name_plural = "정류소 목록"
        ordering = ["stationId"]


class Route_station(models.Model):

    # 노선 FK
    route = models.ForeignKey(
        Bus_route,
        on_delete=models.CASCADE,
        to_field="routeId",
        db_column="routeId",
        related_name="route_stations",
        verbose_name="노선",
    )

    # 정류소 FK
    station = models.ForeignKey(
        Bus_station,
        on_delete=models.CASCADE,
        to_field="stationId",
        db_column="stationId",
        related_name="station_routes",
        verbose_name="정류소",
    )

    # 정류소 순서
    staOrd = models.IntegerField(
        null=False,
        verbose_name="정류소 순서",
    )

    def __str__(self):
        return f"{self.id} - {self.route_id} - {self.station_id} ({self.staOrd})"

    class Meta:
        db_table = "route_station"
        verbose_name = "노선-정류소"
        verbose_name_plural = "노선-정류소 목록"
        ordering = ["route", "staOrd"]
        unique_together = ("route", "station")

class Bus_arrival_info(models.Model):
    """
    차트 조회용 최소 스냅샷 테이블
    """
    # 
    mkTm = models.DateTimeField(db_index=True)

    # 노선 ID
    route = models.ForeignKey(
        Bus_route,
        on_delete=models.CASCADE,
        db_column="route_id",
        related_name="arrival_infos",
    )

    # 정류소 ID
    station = models.ForeignKey(
        Bus_station,
        on_delete=models.CASCADE,
        db_column="station_id",
        related_name="arrival_infos",
    )

    # 정류소 순서
    staOrd = models.PositiveIntegerField(db_index=True)

    # 잔여 좌석
    remaining_seat = models.IntegerField()

    # 만차여부
    full_flag = models.BooleanField(default=False)

    # 생성 날짜
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "bus_arrival_info"
        ordering = ["mkTm", "route", "staOrd"]
        indexes = [
            models.Index(fields=["route", "mkTm"]),
            models.Index(fields=["route", "staOrd"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["mkTm", "route", "station"],
                name="unique_arrival_snapshot"
            )
        ]

    def __str__(self):
        return f"{self.mkTm} / {self.route_id} / {self.station_id} / {self.remaining_seat}"
    
class Weather_station(models.Model):
    stn_id = models.IntegerField(
        primary_key=True,
        verbose_name="관측소 ID",
    )

    name_ko = models.CharField(
        max_length=100,
        null=False,
        verbose_name="관측소명",
    )

    # 경도
    lon = models.FloatField(
        null=False,
        verbose_name="경도",
    )

    # 위도
    lat = models.FloatField(
        null=False,
        verbose_name="위도",
    )

    class Meta:
        db_table = "weather_station"
        ordering = ["stn_id"]

    def __str__(self):
        return f"{self.stn_id} - {self.name_ko}"    