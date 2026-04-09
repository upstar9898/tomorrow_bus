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
