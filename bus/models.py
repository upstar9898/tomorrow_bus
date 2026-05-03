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


class Weather_station(models.Model):
    # 기존 Bus_station.stn 이 CharField 였으므로 타입 맞춤
    stnId = models.CharField(
        max_length=50,
        primary_key=True,
        null=False,
        blank=False,
        verbose_name="관측소 ID",
    )

    stnName = models.CharField(
        max_length=100,
        null=False,
        blank=False,
        verbose_name="관측소명",
    )

    locationX = models.FloatField(
        null=False,
        verbose_name="경도",
    )

    locationY = models.FloatField(
        null=False,
        verbose_name="위도",
    )

    class Meta:
        db_table = "weather_station"
        ordering = ["stnId"]

    def __str__(self):
        return f"{self.stnId} - {self.stnName}"


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
    # 기존 필드명 stn 유지 + FK로 변경
    stn = models.ForeignKey(
        Weather_station,
        on_delete=models.SET_NULL,
        to_field="stnId",
        db_column="stn",
        null=True,
        blank=True,
        related_name="bus_stations",
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
                fields=["mkTm", "route", "station"], name="unique_arrival_snapshot"
            )
        ]

    def __str__(self):
        return (
            f"{self.mkTm} / {self.route_id} / {self.station_id} / {self.remaining_seat}"
        )


class Bus_info(models.Model):
    """
    노선별 운행 기본 정보 테이블
    첫차, 막차, 배차간격 저장용
    """

    # 노선 ID
    route = models.ForeignKey(
        Bus_route,
        on_delete=models.CASCADE,
        db_column="route_id",
        related_name="bus_infos",
    )

    # 평일 첫차
    firstTm = models.TimeField(
        null=False,
        verbose_name="평일 첫차",
    )

    # 평일 막차
    lastTm = models.TimeField(
        null=False,
        verbose_name="평일 막차",
    )

    # 평일 배차간격
    timeGap = models.PositiveIntegerField(
        null=False,
        verbose_name="평일 배차간격",
    )

    # 토요일 첫차
    saturdayFirstTm = models.TimeField(
        null=False,
        verbose_name="토요일 첫차",
    )

    # 토요일 막차
    saturdayLastTm = models.TimeField(
        null=False,
        verbose_name="토요일 막차",
    )

    # 토요일 배차간격
    saturdaytimeGap = models.PositiveIntegerField(
        null=False,
        verbose_name="토요일 배차간격",
    )

    # 공휴일 첫차
    holidayFirstTm = models.TimeField(
        null=False,
        verbose_name="공휴일 첫차",
    )

    # 공휴일 막차
    holidayLastTm = models.TimeField(
        null=False,
        verbose_name="공휴일 막차",
    )

    # 공휴일 배차간격
    holidaytimeGap = models.PositiveIntegerField(
        null=False,
        verbose_name="공휴일 배차간격",
    )

    # 생성 날짜
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "bus_info"
        ordering = ["route"]
        indexes = [
            models.Index(fields=["route"]),
        ]
        constraints = [
            models.UniqueConstraint(fields=["route"], name="unique_bus_info_route")
        ]

    def __str__(self):
        return f"{self.route_id}"
