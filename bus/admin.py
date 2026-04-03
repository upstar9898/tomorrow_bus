from django.contrib import admin, messages
from django.shortcuts import render, redirect
from django.urls import path
import csv
from .models import Bus_route, Bus_station, Route_station


@admin.register(Bus_route)
class BusRouteAdmin(admin.ModelAdmin):
    list_display = ('id', 'routeId', 'routeName')
    search_fields = ('routeId', 'routeName')
    ordering = ('routeId',)


@admin.register(Bus_station)
class BusStationAdmin(admin.ModelAdmin):
    list_display = ('id', 'stationId', 'stationName', 'locationX', 'locationY')
    search_fields = ('stationId', 'stationName')
    ordering = ('stationId',)

