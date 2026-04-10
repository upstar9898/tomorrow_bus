"""
URL configuration for tommorow_bus project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="home"),
    path("service1/", views.service1, name="service1"),
    path("service2/", views.service2, name="service2"),
    path("favorites/", views.favorites, name="favorites"),
    path("ajax/stations/", views.get_stations_by_route, name="ajax_stations"),  # 정류장 목록
    path("ajax/predict/", views.predict_service1, name="ajax_predict"),  # 예측
]
