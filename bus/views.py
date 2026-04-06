from django.shortcuts import render


def index(request):
    return render(request, "index.html")


def service1(request):
    return render(request, "service1.html")


def service2(request):
    return render(request, "service2.html")


def favorites(request):
    return render(request, "favorites.html")
