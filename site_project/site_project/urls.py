from django.urls import path, include

urlpatterns = [
    path("formatter/", include("formatter.urls")),
]
