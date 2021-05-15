from django.urls import path
from . import views

urlpatterns = [
    path('',views.VideoProcessing.as_view()),
]