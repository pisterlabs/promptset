from django.urls import path
from openai_api import views

urlpatterns = [
    path('img/', views.img_generate),
    path('eval/', views.grade_example)
]