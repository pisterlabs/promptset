"""
URL configuration for mysite project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
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
from django.contrib import admin
from django.views.generic import RedirectView
from django.urls import path, include, re_path
from django.shortcuts import render
import os
from openai import OpenAI

from Game import views
from mysite import views

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

tour_assistant_id = os.getenv('TOUR_ASSISTANT_ID')
TOUR_ASSISTANT_ID = tour_assistant_id

def home(request):
    # thread = client.beta.threads.create()
    # request.session['openai_thread_id'] = thread.id
    # threadsession = request.session.get('openai_thread_id', 'cannot find thread id')
    # print('Saved data', threadsession)
    return render(request, "home.html")

urlpatterns = [
    path('admin/', admin.site.urls),
    # game/을 주소에 붙이면 game.urls참조
    path("game/", include("Game.urls")),
    # path('getImage/', views.getImagepage, name='getImage'),
    # path('create_thread', views.create_thread, name='create_thread'),
    re_path(r'^favicon\.ico$', RedirectView.as_view(url='/static/favicon.ico')),

    # 초기화면
    path("", home)
]
