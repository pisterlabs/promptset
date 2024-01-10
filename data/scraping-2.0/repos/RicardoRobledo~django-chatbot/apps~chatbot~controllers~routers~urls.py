from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from ...views import openai_views


urlpatterns = [
    path('', openai_views.OpenAIView.as_view(), name='chatgpt'),
]

urlpatterns = format_suffix_patterns(urlpatterns)
