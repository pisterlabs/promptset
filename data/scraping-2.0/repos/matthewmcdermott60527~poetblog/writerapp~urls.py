from django.urls import path
from . import views
from writerapp.views import openai_app_view
from django.urls import path

app_name = 'writerapp'

urlpatterns = [
    path('create/', views.create_post, name='create_post'),
    path('update/<int:pk>/', views.update_post, name='update_post'),
    path('', views.index, name='index'),
    path("category/", views.category, name="writerapp_category"),
    path('about/', views.about_view, name='about'),
    path('experiments/', views.experiments_view, name='experiments'),
    path('books/', views.books_view, name='books'),
    path('music/', views.music_view, name='music'),
    path('contact/', views.contact_view, name='contact'),
    path('art/', views.art_view, name='art'),
    path('success/', views.success, name='success'),
    path('experiments/', views.openai_app_view, name='openai_app'),
    path('submit_poem/', views.generate_critique, name='submit_poem'),
    path('<slug:slug>/', views.detail, name='detail')
]
