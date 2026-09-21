from django.urls import path
from django.shortcuts import render
from rag.api import api

def home(request):
    return render(request, 'chat.html')

urlpatterns = [
    path('', home, name='home'),
    path('api/', api.urls),
]
