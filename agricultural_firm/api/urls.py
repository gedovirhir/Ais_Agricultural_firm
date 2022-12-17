from django.contrib import admin
from django.urls import path, include

from .views import *

urlpatterns = [
    path('getAllCultures', getAllCultures.as_view()),
    path('getAllSoils', getAllSoils.as_view()),
    
    
]