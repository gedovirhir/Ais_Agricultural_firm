from django.shortcuts import render
from django.views import View
from api.views import getAllCultures, getAllSoils
#from django.contrib.sites.models import Site

import requests

#current_site = Site.objects.get_current()
my_api = f"http://25.46.163.182:8000/api/v1"

class getMainPage(View):
    
    def get(self, request):
        cultures = requests.get(f"{my_api}/getAllCultures").json()
        soils = requests.get(f"{my_api}/getAllSoils").json()
        
        context = {
            'cultures': cultures,
            'soils': soils
        }
        
        return render(request, 'index.html', context=context)