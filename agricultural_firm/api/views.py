from django.http import JsonResponse, HttpRequest
from django.views import View

import json

from .models import *

ErrorJsonResponse = {
    "status": "ERROR",
    "message": "Request Error"
} 

class getPlotInfo(View):
    """
    params:
        plot_id: str
    """
    def get(self, request: HttpRequest):
        try:
            id = request.GET['plot_id']

            plot = Plot.objects.get(id=id)
            p_c = Plot_culture.objects.filter(plot_id=id).all()

            resp = {
                "square": plot.square,
                "soil_quality": plot.soil_quality.title,
                "cultures": [{"name": c.culture.name, "sowing_percent": c.sowing_percent} for c in p_c]
            }

            return JsonResponse(resp)
        
        except Exception as ex:
            return JsonResponse(ErrorJsonResponse)