from rest_framework.views import APIView, Response
from rest_framework import status
import json

from .models import *
from .utils.db_api import *
from .utils.prognose_utils import calculate_productivity_on_period
from .serializers import *

ErrorJsonResponse = {
    "status": "ERROR",
    "message": "Request Error"
} 

class getAllCultures(APIView):
    
    def get(self, request):
        res = get_all_cultures()
        
        resp = CultureSerializer(res, many=True).data
        return Response(resp)

class getAllSoils(APIView):
    
    def get(self, request):
        res = get_all_soils()
        
        resp = SoilSerializer(res, many=True).data
        return Response(resp)

class getPrognose(APIView):
    
    def get(self, request):
        r_params = GetPrognoseSerializer(data=request.query_params)
        r_params.is_valid(raise_exception=True)
        
        params = r_params.data
        
        p = get_actual_period(params.get('date'))
        c = Culture.objects.get(id=params['culture_id'])
        soil = Soil_quality.objects.get(id=params['soil_type_id'])
        
        periods_query = get_all_previous_periods(p.start_date, limit=4 )
        reports_query = get_period_meteo_reports([pr.id for pr in periods_query])
        
        reports = [(rep.report_date, rep.temperature, rep.precipitation) for rep in reports_query]
        
        prodctv, weather_k = calculate_productivity_on_period(
            c.productivity_k,
            (c.fav_temp_bot, c.fav_temp_top),
            (c.fav_precip_bot, c.fav_precip_top),
            soil.fertility_k,
            params['sowing_area'],
            reports,
            p.start_date,
            p.end_date
        )
        
        res = {
            "prod": prodctv,
            "weather": weather_k
        }
        
        return Response(res)
    
        
        
"""
class getAllPreviousPeriodsMeteo(APIView):
    
    def get(self, request):
        p_date = request.query_params.get('period_start_date')
        if p_date:
            try:
                p_date = datetime.fromisoformat(p_date)
            except Exception:
                return Response("Wrong datetime format", status=status.HTTP_400_BAD_REQUEST)
        
        now_per = get_actual_period(p_date)
        
        if not now_per:
            return Response("No period at this date", status=status.HTTP_400_BAD_REQUEST)
        else:
            now_per = now_per.start_date
        
        p_limit = request.query_params.get('limit')
        p_limit = p_limit if p_limit else 1
        
        prev_pers = get_all_previous_periods(now_per, p_limit)
"""