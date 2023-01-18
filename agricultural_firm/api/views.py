from rest_framework.views import APIView, Response
from rest_framework import status
import json

from .models import *
from .utils.db_api import *
from .utils.prognose_utils import calculate_productivity_on_period
from .serializers import *

DEFAULT_SEASONS_N = {
    "Весна": 0,
    "Лето": 1,
    "Осень": 2,
    "Зима": 3
}
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
        
        p = get_actual_period(params.get('year'), params.get('season'))
        c = Culture.objects.get(id=params['culture_id'])
        soil = Soil_quality.objects.get(id=params['soil_type_id'])
        per_n = DEFAULT_SEASONS_N[params['season']]
        periods_count = 4 + per_n
        
        periods_query = get_all_previous_periods(p.start_date, limit=periods_count)
        
        prev_year_rep_q = get_period_meteo_reports([pr.id for pr in periods_query[:4]])
        now_year_rep_q = get_period_meteo_reports([pr.id for pr in periods_query[4:]])
        
        prev_rep = [(rep.report_date, 1, rep.temperature, rep.precipitation, rep.wind) for rep in prev_year_rep_q]
        now_rep = [(rep.report_date, 0, rep.temperature, rep.precipitation, rep.wind) for rep in now_year_rep_q]
        
        reports = prev_rep + now_rep
        
        prognose_res = calculate_productivity_on_period(
            c.productivity_k,
            (c.fav_temp_bot, c.fav_temp_top),
            (c.fav_precip_bot, c.fav_precip_top),
            soil.fertility_k,
            params['sowing_area'],
            reports,
            p.start_date,
            p.end_date
        )
        
        return Response(prognose_res)
    
        
        
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