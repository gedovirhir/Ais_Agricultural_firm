from ..models import *

from datetime import datetime, timedelta
from django.db.models import QuerySet
from typing import Optional, Iterable

def _add_default_data():
    c1 = [Culture(
        name="MegaCulture", 
        productivity_k=10,
        fav_temp_bot=25,
        fav_temp_top=25,
        fav_precip_bot=25,
        fav_precip_top=25,
    )]
    sq1 = [Soil_quality(
        title="SuperGood",
        fertility_k=10,
    )]
    per1 =[
        Period(
            title="Summer", 
            start_date=datetime.now(), 
            end_date=datetime.now() + timedelta(days=90)
        ),
        Period(
            title="Spring", 
            start_date=datetime.now() - timedelta(days=90), 
            end_date=datetime.now()
        ),
        Period(
            title="Autumn", 
            start_date=datetime.now() + timedelta(days=90), 
            end_date=datetime.now() + timedelta(days=180)
        )
    ]
    mr1 = [
        Meteo_report(
            period=per1[0], 
            report_date=datetime.now(),
            temperature=25,
            precipitation=25,
        ),
        Meteo_report(
            period=per1[0], 
            report_date=datetime.now() - timedelta(days=1),
            temperature=15,
            precipitation=15,
        ),
        Meteo_report(
            period=per1[1], 
            report_date=datetime.now(),
            temperature=10,
            precipitation=10,
        ),
        Meteo_report(
            period=per1[1], 
            report_date=datetime.now() - timedelta(days=1),
            temperature=15,
            precipitation=15,
        )
    ]
    
    objs = c1 + sq1 + per1 + mr1
    for o in objs: o.save()

def get_all_cultures() -> QuerySet[Culture]:
    res = Culture.objects.all()
    
    return res

def get_all_soils() -> QuerySet[Soil_quality]:
    res = Soil_quality.objects.all()
    
    return res

def get_actual_period(now_time: Optional[datetime] = None) -> Optional[Period]:
    """
    Возвращает период по дате, если дата не введенна, то по актуальной дате
    """
    now_time = now_time if now_time else datetime.now()
    res = Period.objects.filter(
        start_date__lte=now_time,
        end_date__gte=now_time
    ).first()
    
    return res

def get_all_previous_periods(period_start_date: datetime, limit=1) -> QuerySet[Period]:
    """
    Возвращает все периоды, ранее введенной даты
    """
    res = Period.objects.filter(end_date__lte=period_start_date)\
                        .order_by('start_date')\
                        .all()[:limit]
    
    return res

def get_period_meteo_reports(period_id: int) -> QuerySet[Meteo_report]:
    """
    Возвращает все прогнозы погоды, принадлежащие периоду
    """
    res = Meteo_report.objects.filter(period_id=period_id)\
                              .order_by('report_date')\
                              .all()
    
    return res

def add_regression_prognose(
    period_id: int,
    avg_temp: float,
    avg_precip: float
):
    new = Regression_prognoses(
        period_id=period_id,
        avg_temp=avg_temp,
        avg_precip=avg_precip,
    )
    
    new.save()



if __name__ == "__main__":
    _add_default_data()