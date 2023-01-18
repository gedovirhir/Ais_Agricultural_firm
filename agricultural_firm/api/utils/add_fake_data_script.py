import os
import sys

sys.path.append(os.path.abspath('agricultural_firm/api'))

from ..models import *

import pandas as pd
from datetime import datetime, timedelta
from random import randint

def _delete_all_data():
    mdls = [Culture, Soil_quality, Period, Meteo_report]
    
    for m in mdls:
        m.objects.all().delete()

def _per_bound(year: int, per_i: int) -> datetime:
    periods = {
        0: (-12, 2),
        1: (3, 5),
        2: (6, 8),
        3: (9, 11),
        4: (12, 2)
    }
    per = periods[per_i]
    s_m = per[0]
    e_m = per[1] + 1
    
    s_y = year - abs(s_m - 1) // 13
    e_y = year + abs(s_m + 1) // 13
    
    s_date = datetime(s_y, abs(s_m), 1)
    e_date = datetime(e_y, e_m, 1)
    
    
    return (s_date, e_date)

def _per_by_month(month_i: int):
    if month_i == 12:
        return 4
    return (month_i % 12) // 3

def _add_fake_data():
    _delete_all_data()
    
    PER = {
        0: "Зима",
        1: "Весна",
        2: "Лето",
        3: "Осень",
        4: "Зима"
    }
    def __fake_cultures():
        names = ['Морковь', 'Мандрагора', 'Глазоцвет', 'Кринжовник', 'Картофель', 
                 'Кабачки', 'Цукини', 'Патиссон', 'Крукнек', 'Люффа']
        
        for n in names:
            fav_temp_bot = randint(10, 20)
            fav_temp_top = fav_temp_bot + randint(5, 10)
            fav_precip_bot = randint(1, 20)
            fav_precip_top = fav_precip_bot + randint(1, 20)
            
            new_c = Culture(
                name=n,
                productivity_k=randint(50, 200),
                fav_temp_bot=fav_temp_bot,
                fav_temp_top=fav_temp_top,
                fav_precip_bot=fav_precip_bot,
                fav_precip_top=fav_precip_top,
            )
            new_c.save()
            
    def __fake_soils():
        soils = [
            ('Натуральная', 1),
            ('Антропогенная', 0.4),
            ('Естественно антропогенная', 0.7)
        ]
        for tl, kf in soils:
            new_s = Soil_quality(
                title=tl,
                fertility_k=kf
            )
            new_s.save()
            
    def __fake_period():
        now_year = 2020
        
        for d_year in range(10):
            for pr in PER:
                if pr == 0 and d_year != 0: continue
                
                year = now_year + d_year
                s_date, e_date = _per_bound(year, pr)
                
                new_per = Period(
                    title=PER[pr],
                    start_date=s_date, 
                    end_date=e_date
                )
                new_per.save()
                
    def __fake_prognoses():
        fake_data = pd.read_csv(
            'api/utils/fake_data/seattle-weather.csv',
            sep=','
        )
        
        fake_data['date'] = pd.to_datetime(
            fake_data['date'],
            format = "%Y-%m-%d"
        )
        
        fake_data["year"] = fake_data['date'].dt.year
        fake_data["year"] += 8
        
        fake_data["month"] = fake_data['date'].dt.month
        fake_data["day"] = fake_data['date'].dt.day
        fake_data["temp_avg"] = fake_data['temp_max']
        
        for row in fake_data.to_dict(orient='records'):
            now_date = datetime(row['year'], row['month'], row['day'])
            
            period = Period.objects.filter(
                start_date__lte=now_date,
                end_date__gt=now_date
            ).order_by('-id')\
             .first()

            m_rep = Meteo_report(
                period=period,
                report_date=now_date,
                temperature=row['temp_avg'],
                precipitation=row['precipitation'],
                wind=row['wind'],
                weather=row['weather']
            )
            m_rep.save()

    print("STARTING LOAD FAKE DATA")
    __fake_cultures()
    __fake_soils()
    __fake_period()
    __fake_prognoses()


_add_fake_data()
"""for i in range(1, 13):
    print(((i + 1) % 12) // 3)
    
"""
    