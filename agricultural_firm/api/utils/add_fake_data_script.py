import os
import sys

sys.path.append(os.path.abspath('agricultural_firm/api'))

from ..models import *

import pandas
from datetime import datetime, timedelta
from random import randint

def _from_f_to_c(temperature: float, rnd_c: int = 2):
    return round(temperature * 3/9, rnd_c)

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
    
    s_m = periods[per_i][0]
    e_m = periods[per_i][1] + 1
    
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
                 'Кабачки', 'Цукини', 'Патиссон', 'Крукнек', 'Крукнек', 'Люффа']
        
        for n in names:
            fav_temp_bot = randint(10, 20)
            fav_temp_top = fav_temp_bot + randint(5, 10)
            fav_precip_bot = randint(5, 30)
            fav_precip_top = fav_precip_bot + randint(5, 20)
            
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
        now_year = 2022
        
        for d_year in range(10):
            for pr in PER:
                year = now_year + d_year
                s_date, e_date = _per_bound(year, pr)
                
                new_per = Period(
                    title=PER[pr],
                    start_date=s_date, 
                    end_date=e_date
                )
                new_per.save()
                
    def __fake_prognoses():
        fake_data = pandas.read_csv(
            'api/utils/fake_data/city_temperature.csv',
            sep=','
        )
        fake_data = fake_data.loc[(fake_data['Region'] == 'Africa') & 
                         (fake_data['Country'] == 'Algeria') &
                         (fake_data['Year'] > 2000), ['Month', 'Day', 'Year', 'AvgTemperature']]

        for year in fake_data['Year'].unique():
            periods = {}
            
            # (12 1 2) (3 4 5) (6, 7, 8) (9, 10, 11)
            for _, pr in fake_data.loc[fake_data['Year'] == year].iterrows():
                m_i = pr['Month']
                d_i = pr['Day']
                per_i = _per_by_month(m_i)
                temp = _from_f_to_c(pr['AvgTemperature'])
                temp_p = 0
                
                if not periods.get(per_i):
                    s_date, e_date = _per_bound(year, per_i)
                    per_o = Period(
                        title=PER[per_i],
                        start_date=s_date, 
                        end_date=e_date
                    )
                    per_o.save()
                    
                    periods.update(
                        {per_i: per_o}
                    )
                    
                    
                    temp_p = fake_data.loc[
                        ((per_o.start_date.year <= fake_data['Year']) & (fake_data['Year'] <= per_o.end_date.year)) &
                        (fake_data['Month'] >= per_o.start_date.month - 12 * (fake_data['Year'] - per_o.start_date.year)) &
                        (fake_data['Month'] < per_o.end_date.month + 12 * (fake_data['Year'] - per_o.start_date.year)),
                        'AvgTemperature'
                    ]
                    avg_temp_p = temp_p.mean()
                    avg_temp = _from_f_to_c(avg_temp_p)
                
                #s_d = datetime(year=year, month=0*3 + 1, day=1)
                #e_d = datetime(year=year, month=0*3 + 3, day=1)
                
                prec = round(
                    randint(10, 30) * (avg_temp / temp),
                    2
                )
                report_date = datetime(
                    int(year), 
                    int(m_i),
                    int(d_i)
                )
                
                m_rep = Meteo_report(
                    period=periods[per_i],
                    report_date=report_date,
                    temperature=temp,
                    precipitation=prec
                )
                m_rep.save()
    
    print("STARTING LOAD FAKE DATA")
    __fake_cultures()
    __fake_soils()
    __fake_prognoses()
    __fake_period()


"""for i in range(1, 13):
    print(((i + 1) % 12) // 3)
    
"""
if __name__ == "__main__":
    _add_fake_data()