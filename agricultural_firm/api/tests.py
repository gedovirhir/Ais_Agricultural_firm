from django.test import TestCase
from django.db import connection

from typing import Iterable

from .models import *

from .utils.db_api import _add_default_data
from .utils.db_api import * 

class TestDbAPI(TestCase):
    
    def setUp(self) -> None:
        _add_default_data()
        self.start_queries = len(connection.queries)
    
    def tearDown(self) -> None:
        pass
        #print(f"TOTAL QUERIES: {len(connection.queries) - self.start_queries}")
        
    def test_get_all_cultures(self):
        cults = get_all_cultures()
        
        print("\nCULTURES: ", end=" ")
        for c in cults:
            print(c.name, end=" ")
    
    def test_get_all_soils(self):
        soils = get_all_soils()

        print("\nSOILS: ", end=" ")
        for c in soils:
            print(c.title, end=" ")
    
    def test_get_actual_period(self):
        period = get_actual_period()
        
        print(f'\nACTUAL PERIOD: {period.title}, start - {period.start_date}; end - {period.end_date}')
    
    def test_get_all_previous_periods(self):
        period = get_actual_period()
        
        prev = get_all_previous_periods(period.start_date)
        
        print("\nPREV PERIODS:", end=" ")
        for p in prev:
            print(p.title, end=" ")

    def test_get_period_meteo_reports(self):
        n_per = get_actual_period()
        p_per = get_all_previous_periods(n_per.start_date).first()
        
        n_per_rep = get_period_meteo_reports(n_per.id)
        p_per_rep = get_period_meteo_reports(p_per.id)
        
        print("\nNOW PERIOD REPORTS:")
        for r in n_per_rep:
            print(f'{r.period.title} - t {r.temperature} - p {r.precipitation}')
        print("PREV PERIOD REPORTS:")
        for r in p_per_rep:
            print(f'{r.period.title} - t {r.temperature} - p {r.precipitation}')
        
        
        
