from django.contrib import admin
from rangefilter.filters import NumericRangeFilter, DateRangeFilter


from .models import *

@admin.register(Culture)
class CultureAdmin(admin.ModelAdmin):
    list_display = ("name", "productivity_k", "Fav_temperature", "Fav_precipitation")
    list_filter = (
        ("productivity_k", NumericRangeFilter),
        ("fav_temp_bot", NumericRangeFilter),
        ("fav_temp_top", NumericRangeFilter),
        ("fav_precip_bot", NumericRangeFilter),
        ("fav_precip_top", NumericRangeFilter),
    )
    
    def Fav_temperature(self, model):
        return model.fav_temp_bot
    
    def Fav_precipitation(self, model):
        return f"{model.fav_precip_bot} - {model.fav_precip_top}"

@admin.register(Soil_quality)
class Soil_qualityAdmin(admin.ModelAdmin):
    list_display = ("title", "fertility_k")
    
@admin.register(Meteo_report)
class Meteo_reportAdmin(admin.ModelAdmin):
    list_display = ("period","report_date","temperature","precipitation","wind","weather")
    list_filter = (
        ("report_date", DateRangeFilter),
        ("temperature", NumericRangeFilter),
        ("precipitation", NumericRangeFilter),
        ("wind", NumericRangeFilter),
        "weather"
    )

@admin.register(Period)
class PeriodtAdmin(admin.ModelAdmin):
    list_display = ("title","start_date","end_date")
    list_filter = (
        "title",
        ("start_date", DateRangeFilter),
        ("end_date", DateRangeFilter)
    )

@admin.register(Regression_prognose)
class PeriodtAdmin(admin.ModelAdmin):
    list_display = ("period", "temp_avg", "prec_avg")
    
