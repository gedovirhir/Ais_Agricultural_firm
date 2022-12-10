from django.db import models
from django.db.models import Q, F

class Culture(models.Model):
    
    name = models.CharField(max_length=100)
    productivity_k = models.FloatField()
    
    fav_temp_bot = models.FloatField(default=F('fav_temp_top'))
    fav_temp_top = models.FloatField(default=F('fav_temp_bot'))
    
    fav_precip_bot = models.FloatField(default=F('fav_precip_top'))
    fav_precip_top = models.FloatField(default=F('fav_precip_bot'))
    
    class Meta:
        db_table = 'cultures'
        
        constraints = [
            models.CheckConstraint(
                check=Q(fav_temp_bot__lte=F('fav_temp_top')),
                name='fav_temp_bot_top_check_const'
            ),
            models.CheckConstraint(
                check=Q(fav_precip_bot__lte=F('fav_precip_top')),
                name='fav_precip_bot_top_check_const'
            )
        ]

class Soil_quality(models.Model):
    
    title = models.CharField(
        max_length=80
    )
    fertility_k = models.FloatField()
    
    class Meta:
        db_table = "soil_qualities"
    
class Period(models.Model):
    
    title = models.CharField(
        max_length=100,
        null=True
    )
    start_date = models.DateField()
    end_date = models.DateField()
    
    class Meta:
        db_table = 'periods'
    
class Meteo_report(models.Model):
    
    period = models.ForeignKey(
        Period,
        on_delete=models.CASCADE
    )
    report_date = models.DateField()
    temperature = models.FloatField()
    precipitation = models.FloatField()
    
    class Meta:
        db_table = 'meteo_reports'

class Regression_prognoses(models.Model):
    period = models.ForeignKey(
        Period,
        on_delete=models.CASCADE
    )
    avg_temp = models.FloatField()
    avg_precip = models.FloatField()
    

"""
class Plot_culture(models.Model):
    D_SOWING = 100.0
    
    plot = models.ForeignKey(
        Plot,
        on_delete=models.CASCADE,
    )
    culture = models.ForeignKey(
        Culture,
        on_delete=models.CASCADE
    )
    sowing_percent = models.FloatField(
        default=D_SOWING
    )
    
    class Meta:
        db_table = 'plot_cultures'

class Productivity_report(models.Model):
    D_SOWING = 100.0
    
    plot_culture = models.ForeignKey(
        Plot_culture,
        on_delete=models.CASCADE
    )
    
    period = models.ForeignKey(
        Period,
        null=True,
        on_delete=models.SET_NULL,
    )
    amount = models.FloatField()
    report_date = models.DateTimeField(null=True)
    ...
    
    class Meta:
        db_table = 'productivity_reports'

class Business_rule(models.Model):
    
    informative = models.TextField()
    
    ...
    
    class Meta:
        db_table = 'business_rules'

class Plot(models.Model):

    square = models.FloatField()
    soil_quality = models.ForeignKey(
        Soil_quality,
        null=True,
        on_delete=models.SET_NULL
    )
    
    ...
    
    class Meta:
        db_table = 'plots' 
"""