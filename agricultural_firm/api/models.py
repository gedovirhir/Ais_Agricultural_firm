from django.db import models
from django.db.models import Q, F

class Culture(models.Model):
    
    name = models.CharField(max_length=100, unique=True)
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
    
    def __str__(self) -> str:
        return self.name

class Soil_quality(models.Model):
    
    title = models.CharField(
        max_length=80
    )
    fertility_k = models.FloatField()
    
    class Meta:
        db_table = "soil_qualities"
    
    def __str__(self):
        return self.title
    
class Period(models.Model):
    
    title = models.CharField(
        max_length=100,
        null=True
    )
    start_date = models.DateField()
    end_date = models.DateField()
    
    class Meta:
        db_table = 'periods'
    
    def __str__(self):
        year = self.start_date.year
        return f"{year} {self.title}"
    
class Meteo_report(models.Model):
    
    period = models.ForeignKey(
        Period,
        on_delete=models.CASCADE,
        related_name="meteo_report"
    )
    report_date = models.DateField()
    temperature = models.FloatField()
    precipitation = models.FloatField()
    wind = models.FloatField(null=True)
    weather = models.CharField(
        max_length=100,
        null=True
    )
    
    class Meta:
        db_table = 'meteo_reports'

class Regression_prognose(models.Model):
    period = models.ForeignKey(
        Period,
        on_delete=models.CASCADE
    )
    temp_avg = models.FloatField(null=True)
    prec_avg = models.FloatField(null=True)
    
    class Meta:
        db_table = 'regression_prognoses'
    
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