from rest_framework.serializers import (Serializer,
                                        CharField,
                                        FloatField,
                                        IntegerField, 
                                        DateField)

from .models import *

from datetime import datetime

class CultureSerializer(Serializer):
    id = IntegerField()
    name = CharField(max_length=100)
    productivity_k = FloatField()
    fav_temp_bot = FloatField()
    fav_temp_top = FloatField()
    fav_precip_bot = FloatField()
    fav_precip_top = FloatField()

class SoilSerializer(Serializer):
    id = IntegerField()
    title = CharField(max_length=80)
    fertility_k = FloatField()

class GetPrognoseSerializer(Serializer):
    culture_id = IntegerField()
    soil_type_id = IntegerField()
    sowing_area = FloatField()
    date = DateField(default=None, allow_null=True)
