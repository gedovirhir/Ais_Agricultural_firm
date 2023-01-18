from typing import Union, List, NamedTuple, Tuple, Optional, Any, Collection

from datetime import date, datetime, timedelta
from random import randint

import numpy as np
import torch 
from torch.nn import Linear, MSELoss, ReLU
from torch.nn.functional import normalize, batch_norm
from torch.autograd import Variable

from ..models import Regression_prognose, Period

class synopticReportTuple(NamedTuple):
    date: Union[datetime, date]
    prev_year_flag: bool
    temperature: Union[int, float] 
    precipitation: Union[int, float]
    wind: Union[int, float]

class synopticRegression(torch.nn.Module):
    d_min = 0
    d_max = 365
    
    def __init__(self) -> None:
        super(synopticRegression, self).__init__()
        self.rl = ReLU()
        
        self.linear1 = Linear(2, 10)
        self.linear2 = Linear(10, 10)
        self.linear3 = Linear(10, 3)
    
    def __input_to_variable(self, input_: Tuple[int, int]) -> Variable:
        all_d = np.array([input_], dtype=np.float32)
        all_d = torch.tensor(all_d)
        all_d = Variable(all_d)
        
        return all_d
    
    def __day_normalize(self, var: torch.Tensor):
        return var / torch.tensor((self.d_max, 1))
    
    def forward(self, input_: Union[Variable, Tuple[int, int]]):
        x = input_
        
        if not isinstance(x, Variable):
            x = self.__input_to_variable(x)
        
        #Normalize
        x = self.__day_normalize(x)
        
        out = self.rl(self.linear1(x))
        out = self.rl(self.linear2(out))
        out = self.linear3(out)
        
        if not self.training:
            res = [
                float(b) for b in out[0]
            ]
            return res
        return out

def date_to_days_count(date_: Union[date, datetime]) -> int:
    year_start = date(year=date_.year, month=1, day=1)
    
    if isinstance(date_, datetime):
        date_ = date_.date()
    
    dates_delta = date_ - year_start
    
    return dates_delta.days

def reports_to_traindata(reports: List[synopticReportTuple]):
    x_train = []
    y_train = []
    
    for rep in reports:
        x_train.append(
            # days from year start, prev year flag
            (date_to_days_count(rep[0]), rep[1])
        )
        
        y_train.append((rep[2], rep[3], rep[4]))
        
    x_train, y_train = np.array(x_train, dtype=np.float32),\
                       np.array(y_train, dtype=np.float32)

    x_train, y_train = torch.tensor(x_train), torch.tensor(y_train)
    
    return (x_train, y_train)

def get_finished_model(
    reports: List[synopticReportTuple]
) -> synopticRegression:
    model = synopticRegression()
    """model_p = "static/weather_model.nth"
    try: 
        model.load_state_dict(torch.load(model_p))
        model.eval()
        return model
    except Exception as ex:
        pass"""
    
    x_train, y_train = reports_to_traindata(reports)
    
    lr = 0.001
    epoch = 150
    batch_size = 64
    batch_count = len(x_train) // batch_size
    
    criterion = MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    """torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)"""
    
    for e in range(epoch):
        for batch_i in range(batch_count):
            b_bound = batch_i * batch_size
            
            x_batch = x_train[b_bound : b_bound + batch_size]
            y_batch = y_train[b_bound : b_bound + batch_size]
            
            inputs = Variable(x_batch)
            labels = Variable(y_batch)

            out = model(inputs)

            loss = criterion(out, labels)
            optimizer.zero_grad() 
            loss.backward() 

            optimizer.step()
            
            #print('epoch {}, loss {}'.format(e, loss.item()))
    
    model.eval()
    
    return model

def get_prognose_avg(
    reports: List[synopticReportTuple],
    per_start: Union[date, datetime],
    per_end: Union[date, datetime],
) -> Tuple[float, float]:
    model = get_finished_model(reports)
    
    per_mid = per_end - per_start
    per_start = date_to_days_count(per_start)
    per_end = date_to_days_count(per_end)
    """
    model -> (temperature, precipitation, wind)
    """
    prognoses = [model((per_start, 0)),
                 model((per_mid.days // 2, 0)),
                 model((per_end, 0))]
    
    avg_temp = 0
    avg_prec = 0
    
    for pr in prognoses:
        avg_temp += pr[0]
        avg_prec += pr[1]
    
    avg_temp, avg_prec = avg_temp / len(prognoses), avg_prec / len(prognoses)
    
    return (avg_temp, avg_prec)

def culture_prod_coef(
    culture_recommended_bound: Tuple[float, float],
    real: float
):
    if real < culture_recommended_bound[0]:
        coef = 1 - (1 - real / culture_recommended_bound[0])
    elif real > culture_recommended_bound[1]:
        coef = 1 + (1 - real / culture_recommended_bound[1])
    else:
        coef = 1
    
    if coef < 0:
        return 0
    
    return coef

def calculate_productivity_on_period(
    c_coef: float,
    c_fav_temp_bound: Tuple[float, float],
    c_fav_prec_bound: Tuple[float, float],
    soil_coef: float,
    sowing_area: float,
    reports: List[synopticReportTuple],
    per_start: Optional[Union[date, datetime]] = None,
    per_end: Optional[Union[date, datetime]] = None,
) -> Tuple[float, float]: 
    if not (per_start and per_end):
        per_start = reports[0][0]
        per_end = reports[-1][0]
    
    period_ = Period.objects.filter(
        start_date=per_start,
        end_date=per_end
    ).first()
    
    exist_prognose = Regression_prognose.objects.filter(period=period_).first()
    
    if exist_prognose:
        temp_avg, prec_avg = exist_prognose.temp_avg, exist_prognose.prec_avg
    else:
        temp_avg, prec_avg = get_prognose_avg(
            reports, 
            per_start,
            per_end
        )
        new_progn = Regression_prognose(
            period=period_,
            temp_avg=temp_avg,
            prec_avg=prec_avg
        )
        new_progn.save()
    
    temp_c = culture_prod_coef(
        c_fav_temp_bound,
        temp_avg
    )
    prec_c = culture_prod_coef(
        c_fav_prec_bound,
        prec_avg
    )
    
    prec_c_buff = 1.5
    prec_c *= abs(-(prec_c - 0.5)**3 + 1)**2 
    weather_coef = temp_c * prec_c + 0.0001
    
    productivity = c_coef * sowing_area
    productivity *= soil_coef * weather_coef
    productivity = round(productivity, 2)
    
    prognose = {
        "temp_avg": round(temp_avg, 1),
        "prec_avg": round(prec_avg, 1),
        "prod": round(productivity, 2),
        "weather_k": round(weather_coef, 1)
    }
    
    return prognose
    
if __name__ == "__main__":
    temps = [22.9, 25.1, 25.7, 28, 32.3, 29.7, 25, 26.7, 31.9, 30.1,
    30.4, 21.7, 18.6, 21, 24.6, 27.7, 20.9, 22.9, 22.5, 25.6, 29.3,
    33, 31.8, 30.9, 32.4, 33.9, 33, 32.7, 27.2, 24.2, 24.4]
    
    mx = 33.9
    prec = [i / mx for i in temps]
        
    days = [date(2022, 12, 20) + timedelta(days=i) for i in range(len(temps))]
        
    data = list(zip(days, temps, prec))

    #get_prognose_avg_temp(data, date(2022, 1, 1), date.today())

    d = calculate_productivity_on_period(
        12, (20, 25), (0.5, 0.8), 0.9, 20, data
    )
    print(d)
    