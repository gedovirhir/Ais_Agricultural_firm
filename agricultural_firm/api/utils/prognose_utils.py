from typing import Union, List, NamedTuple, Tuple

from datetime import date, datetime, timedelta

import numpy as np
import torch 
from torch.nn import Linear, MSELoss
from torch.autograd import Variable

class synopticReportTuple(NamedTuple):
    date: Union[datetime, date]
    temperature: Union[int, float]

class synopticRegression(torch.nn.Module):
    def __init__(self) -> None:
        super(synopticRegression, self).__init__()
        self.linear1 = Linear(1, 1)
        #self.linear2 = Linear(10, 10)
        #self.linear3 = Linear(10, 1)
    
    def forward(self, x):
        if not isinstance(x, Variable):
            x = np.array([x], dtype=np.float32)
            x = Variable(torch.from_numpy(x))
        
        out = self.linear1(x)
        #out = self.linear2(out)
        #out = self.linear3(out)
        
        return out

def date_to_days_count(date_: Union[date, datetime]):
    year_start = date(year=date_.year, month=1, day=1)
    
    if isinstance(date_, datetime):
        date_ = date_.date()
    
    dates_delta = date_ - year_start
    
    return dates_delta.days

def get_finished_model(
    reports: List[synopticReportTuple]
) -> synopticRegression:
    x_train, y_train = zip(
        *[(date_to_days_count(r[0]), r[1])
            for r in reports]
    )
    
    x_train, y_train = np.array(x_train, dtype=np.float32).reshape(-1, 1),\
                       np.array(y_train, dtype=np.float32).reshape(-1, 1) 
    
    lr = 0.001
    epoch = 100
    
    model = synopticRegression()
    criterion = MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    """torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)"""
    
    for e in range(epoch):
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))
        
        optimizer.zero_grad()
        
        out = model(inputs)
        
        loss = criterion(out, labels)
        loss.backward()
        
        optimizer.step()
        
        #print('epoch {}, loss {}'.format(e, loss.item()))
    
    return (model, x_train, y_train)

def get_prognose_avg_temp(
    reports: List[synopticReportTuple],
    per_start: Union[date, datetime],
    per_end: Union[date, datetime],
):
    model = get_finished_model(reports)
    
    per_mid = per_end - per_start
    per_start = date_to_days_count(per_start)
    per_end = date_to_days_count(per_end)
    
    avg_temp = (model(per_start) + 
                model(per_mid.days) + 
                model(per_end)) / 3
    
    return avg_temp

def calculate_productivity(): ...



if __name__ == "__main__":
    temps = [22.9, 25.1, 25.7, 28, 32.3, 29.7, 25, 26.7, 31.9, 30.1,
    30.4, 21.7, 18.6, 21, 24.6, 27.7, 20.9, 22.9, 22.5, 25.6, 29.3,
    33, 31.8, 30.9, 32.4, 33.9, 33, 32.7, 27.2, 24.2, 24.4]
        
    days = [date(2022, 1, 1) + timedelta(days=i) for i in range(len(temps))]
        
    data = list(zip(days, temps))

    #get_prognose_avg_temp(data, date(2022, 1, 1), date.today())

    import matplotlib.pyplot as plt
    model, x_train, y_train = get_finished_model(data)

    with torch.no_grad(): # we don't need gradients in the testing phase
        if torch.cuda.is_available():
            predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
        else:
            predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
        print(predicted)

        plt.clf()
        plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
        plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
        plt.legend(loc='best')
        plt.show()   
    