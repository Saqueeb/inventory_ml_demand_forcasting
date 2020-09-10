import numpy as np
import pandas as pd
from demand_forecasting import*
train = pd.read_csv(r'C:\Users\Zia.Accelx\Desktop\Demand_Forecast/train_test.csv')
test = pd.read_csv(r'C:\Users\Zia.Accelx\Desktop\Demand_Forecast/test_test.csv')
result = get_sales_forecast(train, test)
print(result)