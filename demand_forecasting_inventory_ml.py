def get_lag_list(data):          #this function creates the lag list depending on how big the data is
    import random
    import numpy as np
    import pandas as pd
    if len(data)<100:
        x = int((len(data)*20)/100)
        value = random.sample(range(1, x), 5)
        value = sorted(value)
        return value
    elif len(data)<1000:
        x = int((len(data)*3)/100)
        value = random.sample(range(1, x), 8)
        value = sorted(value)
        return value
    elif len(data)<10000:
        x = int((len(data)*.5)/100)
        value = random.sample(range(1, x), 10)
        value = sorted(value)
        return value
    elif len(data)<100000:
        x = int((len(data)*0.1)/100)
        value = random.sample(range(1, x), 12)
        value = sorted(value)
        return value
    elif len(data)<1000000:
        x = int((len(data)*0.02)/100)
        value = random.sample(range(1, x), 15)
        value = sorted(value)
        return value
    elif len(data)<10000000:
        x = int((len(data)*0.003)/100)
        value = random.sample(range(1, x), 15)
        value = sorted(value)
        return value
    else:
        x = int((len(data)*0.0005)/100)
        value = random.sample(range(1, x), 15)
        value = sorted(value)
        return value
    
    
def get_lags(lags, data):                             #this function adds the lag columns with mean and std
    grouping = data.groupby(['store','item'])
    columns = []
    for lag in lags:
        col_name = 'lag-'+str(lag)
        columns.append(col_name)
        data[col_name] = grouping.sales.shift(lag)

    from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
    holidays = calendar().holidays(start=data.date.min(), end=data.date.max())
    data['is_holiday'] = data.date.isin(holidays).astype(int)
    
    for lag in lags[2:5]:
        col_name = 'rolling_mean-'+str(lag)
        columns.append(col_name)
        data[col_name] = grouping.sales.shift(lag).rolling(window=7).mean()
    
    for lag in lags[2:5]:
        col_name = 'rolling_std-'+str(lag)
        columns.append(col_name)
        data[col_name] = grouping.sales.shift(lag).rolling(window=7).std()
    
    return data, columns


def get_sales_forecast(train, test):                #this function contains ml model to forecast demand
    from sklearn.compose import make_column_transformer
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler,StandardScaler
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np
    import pandas as pd
    test_id = test.id
    test.drop('id',axis = 1,inplace = True)  

    train['set'] = 'train'
    test['set']  = 'test'
    data = pd.concat([train,test])
    data = data.fillna(0)
    data.loc[:,'date'] = pd.to_datetime(data.loc[:,'date'])
    data['day_of_month']     = data.date.dt.day    #The day of the datetime.
    data['month_of_year']   = data.date.dt.month   #the month of the datetime
    data['year']    = data.date.dt.year            #the year of the datetime
    data['day_of_year'] = data.date.dt.dayofyear  #The ordinal day of the year.
    data['day_of_week'] = data.date.dt.dayofweek   #The dayofweek() function is used to get the day of the week. 
    #The day of the week with Monday=0, Sunday=6
    data['is_weekday'] = data['day_of_week'].apply(lambda x: 1 if x in (6,7) else 0)
    data['is_month_start']   = data.date.dt.is_month_start.map({False:0,True:1})
    data['is_month_end']     = data.date.dt.is_month_end.map({False:0,True:1})
    grouping = data.groupby(['store','item'])
    
    lags = get_lag_list(data)
    data, lag_columns = get_lags(lags, data)
    test  = data.loc[data.set  == 'test',:]
    train = data.loc[data.set == 'train',:].dropna()
    train.sales = np.log1p(train.sales)
    
    X = train.drop(['date','sales','set'],axis=1).dropna()
    y = train.sales    
    transformer = make_column_transformer(
        (OneHotEncoder(),['store','item','day_of_week']),
        (MinMaxScaler(), ['day_of_month','day_of_year']),
        (StandardScaler(),lag_columns),
        remainder = 'passthrough'
    )

    import xgboost as xgb

    regressor = xgb.XGBRegressor(n_estimators = 500,
                             max_depth = 5)
    
    pipeline = make_pipeline(transformer,regressor)
    print(X.columns)
    pipeline.fit(X,y)

    X_test = test.copy().drop(['date','sales','set'],axis = 1)
    pred_test = np.expm1(pipeline.predict(X_test))

    sub = pd.DataFrame({'id':test_id,'sales':np.round(pred_test)})
    return sub