#========================== 0. import library ==========================#

# 아래 패키지들 다 쓴건지 최종 확인해야함
import glob
import os
import pandas as pd
import numpy as np
import sklearn as skl
import sys
import sktime
import sklearn
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error
import random as rn
import tqdm

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)
from sktime.forecasting.model_selection import temporal_train_test_split

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
from sklearn.model_selection import  GridSearchCV

print("-------------------------- Python & library version --------------------------")
print("Python version: {}".format(sys.version))
print("pandas version: {}".format(pd.__version__))
print("numpy version: {}".format(np.__version__))
print("sktime version: {}".format(sktime.__version__))
print("xgboost version: {}".format(xgb.__version__))
print("scikit-learn version: {}".format(skl.__version__))
print("------------------------------------------------------------------------------")


#========================== 1. 데이터 병합 ==========================#

# 1-(1) 20220101~20231031 데이터 병합
folder_path = "C:/Users/USER/OneDrive/바탕 화면/통계학과 공모전 최최종_VER2/Original"
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
dataframes = [pd.read_csv(file, encoding='UTF-8', header=None) for file in csv_files]
df1 = pd.concat(dataframes, ignore_index=True)
df1 = df1.drop(columns=[0,4,5,6,8,9])
df1.columns = ['Date', 'Time', 'Station_num', 'Passenger']
print(df1)
# 1-(2) 20231101~20240531 데이터 병합
folder_path = "C:/Users/USER/OneDrive/바탕 화면/통계학과 공모전 최최종_VER2/After"
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
dataframes = [pd.read_csv(file, encoding='cp949', header=0) for file in csv_files]
df2 = pd.concat(dataframes, ignore_index=True)
df2= df2.drop(columns=["strd_yymm","swst_nm","swst_lgd_cdn_val","swst_ltd_cdn_val"])
df2.columns = ['Date', 'Time', 'Station_num', 'Passenger']
print(df2)
# 1-(3) 최종 데이터 병합
df = pd.concat([df1,df2], ignore_index=True)
df = df.sort_values(by=['Station_num', 'Date', 'Time'])
print(df)







#========================== 2. 데이터 전처리 ==========================#

df = df[~df['Time'].isin([1, 2, 3])] # 새벽 1시,2시,3시인 값들은 대부분 1에 해당하기에 드랍
print(df)

# 2-(1) 시간 변수 생성(datetime, year, month, day, weekday, hour)
df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + df['Time'].astype(str).str.zfill(2), format='%Y%m%d%H')
df['year'] = df['DateTime'].dt.year
df['month'] = df['DateTime'].dt.month
df['day'] = df['DateTime'].dt.day
df['weekday'] = df['DateTime'].dt.weekday
df['hour'] = df['DateTime'].dt.hour


# 2-(2) 시각화 결과에 따른 301번 역 drop
df = df[df["Station_num"] != 301]

# 2-(3) 95~99, 202번 역의 시작일 확인결과 2022년 4월1일부터 시작하는 것으로 확인
df = df[df['DateTime'] >= '2022-04-01']

# 2-(4) 휴무일 변수 생성
df = df.drop(columns=['Time'])
df = df.drop(columns=['DateTime'])
df['holiday'] = np.where(df["weekday"] >= 5, True, False)

df['Date'] = df['Date'].astype(str)
holidays = ['20220131','20220201','20220202','20220301','20220309','20220505',
            '20220601','20220606','20220815','20220909','20220912',
            '20221003','20221010','20230123','20230124','20230301',
            '20230505','20230529','20230606','20230815','20230928','20230929',
            '20231002','20231003','20231009','20231225','20240101','20240209',
            '20240212','20240301','20240410','20240506','20240515',"20240606"]  

df['holiday'] = df['Date'].isin(holidays) | df['holiday']


# 2-(5) 지하철역별, 요일별, 시간별 탑승객 평균과 표준편차 변수 생성
day_hour_mean = pd.pivot_table(df, values='Passenger', index=['Station_num', 'hour', 'weekday'], aggfunc=np.mean).reset_index()
day_hour_mean.columns = ['Station_num', 'hour', 'weekday', 'day_hour_mean']

day_hour_std = pd.pivot_table(df, values='Passenger', index=['Station_num', 'hour', 'weekday'], aggfunc=np.std).reset_index()
day_hour_std.columns = ['Station_num', 'hour', 'weekday', 'day_hour_std']

hour_mean = pd.pivot_table(df, values='Passenger', index=['Station_num', 'hour'], aggfunc=np.mean).reset_index()
hour_mean.columns = ['Station_num', 'hour', 'hour_mean']

hour_std = pd.pivot_table(df, values='Passenger', index=['Station_num', 'hour'], aggfunc=np.std).reset_index()
hour_std.columns = ['Station_num', 'hour', 'hour_std']

df = df.merge(day_hour_mean, on=['Station_num', 'hour', 'weekday'], how='left')
df = df.merge(day_hour_std, on=['Station_num', 'hour', 'weekday'], how='left')
df = df.merge(hour_mean, on=['Station_num', 'hour'], how='left')
df = df.merge(hour_std, on=['Station_num', 'hour'], how='left')


na_rows = df[df.isna().any(axis=1)]
print(na_rows) #행이 1개인 값들에 대해 std가 NA인 행들이 발견되어 아래와 같이 0으로 보간
df = df.fillna(value=0)
df.isna().sum()






#========================== 3. 이상치 처리 ==========================#

# 6시그마를 기준으로 넘는 값을 6시그마로 변경
up_w = df['day_hour_mean'] + 3*df['day_hour_std']
lo_w = df['day_hour_mean'] - 3*df['day_hour_std']
up_d = df['hour_mean'] + 3*df['hour_std']
lo_d = df['hour_mean'] - 3*df['hour_std']

print(len(df.loc[(df['Passenger'] > up_w) | (df['Passenger'] < lo_w)]))
print(len(df.loc[(df['Passenger'] > up_d) | (df['Passenger'] < lo_d)]))

df['Passenger'] = np.where(df['Passenger'] > up_w, up_w, df['Passenger'])
df['Passenger'] = np.where(df['Passenger'] < lo_w, lo_w, df['Passenger'])






#========================== 4. MODELING ==========================#



######################## 4-(1) XGBoost ########################
# 아래 최적 파라미터 코드의 경우 GPU 4070 ti super에서 대략 50분 걸렸습니다
# 4-1-1) Parameter 튜닝
"""
df2 = df.drop(columns=["Date"])
stations = df2['Station_num'].unique()
hours = df2['hour'].unique()
kf = KFold(n_splits = 7,shuffle=True,random_state=RANDOM_SEED)

def RMSE(y_test, pred):
    return np.sqrt(sum((y_test-pred)*(y_test-pred)))
params_record = []

df3 = pd.DataFrame(columns = ['n_estimators', 'eta', 'min_child_weight','max_depth', 'colsample_bytree', 'subsample'])
preds = np.array([])

grid = {'n_estimators' : [100], 'eta' : [0.1], 'min_child_weight' : np.arange(1, 8, 1), 
        'max_depth' : [6] , 'colsample_bytree' :[1], 
        'subsample' :np.arange(0.8, 1.0, 0.1)} 
        
for i in tqdm.tqdm(stations):
    station_data = df2[df2['Station_num'] == i]
    y = station_data[['Passenger']]
    x = station_data.drop(columns=["Passenger", 'Station_num'])
    y_train, y_test, x_train, x_test = temporal_train_test_split(y = y, X = x, test_size = 0.25)
    
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        seed=RANDOM_SEED,
        gpu_id=1,
        tree_method='gpu_hist',
        predictor='gpu_predictor'
    )

    gcv = GridSearchCV(
        estimator=model,
        param_grid=grid,
        scoring='neg_root_mean_squared_error',
        cv=KFold(n_splits=7, shuffle=True, random_state=RANDOM_SEED),
        refit=True,
        verbose=1
    )
    
    gcv.fit(x_train, y_train)

    best_params = gcv.best_params_
    print(f"Best parameters for station {i}: {best_params}")
    pred = gcv.predict(x_test)
    rmse_value = RMSE(y_test["Passenger"].reset_index(drop=True), pd.Series(pred))
    print(f"Station {i} || RMSE: {rmse_value:.4f}")
    
    params_record.append(best_params)

df3 = pd.DataFrame(params_record)
df3["stations"]=stations





# 4-1-2) 최적 max_depth
depth_list = []
rmse_list = []
for i in tqdm.tqdm(stations):
    station_data = df2[df2['Station_num'] == i]
    y = station_data[['Passenger']]
    x = station_data.drop(columns=["Passenger", 'Station_num'])
    y_train, y_test, x_train, x_test = temporal_train_test_split(y = y, X = x, test_size = 0.25)
    params_row = df3[df3['stations'] == i]    

    xgb = XGBRegressor(seed = 0,
                        n_estimators = params_row['n_estimators'].iloc[0],
                        eta = params_row["eta"].iloc[0],
                        min_child_weight = params_row['min_child_weight'].iloc[0],
                        max_depth = params_row['max_depth'].iloc[0], 
                        colsample_bytree = params_row['colsample_bytree'].iloc[0], 
                        subsample = params_row['subsample'].iloc[0] )
    
    xgb.fit(x_train, y_train)
    pred0 = xgb.predict(x_test)
    max_depth = 6
    score0 = RMSE(y_test["Passenger"].reset_index(drop=True), pd.Series(pred0))
    
    for j in [1,2,3,4,5,6,7,8,9]:
        xgb = XGBRegressor(
            seed=0,
            n_estimators = params_row['n_estimators'].iloc[0],
            eta = params_row["eta"].iloc[0],
            min_child_weight=params_row['min_child_weight'].iloc[0],
            max_depth=j,
            colsample_bytree=params_row['colsample_bytree'].iloc[0],
            subsample=params_row['subsample'].iloc[0],
            objective='reg:squarederror'  
        )

        xgb.fit(x_train, y_train)
        pred1 = xgb.predict(x_test)
        score1 = RMSE(y_test["Passenger"].reset_index(drop=True), pd.Series(pred1))
        if score1 < score0:
            max_depth = j
            score0 = score1
    
    depth_list.append(max_depth)
    rmse_list.append(score0)
    print("station {} || best score : {} || best_depth : {}".format(i, score0, max_depth))

df3["best_depth"]=depth_list




# 4-1-3) Best iteration
df2 = df.drop(columns=["Date"])
df3 = df3.drop(columns=["max_depth"])

stations = df2['Station_num'].unique()
def RMSE(y_test, pred):
    return np.sqrt(sum((y_test-pred)*(y_test-pred)))
params_record = []
scores = []   
best_it = []  
for i in tqdm.tqdm(stations):
    station_data = df2[df2['Station_num'] == i]
    y = station_data[['Passenger']]
    x = station_data.drop(columns=["Passenger", 'Station_num'])
    y_train, y_test, x_train, x_test = temporal_train_test_split(y = y, X = x, test_size = 0.25)

    params_row = df3[df3['stations'] == i]    
    xgb_reg = XGBRegressor(n_estimators = 10000, eta = 0.01, min_child_weight = params_row['min_child_weight'].iloc[0], 
                           max_depth = params_row['best_depth'].iloc[0], colsample_bytree = params_row['colsample_bytree'].iloc[0], 
                           subsample = params_row['subsample'].iloc[0], seed=0,
        objective='reg:squarederror',early_stopping_rounds=300, 
    )
    xgb_reg.fit(x_train, y_train, eval_set=[(x_train, y_train), 
                                            (x_test, y_test)],verbose=False)
    y_pred = xgb_reg.predict(x_test)
    pred = pd.Series(y_pred)   
    
    sm = RMSE(y_test["Passenger"], pd.DataFrame(pred))
    scores.append(sm)
    best_it.append(xgb_reg.best_iteration) 

df3["best_it"]=best_it


# 4-1-4) alpha값 튜닝
alpha_list = []
rmse_list = []
for i in tqdm.tqdm(stations):
    station_data = df2[df2['Station_num'] == i]
    y = station_data[['Passenger']]
    x = station_data.drop(columns=["Passenger", 'Station_num'])
    y_train, y_test, x_train, x_test = temporal_train_test_split(y = y, X = x, test_size = 0.25)
    params_row = df3[df3['stations'] == i]    

    xgb = XGBRegressor(seed = 0,
                      n_estimators = params_row["best_it"].iloc[0], eta = 0.01, min_child_weight = params_row['min_child_weight'].iloc[0],
                           max_depth = params_row['best_depth'].iloc[0], colsample_bytree = params_row['colsample_bytree'].iloc[0], 
                           subsample = params_row['subsample'].iloc[0] )
    
    xgb.fit(x_train, y_train)
    pred0 = xgb.predict(x_test)
    best_alpha = 0
    score0 = RMSE(y_test["Passenger"].reset_index(drop=True), pd.Series(pred0))
    
    for j in [1, 3, 5, 7, 10, 25, 50, 75, 100]:
        xgb = XGBRegressor(
            seed=0,
            n_estimators=params_row["best_it"].iloc[0],
            eta=0.01,
            min_child_weight=params_row['min_child_weight'].iloc[0],
            max_depth=params_row['best_depth'].iloc[0],
            colsample_bytree=params_row['colsample_bytree'].iloc[0],
            subsample=params_row['subsample'].iloc[0],
            reg_alpha=j,
            objective='reg:squarederror'  # Corrected the objective parameter
        )

        xgb.fit(x_train, y_train)
        pred1 = xgb.predict(x_test)
        score1 = RMSE(y_test["Passenger"].reset_index(drop=True), pd.Series(pred1))
        if score1 < score0:
            best_alpha = j
            score0 = score1
    
    alpha_list.append(best_alpha)
    rmse_list.append(score0)
    print("station {} || best score : {} || alpha : {}".format(i, score0, best_alpha))

df3["aplha"]=alpha_list




# 4-1-5) lambda값 튜닝
lambda_list = []
rmse_list = []
for i in tqdm.tqdm(stations):
    station_data = df2[df2['Station_num'] == i]
    y = station_data[['Passenger']]
    x = station_data.drop(columns=["Passenger", 'Station_num'])
    y_train, y_test, x_train, x_test = temporal_train_test_split(y = y, X = x, test_size = 0.25)
    params_row = df3[df3['stations'] == i]    

    xgb = XGBRegressor(seed = 0,
                      n_estimators = params_row["best_it"].iloc[0], eta = 0.01, min_child_weight = params_row['min_child_weight'].iloc[0],
                           max_depth = params_row['best_depth'].iloc[0], colsample_bytree = params_row['colsample_bytree'].iloc[0], 
                           subsample = params_row['subsample'].iloc[0],reg_alpha=params_row['aplha'].iloc[0] )
    
    xgb.fit(x_train, y_train)
    pred0 = xgb.predict(x_test)
    best_lambda = 0
    score0 = RMSE(y_test["Passenger"].reset_index(drop=True), pd.Series(pred0))
    
    for j in [0, 0.01, 0.1, 1, 10, 100]:
        xgb = XGBRegressor(
            seed=0,
            n_estimators=params_row["best_it"].iloc[0],
            eta=0.01,
            min_child_weight=params_row['min_child_weight'].iloc[0],
            max_depth=params_row['best_depth'].iloc[0],
            colsample_bytree=params_row['colsample_bytree'].iloc[0],
            subsample=params_row['subsample'].iloc[0],
            reg_lambda=j,
            objective='reg:squarederror'  # Corrected the objective parameter
        )

        xgb.fit(x_train, y_train)
        pred1 = xgb.predict(x_test)
        score1 = RMSE(y_test["Passenger"].reset_index(drop=True), pd.Series(pred1))
        if score1 < score0:
            best_lambda = j
            score0 = score1
    
    lambda_list.append(best_lambda)
    rmse_list.append(score0)
    print("station {} || best score : {} || lambda : {}".format(i, score0, best_lambda))

df3["lambda"]=lambda_list
df3.to_csv('hyperparameter_XGB_FINAL.csv', index = False)
"""


















#========================== 5. Result ==========================#
# 5-(1) 파일 불러들이기 및 칼럼명 변경
result = pd.read_csv("C:/Users/USER/OneDrive/바탕 화면/통계학과 공모전 최최종_VER2/result.csv",encoding="cp949")
result = result.drop(columns=['V1','V5', 'V6', 'V7', 'V8'])
result.columns = ['Date', 'Time', 'Station_num']

# 5-(2) 시간 변수 생성(datetime, year, month, day, weekday, hour)
result['DateTime'] = pd.to_datetime(result['Date'].astype(str) + result['Time'].astype(str).str.zfill(2), format='%Y%m%d%H')
result['year'] = result['DateTime'].dt.year
result['month'] = result['DateTime'].dt.month
result['day'] = result['DateTime'].dt.day
result['weekday'] = result['DateTime'].dt.weekday
result['hour'] = result['DateTime'].dt.hour

# 5-(3) 휴무일 변수 생성
result = result.drop(columns=['Time'])
result = result.drop(columns=['DateTime'])
result['holiday'] = np.where(result["weekday"] >= 5, True, False)


# 5-(4) 지하철역별, 요일별, 시간별 탑승객 평균과 표준편차 변수 생성
result = result.merge(day_hour_mean, on=['Station_num', 'hour', 'weekday'], how='left')
result = result.merge(day_hour_std, on=['Station_num', 'hour', 'weekday'], how='left')
result = result.merge(hour_mean, on=['Station_num', 'hour'], how='left')
result = result.merge(hour_std, on=['Station_num', 'hour'], how='left')

na_rows = result[result.isna().any(axis=1)]
result = result.fillna(value=0)
result.isna().sum()


# 5-(5) Best hyperparameters 불러오기 
xgb_params = pd.read_csv('C:/Users/USER/OneDrive/바탕 화면/통계학과 공모전 최최종_VER2/hyperparameter_XGB_FINAL.csv')
rs = result.sort_values(by=['Station_num','day','hour'])


# 5-(6) Model training 및 predict
preds = np.array([]) 
df2 = df.drop(columns=["Date"])
stations = df2['Station_num'].unique()

for i in tqdm.tqdm(stations):
    pred_list = []
    station_data = df2[df2['Station_num'] == i]
    y = station_data[['Passenger']]
    x = station_data.drop(columns=["Passenger", 'Station_num'])
    params_row = xgb_params[xgb_params['stations'] == i]

    for seed in [0,1,2,3,4,5]:  # 각 시드별 예측
        xgb = XGBRegressor(seed=seed,
                           n_estimators=params_row["best_it"].iloc[0],
                           eta=0.1,
                           min_child_weight=params_row['min_child_weight'].iloc[0],
                           max_depth=params_row['best_depth'].iloc[0],
                           colsample_bytree=params_row['colsample_bytree'].iloc[0],
                           subsample=params_row['subsample'].iloc[0],
                           reg_alpha=params_row['aplha'].iloc[0],  # Fixed typo here
                           reg_lambda=params_row['lambda'].iloc[0])
    
        result_data = rs[rs['Station_num'] == i]
        result_data = result_data.drop(columns=['Station_num','Date'])
        xgb.fit(x, y)
        y_pred = xgb.predict(result_data)
        pred_list.append(y_pred)
    
    pred_df = pd.DataFrame(pred_list).T
    pred = pred_df.mean(axis=1)
    preds = np.append(preds, pred)
    
result["preds"]=preds

rs = result.sort_values(by=['Station_num','day','hour'])
rs['preds'] = preds
result = rs.sort_index()


# 5-(7) Post Processing(후처리)
rs=result
rs['index'] = range(0,len(rs))
day_hour_min = pd.pivot_table(df, values='Passenger', index=['Station_num', 'hour', 'weekday'], aggfunc=np.min).reset_index()
day_hour_max = pd.pivot_table(df, values='Passenger', index=['Station_num', 'hour', 'weekday'], aggfunc=np.max).reset_index()

rs = rs.merge(day_hour_min, on=['Station_num', 'hour', 'weekday'])
rs = rs.merge(day_hour_max, on=['Station_num', 'hour', 'weekday'])

rs['preds'] = np.where(rs['preds'] > rs['Passenger_y'], rs['Passenger_y'], rs['preds'])
rs['preds'] = np.where(rs['preds'] < rs['Passenger_x'], rs['Passenger_x'], rs['preds'])

rs.drop(columns=['Passenger_x', 'Passenger_y'], inplace=True)


rs.set_index('index', inplace=True)
rs.sort_index(inplace=True)


# 5-(8) 시간대 3인 값 변경
final = pd.read_csv("C:/Users/USER/OneDrive/바탕 화면/통계학과 공모전 최최종_VER2/result.csv",encoding="cp949")
final['V8'] = rs['preds']
final.loc[final['V3'] == 3, 'V8'] = 1
final = final.fillna(0)

final.to_csv('C:/Users/USER/OneDrive/바탕 화면/통계학과 공모전 최최종_VER2/result_submit_xgb_FINAL.csv', index = False,encoding="cp949")