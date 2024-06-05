import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler 		#to normalize data
from custom_objects import LSTMModel, smape

# Standard scalar normalization
def normalize_data(df, model):
    cols_normalize = df.columns.difference(['id','time_cycles','RUL'])
    norm_df = pd.DataFrame(model.scaler.transform(df[cols_normalize]),
                             columns=cols_normalize,
                             index=df.index)
    join_df = df[df.columns.difference(cols_normalize)].join(norm_df)
    df = join_df.reindex(columns = df.columns)
    return df
        
# Data ETL
def etl(df_path, model, ids):
    index_names = ['id', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
    col_names = index_names + setting_names + sensor_names
        
    try:
        df = pd.read_csv(df_path,sep='\s+',header=None,index_col=False,names=col_names)
    except:
        print("No file was found or it's structure is not as expected")
        
    df = df[df['id'].isin(ids)]
    df = df.sort_values(['id','time_cycles'])
    
    # drop the constants features and the settings based on the EDA
    list_columns_droped = ['setting_1', 'setting_2', 'setting_3', 's_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    df.drop(columns=list_columns_droped, inplace=True)
        
    # data normalization
    # it needs to use the saved model
    df = normalize_data(df, model)
        
    print("ETL completed!")
    return df
    
    
if __name__ == "__main__":
    model_path = './trained_models/lstm_pipeline_v2_2024-06-05_09_00_10.pkl'
    test_path  = './dataset/CMaps/test_FD001.txt'
    
    with open(model_path, 'rb') as file:
        lstm_model = pickle.load(file)
    
    ids = [31, 34, 35, 36, 66, 68, 76, 81, 82, 100]
    looking_back = 5
    
    df_test = etl(test_path, lstm_model, ids)
    
    x_test =np.concatenate(list(list(lstm_model.get_window(df_test[df_test['id']==unit], looking_back, lstm_model.features)) for unit in df_test['id'].unique()))
    y_pred = lstm_model.model.predict(x_test)
    print(y_pred)
