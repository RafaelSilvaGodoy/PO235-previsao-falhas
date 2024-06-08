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
def etl(df_path, model):      
    try:
        df = pd.read_csv(df_path,sep=',')
    except:
        print("No file was found or it's structure is not as expected")
        
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
    model_path = './trained_models/lstm_pipeline_w5_v2_2024-06-08_15_56_34.pkl'
    test_path  = './dataset/test_set.csv'
    
    with open(model_path, 'rb') as file:
        lstm_model = pickle.load(file)
    
    df_test = etl(test_path, lstm_model)
    
    x_test =np.concatenate(list(list(lstm_model.get_window(df_test[df_test['id']==unit], lstm_model.window, lstm_model.features)) for unit in df_test['id'].unique()))
    y_pred = lstm_model.model.predict(x_test)
    print(y_pred)
