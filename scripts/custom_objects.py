import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from glob import glob
from json import dumps
from datetime import datetime
from keras import backend as k

#to split the train set in train and val sets using the id
from sklearn.model_selection import GroupShuffleSplit		

from sklearn.preprocessing import StandardScaler 		#to normalize data

#for deep learning
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.callbacks import History, EarlyStopping


# training loss function
class smape(tf.keras.losses.Loss):
    def __init__(self, epsilon=1e-8, name="smape", **kwargs):
        super().__init__(name=name, **kwargs)
        self.epsilon = epsilon
        
    def call(self,y_true,y_pred):
        y_true = tf.cast(y_true, tf.float32)
        numerator = abs(y_pred - y_true)
        denominator = (abs(y_true) + abs(y_pred)) / 2.0 + self.epsilon
        smape_value = k.mean(numerator / denominator) * 100
        return smape_value
        
    def get_config(self):
        config = {
            'epsilon': self.epsilon
        }
        base_config = super().get_config()
        return {**base_config, **config}
        
# LSTM model
class LSTMModel:
     
    def __init__(self, model=None):
        super().__init__()
        self.model = None
        self.is_trained = False
        self.df_train = None
        # for production
        self.scaler = None
        self.features = None
        self.window = None
        
    def build_model(self, window, sz):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(window, sz)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50,return_sequences=False))
        model.add(Dropout(0.1))
        model.add(Dense(units=1, activation='relu'))
        model.compile(loss=smape(), optimizer="adam", metrics=['mae'])
        self.model = model
       
    # data labeling - generate column RUL
    def label_RUL(self, df):
        rul = pd.DataFrame(df.groupby('id')['time_cycles'].max()).reset_index()
        rul.columns = ['id', 'max']
        df = df.merge(rul, on=['id'], how='left')
        df['RUL'] = df['max'] - df['time_cycles']
        df.drop('max', axis=1, inplace=True)
        return df
        
    # Standard scalar normalization
    def normalize_data(self, df_train):
        cols_normalize = df_train.columns.difference(['id','time_cycles','RUL'])
        scaler = StandardScaler()
        # train data normalization
        scaler.fit(df_train[cols_normalize])
        norm_train_df = pd.DataFrame(scaler.transform(df_train[cols_normalize]),
                                     columns=cols_normalize,
                                     index=df_train.index)
        join_df = df_train[df_train.columns.difference(cols_normalize)].join(norm_train_df)
        train_df = join_df.reindex(columns = df_train.columns)
        
        self.scaler = scaler # saving the scaler for production
        return train_df
    
    # function to reshape the data to lstm
    def get_window(self, id_df, seq_length, seq_cols):
        """
            function to prepare train data into (samples, time steps, features)
            id_df = train dataframe
            seq_length = look back period
            seq_cols = feature columns
        """
        data_array = id_df[seq_cols].values
        num_elements = data_array.shape[0]
        lstm_array=[]

        for start, stop in zip(range(0, num_elements-seq_length+1), range(seq_length, num_elements+1)):
            lstm_array.append(data_array[start:stop, :])

        return np.array(lstm_array)

    # function to reshape the label to lstm
    def gen_target(self, id_df, seq_length, label):
        data_array = id_df[label].values
        num_elements = data_array.shape[0]
        return data_array[seq_length-1:num_elements+1]
    
    # Data ETL
    def etl(self, train_path):
        index_names = ['id', 'time_cycles']
        setting_names = ['setting_1', 'setting_2', 'setting_3']
        sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
        col_names = index_names + setting_names + sensor_names
        
        try:
            df_train = pd.read_csv(train_path,sep='\s+',header=None,index_col=False,names=col_names)
        except:
            print("No train file was found or it's structure is not as expected")
        
        df_train = df_train.sort_values(['id','time_cycles'])
        df_train = self.label_RUL(df_train)
        
        # drop the constants features and the settings based on the EDA
        list_columns_droped = ['setting_1', 'setting_2', 'setting_3', 's_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
        df_train.drop(columns=list_columns_droped, inplace=True)
        
        # data normalization
        df_train = self.normalize_data(df_train)
        
        self.df_train = df_train
        print("ETL completed!")
    
        
    def train(self, x_train, y_train):
    	if self.is_trained:
    	    raise Exception("Model should not be a trained model!")
    	else:
    	    self.model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1)
    	self.is_trained = True
    	print("Train completed!")
    	
    	
    def model_train(self, train_path, looking_back):
        try:
            self.etl(train_path)
        except:
            print("An error occured during ETL!")
            
        feature_columns = self.df_train.columns.difference(['id','time_cycles','RUL']).tolist()
        self.features = feature_columns
        self.window   = looking_back
        
        x_train=np.concatenate(list(list(self.get_window(self.df_train[self.df_train['id']==unit], looking_back, feature_columns)) 
                               for unit in self.df_train['id'].unique()))
        y_train=np.concatenate(list(list(self.gen_target(self.df_train[self.df_train['id']==unit], looking_back, "RUL")) 
                               for unit in self.df_train['id'].unique()))
        
        # create model
        self.build_model(looking_back, len(feature_columns))
                
        # train model
        self.train(x_train, y_train)
        
    def save(self):
        if not self.is_trained:
            raise Exception("Model is not trained yet!")
        else:
            time = str(datetime.now())[:19]
            file_path = f"./trained_models/lstm_pipeline_w5_v2_{time}.pkl".replace(" ","_").replace(":","_")
            try:
                with open(file_path, 'wb') as file:
                    pickle.dump(self, file)
                    print(f"ModelPipeline saved to {file_path}")
            except:
                print("An error occured and the model couldn't be saved!")
                
