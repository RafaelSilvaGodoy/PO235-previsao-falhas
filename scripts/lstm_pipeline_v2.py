"""
   This code save the results of 10 repetitions of 10-folds of different seeds
   for validation
"""
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
from sklearn.metrics import mean_absolute_error                 #to evaluate the results

#for deep learning
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.callbacks import History, EarlyStopping

class LSTMPipeline:
     
    def __init__(self, model=None):
        super().__init__()
        self.model = None
        self.is_trained = False
        self.df_train = None
        self.df_test  = None
        self.results = []
        
    def build_model(self, window, sz):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(window, sz)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50,return_sequences=False))
        model.add(Dropout(0.1))
        model.add(Dense(units=1, activation='relu'))
        model.compile(loss=self.smape, optimizer="adam", metrics=['mae'])
        self.model = model
        self.is_trained = False
       
    # data labeling - generate column RUL
    def label_RUL(self, df):
        rul = pd.DataFrame(df.groupby('id')['time_cycles'].max()).reset_index()
        rul.columns = ['id', 'max']
        df = df.merge(rul, on=['id'], how='left')
        df['RUL'] = df['max'] - df['time_cycles']
        df.drop('max', axis=1, inplace=True)
        return df
        
    # Standard scalar normalization
    def normalize_data(self, df_train, df_test):
        cols_normalize = df_train.columns.difference(['id','time_cycles','RUL'])
        scaler = StandardScaler()
        # train data normalization
        scaler.fit(df_train[cols_normalize])
        norm_train_df = pd.DataFrame(scaler.transform(df_train[cols_normalize]),
                                     columns=cols_normalize,
                                     index=df_train.index)
        join_df = df_train[df_train.columns.difference(cols_normalize)].join(norm_train_df)
        train_df = join_df.reindex(columns = df_train.columns)
        # test data normalization
        norm_test_df  = pd.DataFrame(scaler.transform(df_test[cols_normalize]),
                                     columns=cols_normalize,
                                     index=df_test.index)
        join_dfT = df_test[df_test.columns.difference(cols_normalize)].join(norm_test_df)
        test_df = join_dfT.reindex(columns = df_test.columns)
        return train_df, test_df
    
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
    def etl(self, train_path, test_path):
        try:
            df_train = pd.read_csv(train_path, sep=' ', index_col=0)
        except:
            print("No train file was found or it's structure is not as expected")
        try:
            df_test  = pd.read_csv(test_path, sep=' ', index_col=0)
        except:
            print("No test file was found or it's structure is not as expected")
           
        df_train = df_train.sort_values(['id','time_cycles'])
        df_train = self.label_RUL(df_train)
        df_test  = df_test.sort_values(['id','time_cycles'])
        df_test  = self.label_RUL(df_test)
        
        # drop the constants features and the settings based on the EDA
        list_columns_droped = ['setting_1', 'setting_2', 'setting_3', 's_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
       
        df_train.drop(columns=list_columns_droped, inplace=True)
        df_test.drop(columns=list_columns_droped, inplace=True)
        
        
        # data normalization
        df_train, df_test = self.normalize_data(df_train, df_test)
        
        self.df_train = df_train
        self.df_test  = df_test
        print("ETL completed!")
    
    # training loss function
    def smape(self, y_true, y_pred):
        epsilon=1e-8
        y_true = tf.cast(y_true, tf.float32)
        numerator = abs(y_pred - y_true)
        denominator = (abs(y_true) + abs(y_pred)) / 2.0 + epsilon
        smape_value = k.mean(numerator / denominator) * 100
        return smape_value
    
    # training function
    def train(self, x_train, y_train, x_val, y_val):
    	if self.is_trained:
    	    raise Exception("Model should not be a trained model!")
    	else:
            self.model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val), verbose=1, 
    	            callbacks = [History(), EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')])
    	self.is_trained = True
    	print("Train completed!")
        
    # predict function
    def predict(self, x_test):
        if not self.is_trained:
            raise Exception("Model is not trained yet!")
        else:
            return self.model.predict(x_test, verbose=0)        
    
    # MAE for the instances where the RUL is equal or under 30 cycles
    def report(self, y_true, y_pred):
        thr = 30  # RUL threshold for evaluation
        inputs = (y_pred <= thr, y_true <= thr)
        idx = np.any(inputs, axis=0)
        y_pred = y_pred[idx]
        y_true = y_true[idx]
        return mean_absolute_error(y_true, y_pred)
    
    # 10 x 10-folds
    def model_validation(self, fold_path, looking_back):
        for repeat in fold_path:
            for i in range(10):
                train_path = repeat+'/train_'+str(i)+'.csv'
                test_path  = repeat+'/test_'+str(i)+'.csv'
                try:
                   self.etl(train_path, test_path)
                except:
                   print("An error occured during ETL!")
                   
                feature_columns = self.df_train.columns.tolist()
                
                # generate the validation set based on the id
                gss = GroupShuffleSplit(n_splits=1, train_size=.8, random_state=42)
                for i, (train_index, val_index) in enumerate(gss.split(self.df_train, self.df_train['RUL'], self.df_train['id'])):
                    df_val    = self.df_train.iloc[val_index]
                    new_train = self.df_train.iloc[train_index]
                
                x_train=np.concatenate(list(list(self.get_window(new_train[new_train['id']==unit], looking_back, feature_columns)) for unit in new_train['id'].unique()))
                x_test =np.concatenate(list(list(self.get_window(self.df_test[self.df_test['id']==unit], looking_back, feature_columns)) for unit in self.df_test['id'].unique()))
                x_val  =np.concatenate(list(list(self.get_window(df_val[df_val['id']==unit], looking_back, feature_columns)) for unit in df_val['id'].unique()))

                #generate target of train
                y_train = np.concatenate(list(list(self.gen_target(new_train[new_train['id']==unit], looking_back, "RUL")) for unit in new_train['id'].unique()))
                y_test  = np.concatenate(list(list(self.gen_target(self.df_test[self.df_test['id']==unit], looking_back, "RUL")) for unit in self.df_test['id'].unique()))
                y_val   = np.concatenate(list(list(self.gen_target(df_val[df_val['id']==unit], looking_back, "RUL")) for unit in df_val['id'].unique()))
                
                # create model
                self.build_model(looking_back, len(feature_columns))
                
                # train model
                self.train(x_train, y_train, x_val, y_val)
                
                # test predict 
                y_pred = self.predict(x_test)
        	
        	# test evaluation
                mae = self.report(y_test, y_pred[:,0])
                print(mae)
                
                self.results.append(mae)
 
    # save the results    
    def save(self):
        time = str(datetime.now())[:19]
        file_path = f"./models/lstm_pipeline_v2_{time}.pkl".replace(" ","_").replace(":","_")
        try:
            with open(file_path, 'wb') as file:
                pickle.dump(self.results, file)
                print(f"ModelPipeline saved to {file_path}")
        except:
            print("An error occured and the model couldn't be saved!")
            
if __name__ == "__main__":
    # observation window
    looking_back = 5
    # fold with the 10 x 10-fold with different seeds
    fold_path = glob(f'./dataset/split_folders/*')
    
    pipeline = LSTMPipeline()
    pipeline.model_validation(fold_path, looking_back)
    pipeline.save()
