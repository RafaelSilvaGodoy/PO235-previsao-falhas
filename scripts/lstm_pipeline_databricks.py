import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from glob import glob
from json import dumps
from datetime import datetime
from keras import backend as k

# Importing required modules from scikit-learn for data splitting and normalization
#to split the train set in train and val sets using the id
from sklearn.model_selection import GroupShuffleSplit		
from sklearn.preprocessing import StandardScaler 		#to normalize data

# Importing modules from Keras for building the deep learning model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.callbacks import History, EarlyStopping


# Custom loss function for training: Symmetric Mean Absolute Percentage Error (SMAPE)
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
        
# Class LSTM model
class LSTMModel:
     
    def __init__(self, model=None):
        super().__init__()
        self.model = None
        self.is_trained = False
        self.df = None
        self.window = None
        self.evaluation = []
        # for production
        self.scaler = None
        self.features = None
        self.window = None
        
    def build_model(self, window, sz):
        """
        Builds the LSTM model architecture with specified input shape.
        Args:
        window: int - The number of time steps to look back.
        sz: int - The number of features.
        """
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(window, sz)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50,return_sequences=False))
        model.add(Dropout(0.1))
        model.add(Dense(units=1, activation='relu'))
        model.compile(loss=smape(), optimizer="adam", metrics=['mae'])
        self.model = model
       
    def label_RUL(self, df):
        """
        Adds a Remaining Useful Life (RUL) column to the dataframe.
        Args:
        df: DataFrame - The input dataframe.
        Returns:
        DataFrame - The dataframe with the added RUL column.
        """
        rul = pd.DataFrame(df.groupby('id')['time_cycles'].max()).reset_index()
        rul.columns = ['id', 'max']
        df = df.merge(rul, on=['id'], how='left')
        df['RUL'] = df['max'] - df['time_cycles']
        df.drop('max', axis=1, inplace=True)
        return df
        
    def normalize_data(self, df):
        """
        Normalizes the training data using StandardScaler.
        Args:
        df: DataFrame - The training dataframe.
        Returns:
        DataFrame - The normalized training dataframe.
        """
        cols_normalize = df.columns.difference(['id','time_cycles','RUL'])
        scaler = StandardScaler()
        # train data normalization
        scaler.fit(df[cols_normalize])
        norm_df = pd.DataFrame(scaler.transform(df[cols_normalize]),
                                     columns=cols_normalize,
                                     index=df.index)
        join_df = df[df.columns.difference(cols_normalize)].join(norm_df)
        train_df = join_df.reindex(columns = df.columns)
        
        self.scaler = scaler # saving the scaler for production
        return train_df
    
    # function to reshape the data to lstm
    def get_window(self, id_df, seq_length, seq_cols):
        """
        Prepares the data into (samples, time steps, features) format.
        Args:
        id_df: DataFrame - The input dataframe.
        seq_length: int - The sequence length (look-back period).
        seq_cols: list - The list of feature columns.
        Returns:
        np.array - The reshaped data array.
        """
        data_array = id_df[seq_cols].values
        num_elements = data_array.shape[0]
        lstm_array=[]

        for start, stop in zip(range(0, num_elements-seq_length+1), range(seq_length, num_elements+1)):
            lstm_array.append(data_array[start:stop, :])

        return np.array(lstm_array)

    # function to reshape the label to lstm
    def gen_target(self, id_df, seq_length, label):
        """
        Prepares the labels for the LSTM model.
        Args:
        id_df: DataFrame - The input dataframe.
        seq_length: int - The sequence length (look-back period).
        label: str - The label column.
        Returns:
        np.array - The reshaped label array.
        """
        data_array = id_df[label].values
        num_elements = data_array.shape[0]
        return data_array[seq_length-1:num_elements+1]
    
    # Data ETL
    def etl(self, df, debug):
        """
        Extract, Transform, Load (ETL) process for the training data.
        Args:
        df: DataFrame - The input dataframe.
        debug: bool, optional - Flag to print debug information. Defaults to False.
        """
        try:
            df = df.sort_values(['id','time_cycles'])
            df = self.label_RUL(df)
            
            # drop the constants features and the settings based on the EDA
            list_columns_droped = ['setting_1', 'setting_2', 'setting_3', 's_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
            df.drop(columns=list_columns_droped, inplace=True)
        
            self.df = df
            if debug:
               print("ETL completed!")
        except:
            print("An ETL problem occurred!")
    
        
    def train(self, df, looking_back = 5, debug=False):
        """
        Trains the LSTM model.
        Args:
        df: DataFrame - The input dataframe.
        debug (bool, optional): Flag to print debug information. Defaults to False.
        """
        try:
            self.etl(df, debug)
        except:
            raise Exception("An error occurred during ETL!")
	
        feature_columns = self.df.columns.difference(['id','time_cycles','RUL']).tolist()
        self.features = feature_columns
        self.window = looking_back
        
        # data normalization
        self.df = self.normalize_data(self.df)
        
        X = np.concatenate(list(list(self.get_window(self.df[self.df['id']==unit], self.window, self.features)) for unit in self.df['id'].unique()))
        Y = np.concatenate(list(list(self.gen_target(self.df[self.df['id']==unit], self.window, "RUL")) for unit in self.df['id'].unique()))
	
	# create model
        self.build_model(self.window, len(self.features))
	
        self.model.fit(X, Y, epochs=100, batch_size=32, verbose=1)
        self.is_trained = True
        if debug:
            print("Train completed!")
    	
    def predict(self, df, debug=False):
        """
	Makes predictions using the trained LSTM model on data from the given path.

	Args:
	df (DataFrame): Path to the data file.
	debug (bool, optional): Flag to print debug information. Defaults to False.

	Returns:
	tuple: The true and predicted RUL values.

	Raises:
	Exception: If the model is not trained or an error occurs during ETL or prediction.
	"""
        if not self.is_trained:
           raise Exception("Model is not trained yet!")
        try:
           self.etl(df, debug)
        except:
           raise Exception("An error occurred during ETL!")
	   
	# normalize
        norm_df = pd.DataFrame(self.scaler.transform(self.df[self.features]),
                               columns=self.features,
                               index=self.df.index)
        join_df = self.df[self.df.columns.difference(self.features)].join(norm_df)
        self.df = join_df.reindex(columns = self.df.columns)
	   
        #X = np.concatenate(list(list(self.get_window(self.df[self.df['id']==unit], self.window, self.features)) for unit in self.df['id'].unique()))
        #y_true = np.concatenate(list(list(self.gen_target(self.df[self.df['id']==unit], self.window, "RUL")) for unit in new_train['id'].unique()))
        X = self.get_window(self.df, self.window, self.features)
        y_true = self.gen_target(self.df, self.window, "RUL")
	
        y_pred = self.model.predict(X)
        if debug:
           print("Prediction completed!")
        return y_true, y_pred
	   
    def model_evaluation(self, y_true, y_pred):
        """
	Reports the Mean Absolute Error (MAE) for instances where the RUL is 30 cycles or less.

	Args:
	y_true (array-like): The true values.
	y_pred (array-like): The predicted values.

	Returns:
	float: The calculated MAE.
	"""
        inputs = (y_pred <= 30, y_true <= 30)
        idx = np.any(inputs, axis=0)
        filtered_y_pred = y_pred[idx]
        filtered_y_true = y_true[idx]
        return mean_absolute_error(filtered_y_true, filtered_y_pred)
	
    def report(self, data_path="./dataset/split_folders/", looking_back = 5, debug=False):
        """
        Reports the evaluation results for the model using cross-validation on multiple datasets.

        Args:
        data_path (str, optional): Path to the directory containing the datasets. Defaults to "./dataset/split_folders/".
        debug (bool, optional): Flag to print debug information. Defaults to False.

        Raises:
        Exception: If an error occurs during training or evaluation.
        """
        for pasta in range(0, 10):
            for folder in range(0, 10):
                try:
                    path_train = data_path + f"repeat_{pasta}/train_{folder}.csv"
                    path_test = data_path + f"repeat_{pasta}/test_{folder}.csv"
		
                    self.train(path_train, looking_back, debug)
                    y_true, y_pred = self.predict(path_test, debug)
		
                    self.evaluation.append(self.model_evaluation(y_true, y_pred))
                    print(f"repeat_{pasta}/train_{folder}.csv evaluation completed!")
                except:
                    raise Exception(f"It was not able to evaluate repeat_{pasta}/train_{folder}.csv")
        
    def save(self):
        """
        Saves the trained model to a file.
        """
        if not self.is_trained:
            raise Exception("Model is not trained yet!")
        else:
            time = str(datetime.now())[:19]
            file_path = f"./trained_models/lstm_pipeline_w5_v2_{time}.pkl".replace(" ","_").replace(":","_")
            try:
                with open(file_path, 'wb') as file:
                    pickle.dump(self.evaluation, file)
                    print(f"ModelPipeline saved to {file_path}")
            except:
                print("An error occured and the model couldn't be saved!")
                
if __name__ == "__main__":
    
    pipeline = LSTMModel()
    pipeline.report(looking_back = 5)
    print(len(pipeline.evaluation))
    pipeline.save()
    
