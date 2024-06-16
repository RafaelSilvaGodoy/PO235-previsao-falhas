import os
import numpy as np
import pandas as pd
import pickle

from datetime import datetime
from scipy import optimize
from glob import glob
# Importing required modules from scikit-learn for normalization
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA

# Exponential Degradation class
class ExponentialPipeline:

        def __init__(self, model=None):
                self.model = None
                self.is_trained = False
                self.df = None
                self.results = []
                self.threshold = None
                self.exp_parameters_df = None
                # for production
                self.scaler = None
                self.features = None

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
                
        def pca_data(self, df, feats):
		"""
		Performs Principal Component Analysis (PCA) on the given dataframe and returns a dataframe with the top 3 principal components.

		This function reduces the dimensionality of the input features using PCA, retaining the top 3 principal components. 
		It returns a new dataframe that includes these principal components along with the original 'RUL', 'id', and 'time_cycles' columns.

		Args:
		df (pd.DataFrame): The input dataframe containing the data to be transformed.
		feats (list of str): The list of feature column names to be included in the PCA transformation.

		Returns:
		pd.DataFrame: A new dataframe containing the top 3 principal components along with the 'RUL', 'id', and 'time_cycles' columns.
	    	"""
                pca = PCA(n_components=3)
                pca_data = pca.fit_transform(df[feats])
                pca_df = pd.DataFrame(pca_data, columns = ['pc1', 'pc2', 'pc3'])
                pca_df['RUL'] = df['RUL']
                pca_df['id']  = df['id']
                pca_df['time_cycles'] = df['time_cycles']
                return pca_df
                
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
                    
                    
        def exp_degradation(self, parameters, cycle):
		"""
		Calculates the exponential degradation function.

		Args:
		parameters (list of float): List containing the parameters [phi, theta, beta].
		cycle (int or float): The cycle number.

		Returns:
		float: The calculated degradation value.
		"""
                phi = parameters[0]
                theta = parameters[1]
                beta = parameters[2]

                ht = phi + theta * np.exp(beta * cycle)
                return ht

        def residuals (self, parameters, data, y_observed, func):
		"""
		Calculates the residuals between observed and predicted values.

		Args:
		parameters (list of float): List containing the parameters to be optimized.
		data (array-like): The input data for the degradation function.
		y_observed (array-like): The observed data.
		func (function): The degradation function to be used for prediction.

		Returns:
		array: The residuals between observed and predicted values.
		"""
                return func(parameters, data) - y_observed


        def exp_parameters(self, df):
		"""
		Estimates the parameters of the exponential degradation model for each unique id in the dataframe.

		Args:
		df (pd.DataFrame): DataFrame containing the data with columns 'id', 'pc1', and 'time_cycles'.

		Returns:
		pd.DataFrame: DataFrame containing the estimated parameters 'phi', 'theta', and 'beta' for each id.
		"""
                exp_parameters_df = dict(id=list(),phi=list(),theta=list(),beta=list())
                ids = df['id'].unique()
                param_0 = [-1, 0.01, 0.01]
                for id in ids:
                        ht = df.pc1[df.id == id]
                        time_cycles = df.time_cycles[df.id == id]

                        OptimizeResult = optimize.least_squares(self.residuals, param_0, args = (time_cycles, ht, self.exp_degradation))
                        phi, theta, beta = OptimizeResult.x

                        exp_parameters_df['id'].append(id)
                        exp_parameters_df['phi'].append(phi)
                        exp_parameters_df['theta'].append(theta)
                        exp_parameters_df['beta'].append(beta)

                exp_parameters_df = pd.DataFrame(exp_parameters_df)

                return exp_parameters_df
                
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
	        
	def train(self, df, debug=False):
		"""
		Trains the exponential degradation model by estimating the parameters for the training data.

		Returns:
		pd.DataFrame: DataFrame containing the estimated parameters 'phi', 'theta', and 'beta' for the training data.
		"""
		try:
                      self.etl(df, debug)
	        except:
	              raise Exception("An error occurred during ETL!")
	        
	        feature_columns = self.df.columns.difference(['id','time_cycles','RUL']).tolist()
                self.features = feature_columns
                
                # data normalization
                self.df = self.normalize_data(self.df)
	        
	        # PCA
	        self.df = self.pca_data(self.df.reset_index(drop=True), self.features)
	        
                self.exp_parameters_df = self.exp_parameters(self.df)
                self.threshold =  self.df_train.pc1[self.df_train.RUL == 0].mean()
                self.is_trained = True
                if debug:
	             print("Train completed!")
	             
	def predict(self, df, debug=False):
	        """
		Predicts the Remaining Useful Life (RUL) using the exponential degradation model.

		Args:
		df (pd.DataFrame): The test data.

		Returns:
		list of float: The predicted RUL for each test instance.
		"""
		if not self.is_trained:
                        raise Exception("Model is not trained yet!")
                try:
	                self.etl(df, debug)
	        except:
	                raise Exception("An error occurred during ETL!")
	        
	        # PCA
	        self.df = self.pca_data(self.df.reset_index(drop=True), self.features)
	        
	        X = self.df[feature_columns]
		y_true = self.df['RUL']
		
		x_test = self.scaler.transform(X)
	        
	        phi_vals = self.exp_parameters_df.phi
                theta_vals = self.exp_parameters_df.theta
                beta_vals = self.exp_parameters_df.beta
                
                param_1 = [phi_vals.mean(), theta_vals.mean(), beta_vals.mean()]

                lb = 25
                ub = 75
                phi_bounds = [np.percentile(phi_vals, lb), np.percentile(phi_vals, ub)]
                theta_bounds = [np.percentile(theta_vals, lb), np.percentile(theta_vals, ub)]
                beta_bounds = [np.percentile(beta_vals, lb), np.percentile(beta_vals, ub)]
                
                bounds = ([phi_bounds[0], theta_bounds[0], beta_bounds[0]], [phi_bounds[1], theta_bounds[1], beta_bounds[1]])

                y_pred = []

                for i in x_test.id.unique():
                          for x in range(len(x_test[x_test['id'] == i].time_cycles)):
                                ht = x_test.pc1[x_test.id == i][0:x+1]
                                cycle = x_test.time_cycles[x_test.id == i][0:x+1]

                                OptimizeResult = optimize.least_squares(self.residuals, param_1, bounds=bounds,
                                                  args = (cycle, ht, self.exp_degradation))
                                phi, theta, beta = OptimizeResult.x
                                total_cycles = np.log((self.threshold - phi) / theta) / beta
                                RUL = total_cycles - cycle.max()

                                y_pred.append(RUL)
                                
                return y_true, y_pred
                
        def report(self, data_path="./dataset/split_folders/", debug=False):
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
		
		                   self.train(path_train, debug)
		                   y_true, y_pred = self.predict(path_test, debug)
		
		                   self.evaluation.append(self.model_evaluation(y_true, y_pred))
		                   print(f"repeat_{pasta}/train_{folder}.csv evaluation completed!")
		               except:
		                   raise Exception(f"It was not able to evaluate repeat_{pasta}/train_{folder}.csv")
		                   
        def save(self):
        	"""
		Saves the results to a file.
		"""
                time = str(datetime.now())[:19]
                file_path = f"./models/exponential_pipeline_{time}.pkl".replace(" ","_").replace(":","_")
                try:
                        with open(file_path, 'wb') as file:
                                pickle.dump(self.results, file)
                                print(f"ModelPipeline saved to {file_path}")
                except:
                        print("An error occured and the model couldn't be saved!")
                        
if __name__ == "__main__":
 
    pipeline = ExponentialPipeline()
    pipeline.model_validation()
    pipeline.save()
