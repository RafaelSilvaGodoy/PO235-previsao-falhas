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
                self.df_train = None
                self.df_test  = None
                self.results = []
                self.threshold = None

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


        def normalize_data(self, df_train, df_test):
		"""
		Normalizes the training data using StandardScaler.
		Args:
		df_train: DataFrame - The training dataframe.
		df_test: DataFrame - The test dataframe.
		Returns:
		DataFrame - The normalized training dataframe.
		"""
                cols_normalize = df_train.columns.difference(['id','time_cycles','RUL'])
                scaler = StandardScaler()
                # train data normalization
                scaler.fit(df_train[cols_normalize])
                norm_train_df = pd.DataFrame(scaler.transform(df_train[cols_normalize]),
                                     columns=cols_normalize, index=df_train.index)
                join_df = df_train[df_train.columns.difference(cols_normalize)].join(norm_train_df)
                train_df = join_df.reindex(columns = df_train.columns)
                # test data normalization
                norm_test_df  = pd.DataFrame(scaler.transform(df_test[cols_normalize]),
                                    columns=cols_normalize, index=df_test.index)
                join_dfT = df_test[df_test.columns.difference(cols_normalize)].join(norm_test_df)
                test_df = join_dfT.reindex(columns = df_test.columns)
                return train_df, test_df

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
        def etl(self, train_path, test_path):
		"""
		Extract, Transform, Load (ETL) process for the data.
		Args:
		train_path: str - The path to the training data file.
		test_path: str - The path to the test data file.
		"""
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
        
                # drop the constants features and the settings
                select_features = ['id','time_cycles','s_3','s_2','s_17','s_21','s_13','s_8','s_7','s_15','s_20','s_9','s_11','s_12','s_14','RUL']
                df_train = df_train[select_features]       		
                df_test = df_test[select_features]  
        
                # data normalization
                df_train, df_test = self.normalize_data(df_train, df_test)   
        	
                # pca to fuse the features
                feats = ['s_3','s_2','s_17','s_21','s_13','s_8','s_7','s_15','s_20','s_9','s_11','s_12','s_14']
                pca_train = self.pca_data(df_train.reset_index(drop=True), feats)
                pca_test  = self.pca_data(df_test.reset_index(drop=True), feats)

                self.df_train = pca_train
                self.df_test  = pca_test
                print("ETL completed!")


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

        def report(self, y_true, y_pred):
		"""
		Reports the Mean Absolute Error (MAE) for instances where the RUL is 30 cycles or less.

		Args:
		y_true (array-like): The true values.
		y_pred (array-like): The predicted values.

		Returns:
		float: The calculated MAE.
		"""
                inputs = (y_pred <= 30, y_true <= 30)
                idx = np.any(inputs, axis = 0)
                y_pred = y_pred[idx]
                y_true = y_true[idx]
                return mean_absolute_error(y_true, y_pred)

        def predict(self, x_test, exp_parameters_df):
		"""
		Predicts the Remaining Useful Life (RUL) using the exponential degradation model.

		Args:
		x_test (pd.DataFrame): The test data.
		exp_parameters_df (pd.DataFrame): DataFrame containing the estimated parameters 'phi', 'theta', and 'beta' for each id.

		Returns:
		list of float: The predicted RUL for each test instance.
		"""
                if not self.is_trained:
                        raise Exception("Model is not trained yet!")
                else:
                        phi_vals = exp_parameters_df.phi
                        theta_vals = exp_parameters_df.theta
                        beta_vals = exp_parameters_df.beta

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
                        return y_pred

        def train(self):
		"""
		Trains the exponential degradation model by estimating the parameters for the training data.

		Returns:
		pd.DataFrame: DataFrame containing the estimated parameters 'phi', 'theta', and 'beta' for the training data.
		"""
                exp_parameters_df = self.exp_parameters(self.df_train)
                self.threshold =  self.df_train.pc1[self.df_train.RUL == 0].mean()
                self.is_trained = True
                return exp_parameters_df

        def model_validation(self, fold_path):
		"""
		Performs model validation using 10-fold cross-validation.

		Args:
		fold_path (list of str): List containing the paths to the fold directories.

		Returns:
		None
		"""
                 for repeat in fold_path:
                       for i in range(10):
                             train_path = repeat+'/train_'+str(i)+'.csv'
                             test_path  = repeat+'/test_'+str(i)+'.csv'
                             try:
                                       self.etl(train_path, test_path)
                             except:
                                       print("An error occured during ETL!")

                             x_test = self.df_test.drop(columns = ['RUL'])
                             y_true = self.df_test['RUL']
                             exp_parameters_df = self.train()
                             y_pred = self.predict(x_test, exp_parameters_df)
 
                             # test evaluation
                             mae = self.report(np.array(y_true), np.array(y_pred))
                             print(mae)

                             self.results.append(mae)


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

    # fold with the 10 x 10-fold with different seeds
    fold_path = glob(f'./dataset/split_folders/*')
    
    pipeline = ExponentialPipeline()
    pipeline.model_validation(fold_path)
    pipeline.save()
