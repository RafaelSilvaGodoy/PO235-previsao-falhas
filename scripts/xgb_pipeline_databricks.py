import numpy as np
import pandas as pd
from datetime import datetime
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb


class XGBPipeline:
	"""
	A pipeline for data preprocessing, model training, prediction, and evaluation using XGBoost.

	Attributes:
	model (xgb.XGBRegressor): The XGBoost model used for regression.
	is_trained (bool): Flag indicating if the model has been trained.
	df (pd.DataFrame): DataFrame holding the processed data.
	scaler (StandardScaler): Scaler used for standardizing features.
	evaluation (list): List to store evaluation results.
	"""

	def __init__(self, model=None):
		"""
		Initializes the XGBPipeline with default or provided model.

		Args:
		model (xgb.XGBRegressor, optional): Predefined XGBoost model. Defaults to None.
		"""
		self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
		self.is_trained = False
		self.df = None
		self.scaler = None
		self.evaluation = []

	def etl(self, df, debug):
		"""
		Extracts, transforms, and loads data from the given path.

		Args:
		data_path (str): Path to the data file.
		debug (bool, optional): Flag to print debug information. Defaults to False.

		Raises:
		Exception: If the file is not found or the structure is not as expected.
		"""
		try:
			rul = pd.DataFrame(df.groupby('id')['time_cycles'].max()).reset_index()
			rul.columns = ['id', 'max']
			df = df.merge(rul, on=['id'], how='left')
			df['RUL'] = df['max'] - df['time_cycles']
			df.drop('max', axis=1, inplace=True)
            
			list_columns_droped = ['time_cycles', 'setting_3', 's_1', 's_10','s_18', 's_19']

			df.drop(columns=list_columns_droped, inplace=True)
			self.df = df
			if debug:
				print("ETL completed!")
		except:
			print("An ETL problem occurred!")
	
	def train(self, df, debug=False):
		"""
		Trains the XGBoost model using the data from the given path.

		Args:
		data_path (str): Path to the data file.
		debug (bool, optional): Flag to print debug information. Defaults to False.

		Raises:
		Exception: If an error occurs during ETL or training.
		"""
		try:
			self.etl(df, debug)
		except:
			raise Exception("An error occurred during ETL!")

		feature_columns = self.df.columns.tolist()
		feature_columns.remove("RUL")
		feature_columns.remove("id")

		X = self.df[feature_columns]
		y = self.df['RUL']

		scaler = StandardScaler()
		scaler.fit(X)
		self.scaler = scaler

		X_ss = self.scaler.transform(X)

		self.model.fit(X_ss, y)
		self.is_trained = True
		if debug:
			print("Train completed!")

	def predict(self, df, debug=False):
		"""
		Makes predictions using the trained XGBoost model on data from the given path.

		Args:
		data_path (str): Path to the data file.
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

		feature_columns = self.df.columns.tolist()
		feature_columns.remove("RUL")
		feature_columns.remove("id")

		X = self.df[feature_columns]
		y_true = self.df['RUL']

		X_ss = self.scaler.transform(X)
		y_pred = self.model.predict(X_ss)
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
		Saves the evaluation results to a file with a timestamp.

		Raises:
		Exception: If an error occurs during saving.
		"""
		time = str(datetime.now())[:19]
		file_path = f"./models/xgb_pipeline_{time}.pkl".replace(" ", "_").replace(":", "_")
		try:
			with open(file_path, 'wb') as file:
				pickle.dump(self.evaluation, file)
				print(f"ModelPipeline saved to {file_path}")
		except:
			print("An error occurred and the model couldn't be saved!")


if __name__ == "__main__":
	pipeline = XGBPipeline()
	pipeline.report()
	print(len(pipeline.evaluation))
	pipeline.save()
