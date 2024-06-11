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

    def etl(self, data_path,debug):
    	"""
        Extracts, transforms, and loads data from the given path.

        Args:
        data_path (str): Path to the data file.
        debug (bool, optional): Flag to print debug information. Defaults to False.

        Raises:
        Exception: If the file is not found or the structure is not as expected.
        """

        index_names = ['id', 'time_cycles']
        setting_names = ['setting_1', 'setting_2', 'setting_3']
        sensor_names = ["(Fan inlet temperature) (◦R)","(LPC outlet temperature) (◦R)",
                        "(HPC outlet temperature) (◦R)","(LPT outlet temperature) (◦R)",
                        "(Fan inlet Pressure) (psia)","(bypass-duct pressure) (psia)",
                        "(HPC outlet pressure) (psia)","(Physical fan speed) (rpm)",
                        "(Physical core speed) (rpm)","(Engine pressure ratio(P50/P2)",
                        "(HPC outlet Static pressure) (psia)","(Ratio of fuel flow to Ps30) (pps/psia)",
                        "(Corrected fan speed) (rpm)", "(Corrected core speed) (rpm)",
                        "(Bypass Ratio) ","(Burner fuel-air ratio)","(Bleed Enthalpy)",
                        "(Required fan speed)","(Required fan conversion speed)","(High-pressure turbines Cool air flow)",
                        "(Low-pressure turbines Cool air flow)"]

        col_names = index_names + setting_names + sensor_names
        old_name = ["id","time_cycles","setting_1","setting_2",
                    "setting_3","s_1","s_2","s_3","s_4","s_5",
                    "s_6","s_7","s_8","s_9","s_10","s_11","s_12","s_13",
                    "s_14","s_15","s_16","s_17","s_18","s_19","s_20","s_21"]
        cols_dict = {}

        for name_old, name_new in zip(old_name, col_names):
            cols_dict[name_old] = name_new
        try:
            df = pd.read_csv(data_path,sep='\s+',index_col=0)
            df = df.rename(columns=cols_dict).sort_values(['id','time_cycles'])
            df.reset_index(drop=True,inplace=True)
        except:
            raise Exception("No file was found or it's structure is not as expected")

        rul = pd.DataFrame(df.groupby('id')['time_cycles'].max()).reset_index()
        rul.columns = ['id', 'max']
        df = df.merge(rul, on=['id'], how='left')
        df['RUL'] = df['max'] - df['time_cycles']
        df.drop('max', axis=1, inplace=True)
        
        list_columns_droped = ['time_cycles','setting_3','(Fan inlet temperature) (◦R)','(Engine pressure ratio(P50/P2)',
                    '(Required fan speed)','(Required fan conversion speed)']

        df.drop(columns=list_columns_droped, inplace=True)
        self.df = df
        if debug:
            print("ETL completed!")
    
    def train(self, data_path, debug):
    	"""
        Trains the XGBoost model using the data from the given path.

        Args:
        data_path (str): Path to the data file.
        debug (bool, optional): Flag to print debug information. Defaults to False.

        Raises:
        Exception: If an error occurs during ETL or training.
        """
        try:
            pipeline.etl(data_path, debug)
        except:
            raise Exception("An error occured during ETL!")

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

    def predict(self, data_path, debug):
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
            pipeline.etl(data_path, debug)
        except:
            raise Exception("An error occured during ETL!")

        feature_columns = self.df.columns.tolist()
        feature_columns.remove("RUL")
        feature_columns.remove("id")

        X = self.df[feature_columns]
        y_true = self.df['RUL']

        X_ss = self.scaler.transform(X)
        y_pred = self.model.predict(X_ss)
        if debug:
            print("Prediction completed!")
        return y_true,y_pred
    
    def model_evaluation(self,y_true,y_pred):
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
        filtered_y_pred = y_pred[idx]
        filtered_y_true = y_true[idx]
        return mean_absolute_error(filtered_y_true, filtered_y_pred)

    
    def report(self, data_path="./dataset/split_folders/", debug = False):
	"""
        Reports the evaluation results for the model using cross-validation on multiple datasets.

        Args:
        data_path (str, optional): Path to the directory containing the datasets. Defaults to "./dataset/split_folders/".
        debug (bool, optional): Flag to print debug information. Defaults to False.

        Raises:
        Exception: If an error occurs during training or evaluation.
        """
        for pasta in range(0,10):
            for folder in range(0,10):
                try:
                    path_train = data_path + f"repeat_{pasta}/train_{folder}.csv"
                    path_test = data_path + f"repeat_{pasta}/test_{folder}.csv"
                    
                    pipeline.train(path_train,debug)
                    y_true,y_pred = pipeline.predict(path_test,debug)
                    
                    self.evaluation.append(self.model_evaluation(y_true,y_pred))
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
        file_path = f"./models/xgb_pipeline_{time}.pkl".replace(" ","_").replace(":","_")
        try:
            with open(file_path, 'wb') as file:
                pickle.dump(self.evaluation, file)
                print(f"ModelPipeline saved to {file_path}")
        except:
            print("An error occured and the model couldn't be saved!")




if __name__ == "__main__":

    pipeline = XGBPipeline()

    pipeline.report()
    print(len(pipeline.evaluation))
    pipeline.save()

