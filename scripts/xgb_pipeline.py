import numpy as np
import pandas as pd
from datetime import datetime
import pickle
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

class XGBPipeline:

    def __init__(self, model=None):
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        self.is_trained = False
        self.df = None

    def etl(self, data_path):

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
        try:
            df = pd.read_csv(data_path,sep='\s+',header=None,index_col=False,names=col_names).sort_values(['id','time_cycles'])
        except:
            print("No file was found or it's structure is not as expected")

        rul = pd.DataFrame(df.groupby('id')['time_cycles'].max()).reset_index()
        rul.columns = ['id', 'max']
        df = df.merge(rul, on=['id'], how='left')
        df['RUL'] = df['max'] - df['time_cycles']
        df.drop('max', axis=1, inplace=True)

        list_columns_droped = ['time_cycles','setting_3','(Fan inlet temperature) (◦R)','(Engine pressure ratio(P50/P2)',
                    '(Required fan speed)','(Required fan conversion speed)']

        df.drop(columns=list_columns_droped, inplace=True)
        self.df = df
        print("ETL completed!")

    def model_evaluation(self):
        return 0
    
    def train(self, data_path):
        try:
            pipeline.etl(data_path)
        except:
            print("An error occured during ETL!")

        feature_columns = self.df.columns.tolist()
        feature_columns.remove("RUL")
        feature_columns.remove("id")

        X = self.df[feature_columns]
        y = self.df['RUL']
        groups = self.df['id']

        self.model.fit(X, y)
        self.is_trained = True
        print("Train completed!")

    def predict(self, X):
        if not self.is_trained:
            raise Exception("Model is not trained yet!")
        return self.model.predict(X)

    def report(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)

    def save(self):
        time = str(datetime.now())[:19]
        file_path = f"./models/xgb_pipeline_{time}.pkl".replace(" ","_").replace(":","_")
        try:
            with open(file_path, 'wb') as file:
                pickle.dump(self, file)
                print(f"ModelPipeline saved to {file_path}")
        except:
            print("An error occured and the model couldn't be saved!")




if __name__ == "__main__":

    pipeline = XGBPipeline()
    pipeline.train("./dataset/CMaps/train_FD001.txt")
    #y_pred = pipeline.predict(X_test)
    #pipeline.report(y_test, y_pred)
    pipeline.save()
