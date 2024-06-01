import os
import numpy as np
import pandas as pd
import pickle

from datetime import datetime
from scipy import optimize
from glob import glob
from sklearn.preprocessing import StandardScaler #to normalize data
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA

class ExponentialPipeline:

        def __init__(self, model=None):
                self.model = None
                self.is_trained = False
                self.df_train = None
                self.df_test  = None
                self.results = []
                self.threshold = None

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
                                     columns=cols_normalize, index=df_train.index)
                join_df = df_train[df_train.columns.difference(cols_normalize)].join(norm_train_df)
                train_df = join_df.reindex(columns = df_train.columns)
                # test data normalization
                norm_test_df  = pd.DataFrame(scaler.transform(df_test[cols_normalize]),
                                    columns=cols_normalize, index=df_test.index)
                join_dfT = df_test[df_test.columns.difference(cols_normalize)].join(norm_test_df)
                test_df = join_dfT.reindex(columns = df_test.columns)
                return train_df, test_df

        # principal component analysis
        def pca_data(self, df, feats):
                pca = PCA(n_components=3)
                pca_data = pca.fit_transform(df[feats])
                pca_df = pd.DataFrame(pca_data, columns = ['pc1', 'pc2', 'pc3'])
                pca_df['RUL'] = df['RUL']
                pca_df['id']  = df['id']
                pca_df['time_cycles'] = df['time_cycles']
                return pca_df


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
        
                # drop the constants features and the settings
                select_features = ['id','time_cycles','s_9','s_11','s_12','s_14','RUL']
                df_train = df_train[select_features]       		
                df_test = df_test[select_features]  
        
                # data normalization
                df_train, df_test = self.normalize_data(df_train, df_test)   
        	
                # pca
                feats = ['s_9','s_11','s_12','s_14']
                pca_train = self.pca_data(df_train.reset_index(drop=True), feats)
                pca_test  = self.pca_data(df_test.reset_index(drop=True), feats)

                self.df_train = pca_train
                self.df_test  = pca_test
                print("ETL completed!")


        def exp_degradation(self, parameters, cycle):
                phi = parameters[0]
                theta = parameters[1]
                beta = parameters[2]

                ht = phi + theta * np.exp(beta * cycle)
                return ht

        def residuals (self, parameters, data, y_observed, func):
                return func(parameters, data) - y_observed


        def exp_parameters(self, df):
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
                y_pred = y_pred[y_true<=30]
                y_true = y_true[y_true<=30]
                return mean_absolute_error(y_true, y_pred)

        def predict(self, x_test, exp_parameters_df):
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
                exp_parameters_df = self.exp_parameters(self.df_train)
                self.threshold =  self.df_train.pc1[self.df_train.RUL == 0].mean()
                self.is_trained = True
                return exp_parameters_df

        def model_validation(self, fold_path):
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